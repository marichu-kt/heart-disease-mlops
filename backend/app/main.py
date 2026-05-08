from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .metrics import (
    HEART_API_UP,
    HEART_MODEL_INFO,
    HEART_PREDICTION_DURATION_SECONDS,
    HEART_PREDICTION_ERRORS_TOTAL,
    HEART_PREDICTIONS_BY_CLASS_TOTAL,
    HEART_PREDICTIONS_TOTAL,
    PrometheusMiddleware,
    metrics_response,
)
from .model_service import ModelService
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PatientData,
    PredictionResponse,
    VersionResponse,
)


APP_NAME = "Heart Disease MLOps API"
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
APP_ENVIRONMENT = os.getenv("APP_ENVIRONMENT", "local")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
model_service = ModelService()

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if origin.strip()
]

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        logger.info("Starting API and loading model")
        model_service.load_latest_model()
        info = model_service.model_info()
        HEART_MODEL_INFO.labels(
            model_name=info["model_name"],
            version=info["version"],
            algorithm=info["algorithm"],
        ).set(1)
        HEART_API_UP.set(1)
        logger.info(
            "Model ready: model_name=%s model_version=%s algorithm=%s",
            info["model_name"],
            info["version"],
            info["algorithm"],
        )
    except Exception:
        HEART_API_UP.set(0)
        logger.exception("Model startup failed")
        raise
    yield


app = FastAPI(
    title=APP_NAME,
    description="API FastAPI para predicción de enfermedad cardíaca con modelo MLP versionado.",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(PrometheusMiddleware)


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "name": "Heart Disease MLOps API",
        "docs": "/docs",
        "health": "/health",
        "info": "/info",
        "version": "/version",
        "predict": "/predict",
        "batch": "/predict/batch",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    status = "healthy" if model_service.loaded else "unhealthy"
    HEART_API_UP.set(1 if model_service.loaded else 0)
    return HealthResponse(
        status=status,
        model_loaded=model_service.loaded,
        model_version=model_service.version if model_service.loaded else None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    model_name = model_service.metadata.get("model_name", "unavailable")
    model_version = model_service.version if model_service.loaded else "unavailable"
    return VersionResponse(
        app_name=APP_NAME,
        app_version=APP_VERSION,
        model_version=model_version,
        model_name=model_name,
        environment=APP_ENVIRONMENT,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/info", response_model=ModelInfoResponse)
async def info() -> ModelInfoResponse:
    try:
        return ModelInfoResponse(**model_service.model_info())
    except RuntimeError as exc:
        logger.warning("Model info requested but model is unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="El modelo no está disponible.") from exc


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData) -> PredictionResponse:
    logger.info("Prediction request received: endpoint=/predict")
    try:
        with HEART_PREDICTION_DURATION_SECONDS.labels(endpoint="/predict").time():
            result = model_service.predict(patient)
    except RuntimeError as exc:
        HEART_PREDICTION_ERRORS_TOTAL.labels(endpoint="/predict").inc()
        logger.warning("Prediction rejected because model is unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="El modelo no está disponible.") from exc
    except ValueError as exc:
        HEART_PREDICTION_ERRORS_TOTAL.labels(endpoint="/predict").inc()
        logger.warning("Prediction rejected because input could not be processed: %s", exc)
        raise HTTPException(status_code=422, detail="La entrada no pudo procesarse correctamente.") from exc
    except Exception as exc:
        HEART_PREDICTION_ERRORS_TOTAL.labels(endpoint="/predict").inc()
        logger.exception("Prediction inference failed")
        raise HTTPException(status_code=500, detail="No se pudo generar la predicción.") from exc

    HEART_PREDICTIONS_TOTAL.labels(endpoint="/predict").inc()
    HEART_PREDICTIONS_BY_CLASS_TOTAL.labels(
        prediction=str(result.prediction),
        label=result.label,
        risk_level=result.risk_level,
    ).inc()
    logger.info(
        "Prediction response generated: prediction=%s risk_level=%s model_version=%s inference_time_ms=%s",
        result.prediction,
        result.risk_level,
        result.model_version,
        result.inference_time_ms,
    )
    return PredictionResponse(**result.__dict__)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    logger.info("Batch prediction request received: endpoint=/predict/batch count=%s", len(request.patients))
    try:
        with HEART_PREDICTION_DURATION_SECONDS.labels(endpoint="/predict/batch").time():
            results = model_service.predict_batch(request.patients)
    except RuntimeError as exc:
        HEART_PREDICTION_ERRORS_TOTAL.labels(endpoint="/predict/batch").inc()
        logger.warning("Batch prediction rejected because model is unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="El modelo no está disponible.") from exc
    except ValueError as exc:
        HEART_PREDICTION_ERRORS_TOTAL.labels(endpoint="/predict/batch").inc()
        logger.warning("Batch prediction rejected because input could not be processed: %s", exc)
        raise HTTPException(status_code=422, detail="La entrada no pudo procesarse correctamente.") from exc
    except Exception as exc:
        HEART_PREDICTION_ERRORS_TOTAL.labels(endpoint="/predict/batch").inc()
        logger.exception("Batch prediction inference failed")
        raise HTTPException(status_code=500, detail="No se pudieron generar las predicciones.") from exc

    for result in results:
        HEART_PREDICTIONS_TOTAL.labels(endpoint="/predict/batch").inc()
        HEART_PREDICTIONS_BY_CLASS_TOTAL.labels(
            prediction=str(result.prediction),
            label=result.label,
            risk_level=result.risk_level,
        ).inc()

    total_ms = round(sum(item.inference_time_ms for item in results), 3)
    logger.info(
        "Batch prediction response generated: count=%s model_version=%s total_inference_time_ms=%s",
        len(results),
        model_service.version,
        total_ms,
    )
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**item.__dict__) for item in results],
        count=len(results),
        total_inference_time_ms=total_ms,
        model_version=model_service.version,
    )


@app.get("/metrics")
async def metrics():
    return metrics_response()
