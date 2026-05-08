from contextlib import asynccontextmanager
from datetime import datetime, timezone

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
)

import os


model_service = ModelService()

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if origin.strip()
]

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        model_service.load_latest_model()
        info = model_service.model_info()
        HEART_MODEL_INFO.labels(
            model_name=info["model_name"],
            version=info["version"],
            algorithm=info["algorithm"],
        ).set(1)
        HEART_API_UP.set(1)
    except Exception:
        HEART_API_UP.set(0)
        raise
    yield


app = FastAPI(
    title="Heart Disease MLOps API",
    description="API FastAPI para prediccion de enfermedad cardiaca con modelo MLP versionado.",
    version="2.0.0",
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


@app.get("/info", response_model=ModelInfoResponse)
async def info() -> ModelInfoResponse:
    try:
        return ModelInfoResponse(**model_service.model_info())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData) -> PredictionResponse:
    try:
        with HEART_PREDICTION_DURATION_SECONDS.labels(endpoint="/predict").time():
            result = model_service.predict(patient)
    except Exception as exc:
        HEART_PREDICTION_ERRORS_TOTAL.labels(endpoint="/predict").inc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    HEART_PREDICTIONS_TOTAL.labels(endpoint="/predict").inc()
    HEART_PREDICTIONS_BY_CLASS_TOTAL.labels(
        prediction=str(result.prediction),
        label=result.label,
        risk_level=result.risk_level,
    ).inc()
    return PredictionResponse(**result.__dict__)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        with HEART_PREDICTION_DURATION_SECONDS.labels(endpoint="/predict/batch").time():
            results = model_service.predict_batch(request.patients)
    except Exception as exc:
        HEART_PREDICTION_ERRORS_TOTAL.labels(endpoint="/predict/batch").inc()
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}") from exc

    for result in results:
        HEART_PREDICTIONS_TOTAL.labels(endpoint="/predict/batch").inc()
        HEART_PREDICTIONS_BY_CLASS_TOTAL.labels(
            prediction=str(result.prediction),
            label=result.label,
            risk_level=result.risk_level,
        ).inc()

    total_ms = round(sum(item.inference_time_ms for item in results), 3)
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**item.__dict__) for item in results],
        count=len(results),
        total_inference_time_ms=total_ms,
        model_version=model_service.version,
    )


@app.get("/metrics")
async def metrics():
    return metrics_response()
