import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware


API_REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total HTTP requests received by the API.",
    ["method", "endpoint", "status_code"],
)

API_REQUEST_DURATION_SECONDS = Histogram(
    "api_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "endpoint"],
)

API_REQUEST_ERRORS_TOTAL = Counter(
    "api_request_errors_total",
    "Total API requests that returned server errors.",
    ["method", "endpoint"],
)

HEART_PREDICTIONS_TOTAL = Counter(
    "heart_predictions_total",
    "Total heart disease predictions.",
    ["endpoint"],
)

HEART_PREDICTION_ERRORS_TOTAL = Counter(
    "heart_prediction_errors_total",
    "Total heart disease prediction errors.",
    ["endpoint"],
)

HEART_PREDICTION_DURATION_SECONDS = Histogram(
    "heart_prediction_duration_seconds",
    "Heart disease model inference latency in seconds.",
    ["endpoint"],
)

HEART_PREDICTIONS_BY_CLASS_TOTAL = Counter(
    "heart_predictions_by_class_total",
    "Predictions grouped by predicted class and risk level.",
    ["prediction", "label", "risk_level"],
)

HEART_MODEL_INFO = Gauge(
    "heart_model_info",
    "Loaded model metadata. Gauge value is always 1.",
    ["model_name", "version", "algorithm"],
)

HEART_API_UP = Gauge("heart_api_up", "API health status. 1 means healthy.")


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        endpoint = request.scope.get("route").path if request.scope.get("route") else request.url.path
        method = request.method
        status_code = "500"

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            return response
        except Exception:
            API_REQUEST_ERRORS_TOTAL.labels(method=method, endpoint=endpoint).inc()
            raise
        finally:
            duration = time.perf_counter() - start
            API_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
            API_REQUEST_DURATION_SECONDS.labels(method=method, endpoint=endpoint).observe(duration)
            if status_code.startswith("5"):
                API_REQUEST_ERRORS_TOTAL.labels(method=method, endpoint=endpoint).inc()


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
