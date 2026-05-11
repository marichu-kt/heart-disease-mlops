"""
TEST DE INTEGRACIÓN PARA LA API FastAPI

Este archivo valida los endpoints principales de la aplicación:
- estado de salud de la API
- información y versión del modelo cargado
- predicción individual
- predicción por lotes
- validación de entradas incorrectas
- exposición de métricas Prometheus
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


PATIENT = {
    "age": 63,
    "sex": 1,
    "chest": 3,
    "resting_blood_pressure": 145,
    "serum_cholestoral": 233,
    "fasting_blood_sugar": 1,
    "resting_electrocardiographic_results": 0,
    "maximum_heart_rate_achieved": 150,
    "exercise_induced_angina": 0,
    "oldpeak": 2.3,
    "slope": 1,
    "number_of_major_vessels": 0,
    "thal": 6,
}

EXPECTED_FEATURES = set(PATIENT.keys())


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()

    assert payload["name"] == "Heart Disease MLOps API"
    assert payload["docs"] == "/docs"
    assert payload["health"] == "/health"
    assert payload["predict"] == "/predict"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True
    assert payload["model_version"] == "v4.2.0"
    assert "timestamp" in payload


def test_info_endpoint(client):
    response = client.get("/info")

    assert response.status_code == 200
    payload = response.json()

    assert payload["model_name"] == "heart_disease_mlp_calibrated"
    assert payload["version"] == "v4.2.0"
    assert "MLPClassifier" in payload["algorithm"]

    assert payload["selected_metric"] == "f2_score"
    assert payload["decision_threshold"] is not None
    assert 0 < payload["decision_threshold"] < 1

    assert payload["accuracy"] is not None
    assert payload["precision"] is not None
    assert payload["recall"] is not None
    assert payload["f1_score"] is not None
    assert payload["f2_score"] is not None
    assert payload["roc_auc"] is not None

    assert payload["recall"] >= 0.9
    assert payload["calibration_method"] is not None
    assert payload["calibration_applied"] in [True, False]

    assert set(payload["input_features"]) == EXPECTED_FEATURES


def test_version_endpoint(client):
    response = client.get("/version")

    assert response.status_code == 200
    payload = response.json()

    assert payload["app_name"] == "Heart Disease MLOps API"
    assert payload["app_version"] == "1.0.0"
    assert payload["model_version"] == "v4.2.0"
    assert payload["model_name"] == "heart_disease_mlp_calibrated"
    assert payload["environment"] == "local"
    assert "timestamp" in payload


def test_predict_endpoint(client):
    response = client.post("/predict", json=PATIENT)

    assert response.status_code == 200
    payload = response.json()

    assert payload["prediction"] in [0, 1]
    assert payload["label"] in ["Disease", "No Disease"]
    assert payload["risk_level"] in ["Low", "Medium", "High"]
    assert payload["model_version"] == "v4.2.0"

    assert payload["probability"] is not None
    assert 0 <= payload["probability"] <= 1

    assert isinstance(payload["inference_time_ms"], float)
    assert payload["inference_time_ms"] >= 0

    assert payload["decision_threshold"] is not None
    assert 0 < payload["decision_threshold"] < 1

    assert payload["probabilities"] is not None
    assert "disease" in payload["probabilities"]
    assert "no_disease" in payload["probabilities"]
    assert 0 <= payload["probabilities"]["disease"] <= 1
    assert 0 <= payload["probabilities"]["no_disease"] <= 1


def test_predict_endpoint_rejects_invalid_input(client):
    invalid_patient = PATIENT.copy()
    invalid_patient["age"] = 999

    response = client.post("/predict", json=invalid_patient)

    assert response.status_code == 422


def test_predict_endpoint_rejects_missing_field(client):
    invalid_patient = PATIENT.copy()
    invalid_patient.pop("thal")

    response = client.post("/predict", json=invalid_patient)

    assert response.status_code == 422


def test_batch_prediction_endpoint(client):
    response = client.post(
        "/predict/batch",
        json={
            "patients": [
                PATIENT,
                PATIENT,
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["count"] == 2
    assert payload["model_version"] == "v4.2.0"
    assert payload["total_inference_time_ms"] >= 0
    assert len(payload["predictions"]) == 2

    for prediction in payload["predictions"]:
        assert prediction["prediction"] in [0, 1]
        assert prediction["label"] in ["Disease", "No Disease"]
        assert prediction["risk_level"] in ["Low", "Medium", "High"]
        assert prediction["probability"] is not None
        assert 0 <= prediction["probability"] <= 1


def test_batch_prediction_rejects_empty_list(client):
    response = client.post(
        "/predict/batch",
        json={
            "patients": []
        },
    )

    assert response.status_code == 422


def test_metrics_endpoint(client):
    client.post("/predict", json=PATIENT)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    metrics = response.text

    expected_metrics = [
        "heart_predictions_total",
        "api_requests_total",
        "api_request_duration_seconds",
        "api_request_errors_total",
        "heart_prediction_duration_seconds",
        "heart_prediction_errors_total",
        "heart_predictions_by_class_total",
        "heart_model_info",
        "heart_api_up",
    ]

    for metric in expected_metrics:
        assert metric in metrics

    assert 'version="v4.2.0"' in metrics
