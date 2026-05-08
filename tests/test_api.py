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


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True


def test_info_endpoint():
    with TestClient(app) as client:
        response = client.get("/info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "heart_disease_mlp"
    assert payload["version"] == "v2.0.0"
    assert "age" in payload["input_features"]


def test_version_endpoint():
    with TestClient(app) as client:
        response = client.get("/version")

    assert response.status_code == 200
    payload = response.json()
    assert payload["app_name"] == "Heart Disease MLOps API"
    assert payload["app_version"] == "1.0.0"
    assert payload["model_version"] == "v2.0.0"
    assert payload["model_name"] == "heart_disease_mlp"
    assert payload["environment"] == "local"
    assert "timestamp" in payload


def test_predict_endpoint():
    with TestClient(app) as client:
        response = client.post("/predict", json=PATIENT)

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in [0, 1]
    assert payload["label"] in ["Disease", "No Disease"]
    assert payload["risk_level"] in ["Low", "Medium", "High"]
    assert payload["model_version"] == "v2.0.0"
    assert isinstance(payload["inference_time_ms"], float)
    assert payload["inference_time_ms"] >= 0
    assert "probabilities" in payload


def test_metrics_endpoint():
    with TestClient(app) as client:
        client.post("/predict", json=PATIENT)
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "heart_predictions_total" in response.text
    assert "api_requests_total" in response.text
    assert "heart_prediction_duration_seconds" in response.text
    assert "heart_predictions_by_class_total" in response.text
    assert "heart_model_info" in response.text
    assert "heart_api_up" in response.text
