"""Comprehensive test suite for PredictIQ FastAPI REST service.

Verifies:
- Health and readiness probes.
- Model loading lifecycle and missing artifact error handling.
- Valid single-record and batch predictions with frozen decision threshold (p*=0.3400).
- Input validation (missing fields, invalid types, boundary values, malformed JSON).
- Handling of unknown categorical levels.
- Bounded batch-size enforcement (1 to 500 records).
- Absence of sensitive information (customer identifiers, stack traces, internal paths) in responses.
- Production documentation suppression.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from predictiq.api.app import create_app
from predictiq.api.deps import get_predictor
from predictiq.api.schemas import MAX_BATCH_SIZE
from predictiq.config import Config


@pytest.fixture
def valid_customer_payload() -> Dict[str, Any]:
    """Provide a complete, valid customer payload with tracking customerID."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 84.80,
        "TotalCharges": 1017.60,
        "customerID": "CONFIDENTIAL-ID-999",
    }


@pytest.fixture
def client() -> TestClient:
    """Create a TestClient with the default application and loaded champion model."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


# ==============================================================================
# 1. Health & Readiness Tests
# ==============================================================================

def test_health_endpoint(client: TestClient) -> None:
    """GET /health must report healthy status, environment, and uptime."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "environment" in data
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0.0
    assert "timestamp" in data


def test_ready_endpoint_when_loaded(client: TestClient) -> None:
    """GET /ready must return 200 and model_loaded=True when model is in memory."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert "Random Forest" in data["model_name"]


def test_ready_endpoint_when_not_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /ready must return 503 and model_loaded=False when model is not available."""
    monkeypatch.setenv("MODEL_PATH", "missing/artifact/path.joblib")
    monkeypatch.setenv("PREDICTIQ_STRICT_STARTUP", "false")

    app = create_app()
    with TestClient(app) as unready_client:
        response = unready_client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["model_loaded"] is False
        assert "not ready" in data["detail"].lower()


def test_missing_model_artifact_startup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Startup must raise RuntimeError when PREDICTIQ_STRICT_STARTUP=true and artifact missing."""
    monkeypatch.setenv("MODEL_PATH", "nonexistent/path/to/missing_model.joblib")
    monkeypatch.setenv("PREDICTIQ_STRICT_STARTUP", "true")

    app = create_app()
    with pytest.raises(RuntimeError, match="PredictIQ API startup aborted"):
        with TestClient(app):
            pass


# ==============================================================================
# 2. Metadata & Model Info Tests
# ==============================================================================

def test_model_info_endpoint(client: TestClient) -> None:
    """GET /model-info must return safe configuration and frozen threshold."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()

    assert "Random Forest" in data["model_name"]
    assert data["decision_threshold"] == 0.34
    assert data["threshold_strategy"] == "Empirical F1-Score Maximization (Training OOF)"
    assert data["calibration_method"] == "Platt Scaling (Sigmoid)"
    assert isinstance(data["top_features"], list)
    assert len(data["top_features"]) > 0

    # Safety check: ensure response does not leak internal filesystem paths
    for key, value in data.items():
        if isinstance(value, str):
            assert "D:\\" not in value
            assert "C:\\" not in value
            assert "/home/" not in value


# ==============================================================================
# 3. Single Prediction & Threshold Tests
# ==============================================================================

def test_predict_single_valid(client: TestClient, valid_customer_payload: Dict[str, Any]) -> None:
    """POST /predict returns valid schema with calibrated probability and risk tier."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_churn"] in ("Yes", "No")
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["decision_threshold"] == 0.34
    assert data["risk_tier"] in ("Low Risk", "Medium Risk", "High Risk")

    # Tracking customerID must NEVER be leaked in prediction response
    assert "customerID" not in data
    assert "CONFIDENTIAL-ID-999" not in str(data)


def test_predict_frozen_decision_threshold_consistency(
    client: TestClient,
    valid_customer_payload: Dict[str, Any],
) -> None:
    """Predicted label must strictly match condition (churn_probability >= decision_threshold)."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()

    prob = data["churn_probability"]
    threshold = data["decision_threshold"]
    expected_label = "Yes" if prob >= threshold else "No"
    assert data["predicted_churn"] == expected_label


def test_predict_handles_unknown_categorical_values(
    client: TestClient,
    valid_customer_payload: Dict[str, Any],
) -> None:
    """Novel categorical values must be safely handled via OneHotEncoder(handle_unknown='ignore')."""
    novel_payload = valid_customer_payload.copy()
    novel_payload["PaymentMethod"] = "Cryptocurrency (Bitcoin)"
    novel_payload["Contract"] = "Three year corporate agreement"
    novel_payload["InternetService"] = "Satellite Quantum Link"

    response = client.post("/predict", json=novel_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_churn"] in ("Yes", "No")
    assert 0.0 <= data["churn_probability"] <= 1.0


# ==============================================================================
# 4. Batch Prediction Tests
# ==============================================================================

def test_predict_batch_valid(client: TestClient, valid_customer_payload: Dict[str, Any]) -> None:
    """POST /predict/batch scores ordered list of records accurately."""
    batch_payload = {"records": [valid_customer_payload, valid_customer_payload, valid_customer_payload]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_records"] == 3
    assert data["batch_threshold"] == 0.34
    assert len(data["predictions"]) == 3

    for item in data["predictions"]:
        assert item["predicted_churn"] in ("Yes", "No")
        assert 0.0 <= item["churn_probability"] <= 1.0
        assert item["decision_threshold"] == 0.34


def test_predict_batch_empty_records_rejected(client: TestClient) -> None:
    """POST /predict/batch with empty records list returns 422 Unprocessable Entity."""
    response = client.post("/predict/batch", json={"records": []})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Request Validation Error"


def test_predict_batch_exceeding_max_limit_rejected(
    client: TestClient,
    valid_customer_payload: Dict[str, Any],
) -> None:
    """POST /predict/batch exceeding MAX_BATCH_SIZE (500) returns 422."""
    oversized_records = [valid_customer_payload] * (MAX_BATCH_SIZE + 1)
    response = client.post("/predict/batch", json={"records": oversized_records})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Request Validation Error"


# ==============================================================================
# 5. Input Validation & Error Handling Tests
# ==============================================================================

def test_predict_missing_required_fields(
    client: TestClient,
    valid_customer_payload: Dict[str, Any],
) -> None:
    """Missing core fields must return 422 with descriptive error explanation."""
    bad_payload = valid_customer_payload.copy()
    del bad_payload["MonthlyCharges"]
    del bad_payload["tenure"]

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Request Validation Error"
    assert "MonthlyCharges" in data["detail"]
    assert "tenure" in data["detail"]


def test_predict_invalid_field_types(
    client: TestClient,
    valid_customer_payload: Dict[str, Any],
) -> None:
    """Invalid types or negative boundary values must return 422."""
    bad_payload = valid_customer_payload.copy()
    bad_payload["MonthlyCharges"] = -50.0  # Must be >= 0.0

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Request Validation Error"


def test_predict_malformed_json_syntax(client: TestClient) -> None:
    """Non-JSON raw body must return 422."""
    response = client.post(
        "/predict",
        content="This is not valid JSON content",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_predict_when_unready_returns_503(valid_customer_payload: Dict[str, Any]) -> None:
    """POST /predict returns 503 when predictor is not loaded."""
    app = create_app()
    app.dependency_overrides[get_predictor] = lambda: None  # Simulate missing predictor

    with TestClient(app) as unready_client:
        response = unready_client.post("/predict", json=valid_customer_payload)
        assert response.status_code == 503
        data = response.json()
        assert "not ready" in data["detail"].lower()


# ==============================================================================
# 6. Security & Documentation Visibility Tests
# ==============================================================================

def test_production_mode_hides_api_documentation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interactive docs (/docs, /redoc, /openapi.json) must be disabled in production by default."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ENABLE_DOCS", raising=False)

    prod_config = Config.from_env(load_env=False)
    prod_app = create_app(config=prod_config)

    with TestClient(prod_app) as prod_client:
        assert prod_client.get("/docs").status_code == 404
        assert prod_client.get("/redoc").status_code == 404
        assert prod_client.get("/openapi.json").status_code == 404


def test_production_mode_with_explicit_enable_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ENABLE_DOCS=true in production, interactive docs must be enabled."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ENABLE_DOCS", "true")

    prod_config = Config.from_env(load_env=False)
    prod_app = create_app(config=prod_config)

    with TestClient(prod_app) as prod_client:
        assert prod_client.get("/docs").status_code == 200
        assert prod_client.get("/openapi.json").status_code == 200


def test_root_endpoint(client: TestClient) -> None:
    """GET / must return 200 with service index metadata and available endpoints."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "PredictIQ Inference API"
    assert data["status"] == "online"
    assert "endpoints" in data
    assert data["endpoints"]["health"] == "/health"
    assert data["endpoints"]["ready"] == "/ready"
    assert data["endpoints"]["model_info"] == "/model-info"
    assert data["endpoints"]["predict"] == "/predict"


def test_model_info_alias_matches_canonical(client: TestClient) -> None:
    """GET /model_info (underscore) must return identical response to /model-info (hyphen)."""
    canonical_resp = client.get("/model-info")
    alias_resp = client.get("/model_info")

    assert canonical_resp.status_code == 200
    assert alias_resp.status_code == 200
    assert canonical_resp.json() == alias_resp.json()
