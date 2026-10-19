"""Unit tests for the PredictIQ API client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from dashboard.services.api_client import PredictIQAPIError, PredictIQClient


@pytest.fixture
def client():
    """Create a client instance targeting test URL."""
    return PredictIQClient(base_url="http://testserver:8000")


def test_client_init_strips_trailing_slash():
    c = PredictIQClient(base_url="http://testserver:8000/")
    assert c.base_url == "http://testserver:8000"


def test_check_health_success(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "healthy",
        "environment": "testing",
        "uptime_seconds": 12.34,
        "timestamp": "2026-10-18T16:00:00Z",
    }

    with patch.object(client.session, "get", return_value=mock_resp):
        res = client.check_health()
        assert res["status"] == "healthy"
        assert res["environment"] == "testing"


def test_check_health_connection_error(client):
    with patch.object(client.session, "get", side_effect=requests.exceptions.ConnectionError("Refused")):
        res = client.check_health()
        assert res["status"] == "unavailable"
        assert "Cannot connect" in res["error"]


def test_check_health_timeout(client):
    with patch.object(client.session, "get", side_effect=requests.exceptions.Timeout("Timed out")):
        res = client.check_health()
        assert res["status"] == "timeout"


def test_check_readiness_ready(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ready",
        "model_loaded": True,
        "model_name": "Random Forest (Calibrated)",
        "detail": "Model pipeline is loaded.",
    }

    with patch.object(client.session, "get", return_value=mock_resp):
        res = client.check_readiness()
        assert res["is_ready"] is True
        assert res["status"] == "ready"
        assert res["model_name"] == "Random Forest (Calibrated)"


def test_check_readiness_model_unready_503(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.json.return_value = {
        "status": "not_ready",
        "model_loaded": False,
        "detail": "Service not ready: Model artifact is not loaded.",
    }

    with patch.object(client.session, "get", return_value=mock_resp):
        res = client.check_readiness()
        assert res["is_ready"] is False
        assert res["status"] == "not_ready"
        assert "not loaded" in res["detail"]


def test_get_service_status_distinctions(client):
    # Case 1: Healthy
    with patch.object(client, "check_readiness", return_value={"is_ready": True, "model_name": "RF", "detail": "OK"}):
        code, label, _ = client.get_service_status()
        assert code == "healthy"
        assert "Ready" in label

    # Case 2: Unready (503)
    with patch.object(client, "check_readiness", return_value={"is_ready": False, "status": "not_ready", "detail": "Unready"}):
        code, label, _ = client.get_service_status()
        assert code == "unready"
        assert "503" in label

    # Case 3: Unavailable (Connection refused)
    with patch.object(client, "check_readiness", return_value={"is_ready": False, "status": "unavailable", "detail": "Refused"}):
        code, label, _ = client.get_service_status()
        assert code == "unavailable"
        assert "Offline" in label


def test_predict_success(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "predicted_churn": "Yes",
        "churn_probability": 0.5842,
        "decision_threshold": 0.3400,
        "risk_tier": "High Risk",
    }

    with patch.object(client.session, "post", return_value=mock_resp):
        res = client.predict({"gender": "Female", "tenure": 1})
        assert res["predicted_churn"] == "Yes"
        assert res["churn_probability"] == 0.5842
        assert res["decision_threshold"] == 0.3400
        assert res["risk_tier"] == "High Risk"


def test_predict_validation_error_422(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 422
    mock_resp.json.return_value = {"detail": "Missing required field 'gender'"}

    with patch.object(client.session, "post", return_value=mock_resp):
        with pytest.raises(PredictIQAPIError) as exc_info:
            client.predict({})
        assert exc_info.value.status_code == 422
        assert "validation error" in exc_info.value.message.lower()


def test_predict_service_unready_503(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.json.return_value = {"detail": "Model pipeline not loaded"}

    with patch.object(client.session, "post", return_value=mock_resp):
        with pytest.raises(PredictIQAPIError) as exc_info:
            client.predict({"gender": "Female"})
        assert exc_info.value.status_code == 503


def test_predict_batch_success(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "predictions": [
            {
                "predicted_churn": "Yes",
                "churn_probability": 0.58,
                "decision_threshold": 0.34,
                "risk_tier": "High Risk",
            },
            {
                "predicted_churn": "No",
                "churn_probability": 0.05,
                "decision_threshold": 0.34,
                "risk_tier": "Low Risk",
            },
        ],
        "total_records": 2,
        "batch_threshold": 0.34,
    }

    with patch.object(client.session, "post", return_value=mock_resp):
        res = client.predict_batch([{"record": 1}, {"record": 2}])
        assert len(res["predictions"]) == 2
        assert res["total_records"] == 2


def test_predict_batch_exceeds_max_limit_rejected_locally(client):
    records = [{"item": i} for i in range(501)]
    with pytest.raises(PredictIQAPIError) as exc_info:
        client.predict_batch(records)
    assert exc_info.value.status_code == 422
    assert "exceeds maximum allowed limit of 500" in exc_info.value.message


def test_predict_batch_empty_records_rejected_locally(client):
    with pytest.raises(PredictIQAPIError) as exc_info:
        client.predict_batch([])
    assert exc_info.value.status_code == 422
    assert "at least 1 record" in exc_info.value.message
