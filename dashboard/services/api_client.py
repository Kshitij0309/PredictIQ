"""REST API client for communicating with the PredictIQ FastAPI backend."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

DEFAULT_API_URL: str = "http://localhost:8000"
CONNECT_TIMEOUT: float = 3.0
DEFAULT_READ_TIMEOUT: float = 10.0
BATCH_READ_TIMEOUT: float = 45.0


class PredictIQAPIError(Exception):
    """Base exception for PredictIQ API client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, detail: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail


class PredictIQClient:
    """Client for the PredictIQ FastAPI prediction service."""

    def __init__(self, base_url: Optional[str] = None):
        env_url = (os.getenv("PREDICTIQ_API_URL") or "").strip()
        raw_url = base_url or (env_url if env_url else DEFAULT_API_URL)
        self.base_url: str = raw_url.rstrip("/")
        self.session = requests.Session()

    def check_health(self) -> Dict[str, Any]:
        """Check process health and liveness."""
        url = f"{self.base_url}/health"
        try:
            resp = self.session.get(url, timeout=(CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT))
            if resp.status_code == 200:
                return resp.json()
            return {
                "status": "unhealthy",
                "status_code": resp.status_code,
                "error": f"Health probe returned HTTP {resp.status_code}",
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "unavailable",
                "error": f"Cannot connect to PredictIQ API at {self.base_url}. Ensure the FastAPI server is running.",
            }
        except requests.exceptions.Timeout:
            return {
                "status": "timeout",
                "error": f"Health check request to {self.base_url} timed out.",
            }
        except Exception as err:
            return {
                "status": "error",
                "error": f"Unexpected health check failure: {err}",
            }

    def check_readiness(self) -> Dict[str, Any]:
        """Check whether the model pipeline is loaded and ready for scoring."""
        url = f"{self.base_url}/ready"
        try:
            resp = self.session.get(url, timeout=(CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT))
            if resp.status_code == 200:
                data = resp.json()
                data["is_ready"] = True
                return data
            if resp.status_code == 503:
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                data["is_ready"] = False
                data["status"] = "not_ready"
                return data
            return {
                "is_ready": False,
                "status": "error",
                "status_code": resp.status_code,
                "detail": f"Readiness probe returned HTTP {resp.status_code}",
            }
        except requests.exceptions.ConnectionError:
            return {
                "is_ready": False,
                "status": "unavailable",
                "detail": f"Cannot connect to PredictIQ API at {self.base_url}. Connection refused.",
            }
        except requests.exceptions.Timeout:
            return {
                "is_ready": False,
                "status": "timeout",
                "detail": "Readiness probe timed out.",
            }
        except Exception as err:
            return {
                "is_ready": False,
                "status": "error",
                "detail": f"Unexpected error during readiness check: {err}",
            }

    def get_service_status(self) -> Tuple[str, str, str]:
        """Evaluate overall API health and readiness status.

        Returns:
            Tuple of (status_code, badge_label, description_message)
            - status_code: 'healthy' | 'unready' | 'unavailable'
        """
        readiness = self.check_readiness()
        if readiness.get("is_ready"):
            model_name = readiness.get("model_name", "Random Forest (Calibrated)")
            return (
                "healthy",
                f"API Online • {model_name} Ready",
                readiness.get("detail", "PredictIQ model is loaded and ready for inference."),
            )

        if readiness.get("status") == "not_ready":
            return (
                "unready",
                "API Online • Model Not Loaded (503)",
                readiness.get("detail", "FastAPI is running, but model pipeline is not loaded."),
            )

        # Connection error or timeout
        return (
            "unavailable",
            "API Offline • Connection Refused",
            readiness.get("detail", f"Cannot connect to PredictIQ API at {self.base_url}."),
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Fetch model metadata and frozen decision threshold."""
        url = f"{self.base_url}/model-info"
        try:
            resp = self.session.get(url, timeout=(CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT))
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 503:
                raise PredictIQAPIError(
                    "Model metadata unavailable: model pipeline is not loaded on backend.",
                    status_code=503,
                )
            raise PredictIQAPIError(
                f"Failed to retrieve model info: HTTP {resp.status_code}",
                status_code=resp.status_code,
            )
        except requests.exceptions.ConnectionError as err:
            raise PredictIQAPIError(
                f"Cannot connect to PredictIQ API at {self.base_url}.",
                status_code=503,
            ) from err
        except requests.exceptions.Timeout as err:
            raise PredictIQAPIError("Model info request timed out.", status_code=504) from err

    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single customer record.

        Args:
            record: Customer input dictionary conforming to CustomerInputSchema.

        Returns:
            Dict containing predicted_churn, churn_probability, decision_threshold, risk_tier.

        Raises:
            PredictIQAPIError: On validation failure, unready service, or network error.
        """
        url = f"{self.base_url}/predict"
        try:
            resp = self.session.post(url, json=record, timeout=(CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT))
            if resp.status_code == 200:
                return resp.json()

            # Handle structured errors
            try:
                err_data = resp.json()
                detail = err_data.get("detail", resp.text)
            except Exception:
                detail = resp.text

            if resp.status_code == 422:
                raise PredictIQAPIError(f"Input validation error: {detail}", status_code=422, detail=str(detail))
            if resp.status_code == 503:
                raise PredictIQAPIError(f"Prediction service unavailable: {detail}", status_code=503, detail=str(detail))

            raise PredictIQAPIError(
                f"Prediction failed with HTTP {resp.status_code}: {detail}",
                status_code=resp.status_code,
                detail=str(detail),
            )
        except requests.exceptions.ConnectionError as err:
            raise PredictIQAPIError(
                f"Cannot connect to PredictIQ API at {self.base_url}. Ensure the service is running.",
                status_code=503,
            ) from err
        except requests.exceptions.Timeout as err:
            raise PredictIQAPIError("Prediction request timed out.", status_code=504) from err

    def predict_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Score a batch of customer records ($1 \\le N \\le 500$).

        Args:
            records: List of customer input dictionaries.

        Returns:
            Dict containing predictions, total_records, batch_threshold.

        Raises:
            PredictIQAPIError: On validation failure, batch size violation, or backend error.
        """
        if not records:
            raise PredictIQAPIError("Batch payload must contain at least 1 record.", status_code=422)
        if len(records) > 500:
            raise PredictIQAPIError(
                f"Batch size {len(records)} exceeds maximum allowed limit of 500 records.",
                status_code=422,
            )

        url = f"{self.base_url}/predict/batch"
        payload = {"records": records}
        try:
            resp = self.session.post(url, json=payload, timeout=(CONNECT_TIMEOUT, BATCH_READ_TIMEOUT))
            if resp.status_code == 200:
                return resp.json()

            try:
                err_data = resp.json()
                detail = err_data.get("detail", resp.text)
            except Exception:
                detail = resp.text

            if resp.status_code == 422:
                raise PredictIQAPIError(f"Batch validation error: {detail}", status_code=422, detail=str(detail))
            if resp.status_code == 503:
                raise PredictIQAPIError(f"Prediction service unavailable: {detail}", status_code=503, detail=str(detail))

            raise PredictIQAPIError(
                f"Batch prediction failed with HTTP {resp.status_code}: {detail}",
                status_code=resp.status_code,
                detail=str(detail),
            )
        except requests.exceptions.ConnectionError as err:
            raise PredictIQAPIError(
                f"Cannot connect to PredictIQ API at {self.base_url}. Ensure the service is running.",
                status_code=503,
            ) from err
        except requests.exceptions.Timeout as err:
            raise PredictIQAPIError("Batch prediction request timed out.", status_code=504) from err
