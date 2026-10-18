"""Dependency injection providers for the PredictIQ FastAPI application."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, status

from predictiq.inference.predictor import ChurnPredictor


def get_predictor(request: Request) -> ChurnPredictor:
    """Dependency retrieving the shared ChurnPredictor instance from application state.

    Raises:
        HTTPException: 503 Service Unavailable if model is not yet loaded or failed.
    """
    predictor: Optional[ChurnPredictor] = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor service is not ready: model pipeline is not loaded.",
        )
    return predictor


def get_model_metadata(request: Request) -> Dict[str, Any]:
    """Dependency retrieving safe metadata dictionary from application state."""
    meta: Optional[Dict[str, Any]] = getattr(request.app.state, "metadata", None)
    return meta or {}
