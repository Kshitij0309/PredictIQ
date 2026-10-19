"""API endpoint route definitions for PredictIQ."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from predictiq.api.deps import get_model_metadata, get_predictor
from predictiq.api.schemas import (
    BatchCustomerInputSchema,
    BatchPredictionResponse,
    CustomerInputSchema,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadyResponse,
)
from predictiq.inference.predictor import ChurnPredictor

router = APIRouter()


@router.get(
    "/",
    summary="Service Index",
    description="Returns service metadata, health summary, and navigational links to active endpoints.",
    tags=["System"],
)
async def root(request: Request) -> Dict[str, Any]:
    """Root service index returning API metadata and available endpoints."""
    start_time = getattr(request.app.state, "start_time", time.time())
    uptime = time.time() - start_time
    env = os.getenv("ENVIRONMENT", "development")
    docs_available = getattr(request.app, "docs_url", None) is not None

    endpoints = {
        "health": "/health",
        "ready": "/ready",
        "model_info": "/model-info",
        "predict": "/predict",
        "predict_batch": "/predict/batch",
    }
    if docs_available:
        endpoints["docs"] = "/docs"
        endpoints["openapi"] = "/openapi.json"

    return {
        "service": "PredictIQ Inference API",
        "version": "0.1.0",
        "status": "online",
        "environment": env,
        "uptime_seconds": round(uptime, 2),
        "endpoints": endpoints,
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Process Health Probe",
    description="Returns process liveness and uptime status.",
    tags=["System"],
)
async def health(request: Request) -> HealthResponse:
    """Check process health and uptime."""
    start_time = getattr(request.app.state, "start_time", time.time())
    uptime = time.time() - start_time
    env = os.getenv("ENVIRONMENT", "development")
    return HealthResponse(
        status="healthy",
        environment=env,
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ReadyResponse}},
    summary="Readiness Probe",
    description="Returns 200 if model pipeline is loaded and ready to serve traffic; 503 otherwise.",
    tags=["System"],
)
async def ready(request: Request) -> JSONResponse:
    """Verify that model artifacts are loaded in memory and ready for inference."""
    is_ready = getattr(request.app.state, "model_loaded", False)
    predictor: ChurnPredictor | None = getattr(request.app.state, "predictor", None)

    if is_ready and predictor is not None:
        meta = getattr(request.app.state, "metadata", {}) or predictor.metadata or {}
        model_name = meta.get("model_name", "Random Forest (Calibrated)")
        payload = ReadyResponse(
            status="ready",
            model_loaded=True,
            model_name=model_name,
            detail="PredictIQ model pipeline is loaded and ready for scoring.",
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=payload.model_dump())

    load_error = getattr(request.app.state, "load_error", "Model artifact is not loaded.")
    payload = ReadyResponse(
        status="not_ready",
        model_loaded=False,
        model_name=None,
        detail=f"Service not ready: {load_error}",
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(),
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Metadata & Threshold Information",
    description="Returns safe model specifications, calibrated scores, threshold, and top drivers without secrets or paths.",
    tags=["Model"],
)
async def model_info(
    predictor: ChurnPredictor = Depends(get_predictor),
    metadata: Dict[str, Any] = Depends(get_model_metadata),
) -> ModelInfoResponse:
    """Retrieve safe model configuration, frozen threshold, and explainability drivers."""
    meta = metadata or predictor.metadata or {}

    brier = meta.get("brier_score_calibrated")
    if brier is not None:
        brier = round(float(brier), 4)

    cv_mean = meta.get("cv_mean_pr_auc")
    if cv_mean is not None:
        cv_mean = round(float(cv_mean), 4)

    cv_std = meta.get("cv_std_pr_auc")
    if cv_std is not None:
        cv_std = round(float(cv_std), 4)

    return ModelInfoResponse(
        model_name=meta.get("model_name", "Random Forest (Calibrated)"),
        base_estimator=meta.get("base_estimator", "Random Forest"),
        decision_threshold=float(meta.get("optimal_decision_threshold", predictor.decision_threshold)),
        threshold_strategy=meta.get("threshold_selection_strategy", "Empirical F1-Score Maximization (Training OOF)"),
        calibration_method="Platt Scaling (Sigmoid)",
        brier_score_calibrated=brier,
        cv_mean_pr_auc=cv_mean,
        cv_std_pr_auc=cv_std,
        top_features=meta.get("top_10_shap_features", []),
        api_version="0.1.0",
    )


@router.get(
    "/model_info",
    response_model=ModelInfoResponse,
    include_in_schema=False,
)
async def model_info_alias(
    predictor: ChurnPredictor = Depends(get_predictor),
    metadata: Dict[str, Any] = Depends(get_model_metadata),
) -> ModelInfoResponse:
    """Handle underscore requests for model info without 404."""
    return await model_info(predictor=predictor, metadata=metadata)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single Customer Churn Prediction",
    description="Scores a single customer record using the calibrated champion pipeline and frozen decision threshold.",
    tags=["Inference"],
)
async def predict(
    payload: CustomerInputSchema,
    predictor: ChurnPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Score a single customer record and return calibrated churn probability and risk tier."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor service is not ready: model pipeline is not loaded.",
        )

    # Exclude tracking customerID if present
    data = payload.model_dump(exclude={"customerID"})
    df = pd.DataFrame([data])

    # Run CPU-bound prediction in worker thread
    detailed_df = await run_in_threadpool(predictor.predict_detailed, df)
    row = detailed_df.iloc[0]

    return PredictionResponse(
        predicted_churn=str(row["predicted_churn"]),
        churn_probability=round(float(row["churn_probability"]), 4),
        decision_threshold=round(float(row["decision_threshold"]), 4),
        risk_tier=str(row["risk_tier"]),
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Customer Churn Predictions",
    description="Scores a bounded batch of customer records (1 to 500 records) in a single request.",
    tags=["Inference"],
)
async def predict_batch(
    payload: BatchCustomerInputSchema,
    predictor: ChurnPredictor = Depends(get_predictor),
) -> BatchPredictionResponse:
    """Score a bounded batch of customer records and return ordered prediction responses."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor service is not ready: model pipeline is not loaded.",
        )

    records_data = [rec.model_dump(exclude={"customerID"}) for rec in payload.records]
    df = pd.DataFrame(records_data)

    # Run CPU-bound batch inference in worker thread
    detailed_df = await run_in_threadpool(predictor.predict_detailed, df)

    predictions = [
        PredictionResponse(
            predicted_churn=str(row["predicted_churn"]),
            churn_probability=round(float(row["churn_probability"]), 4),
            decision_threshold=round(float(row["decision_threshold"]), 4),
            risk_tier=str(row["risk_tier"]),
        )
        for _, row in detailed_df.iterrows()
    ]

    return BatchPredictionResponse(
        predictions=predictions,
        total_records=len(predictions),
        batch_threshold=round(float(predictor.decision_threshold), 4),
    )
