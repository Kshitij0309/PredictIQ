"""Pydantic v2 data schemas for the PredictIQ REST API.

Defines strict input validation, batch constraints, and safe response models
without exposing customer identifiers, raw training data, internal paths, or secrets.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

# Maximum allowed batch size for POST /predict/batch
MAX_BATCH_SIZE: int = 500


class CustomerInputSchema(BaseModel):
    """Customer features schema for churn inference."""

    gender: str = Field(..., description="Customer gender: Female or Male", examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen indicator (0 or 1)", examples=[0])
    Partner: str = Field(..., description="Whether customer has a partner (Yes or No)", examples=["Yes"])
    Dependents: str = Field(..., description="Whether customer has dependents (Yes or No)", examples=["No"])
    tenure: int = Field(..., ge=0, le=120, description="Months customer has stayed with company", examples=[12])
    PhoneService: str = Field(..., description="Whether customer has phone service (Yes or No)", examples=["Yes"])
    MultipleLines: str = Field(..., description="Multiple lines status", examples=["No"])
    InternetService: str = Field(..., description="Internet service provider (DSL, Fiber optic, No)", examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., description="Online security add-on status", examples=["No"])
    OnlineBackup: str = Field(..., description="Online backup add-on status", examples=["Yes"])
    DeviceProtection: str = Field(..., description="Device protection add-on status", examples=["No"])
    TechSupport: str = Field(..., description="Tech support add-on status", examples=["No"])
    StreamingTV: str = Field(..., description="Streaming TV add-on status", examples=["Yes"])
    StreamingMovies: str = Field(..., description="Streaming movies add-on status", examples=["No"])
    Contract: str = Field(..., description="Contract term (Month-to-month, One year, Two year)", examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., description="Paperless billing (Yes or No)", examples=["Yes"])
    PaymentMethod: str = Field(..., description="Payment method", examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly charges in USD", examples=[70.35])
    TotalCharges: Union[float, str] = Field(
        default=0.0,
        description="Total charges in USD (numeric or string representation)",
        examples=[844.20],
    )
    customerID: Optional[str] = Field(
        default=None,
        description="Optional customer tracking identifier (omitted from output and logging)",
        examples=["7590-VHVEG"],
    )

    @field_validator(
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        mode="before",
    )
    @classmethod
    def strip_strings(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def validate_total_charges(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_str = v.strip()
            if v_str == "":
                return 0.0
            try:
                val = float(v_str)
            except ValueError:
                raise ValueError("TotalCharges must be a valid numeric value or empty string")
            if val < 0.0:
                raise ValueError("TotalCharges must be non-negative")
            return val
        if isinstance(v, (int, float)):
            if v < 0.0:
                raise ValueError("TotalCharges must be non-negative")
            return float(v)
        return v


class BatchCustomerInputSchema(BaseModel):
    """Batch input schema with strict size limits."""

    records: List[CustomerInputSchema] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"List of customer records to evaluate (1 to {MAX_BATCH_SIZE} records)",
    )


class PredictionResponse(BaseModel):
    """Standardized single customer prediction response."""

    predicted_churn: str = Field(..., description="Predicted churn outcome ('Yes' or 'No')", examples=["Yes"])
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated churn probability estimate", examples=[0.7484])
    decision_threshold: float = Field(..., ge=0.0, le=1.0, description="Frozen classification decision threshold", examples=[0.3400])
    risk_tier: str = Field(..., description="Business risk tier ('Low Risk', 'Medium Risk', 'High Risk')", examples=["High Risk"])


class BatchPredictionResponse(BaseModel):
    """Batch prediction response containing ordered predictions."""

    predictions: List[PredictionResponse] = Field(..., description="List of predictions matching input record order")
    total_records: int = Field(..., description="Total records evaluated", examples=[10])
    batch_threshold: float = Field(..., description="Decision threshold applied across the batch", examples=[0.3400])


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str = Field(default="healthy", description="Application service health status")
    environment: str = Field(..., description="Current operational environment")
    uptime_seconds: float = Field(..., description="Server process uptime in seconds")
    timestamp: str = Field(..., description="Current UTC timestamp (ISO 8601)")


class ReadyResponse(BaseModel):
    """Readiness probe response indicating model availability."""

    status: str = Field(..., description="'ready' when model pipeline is available, 'not_ready' otherwise")
    model_loaded: bool = Field(..., description="Whether the prediction pipeline is loaded into memory")
    model_name: Optional[str] = Field(default=None, description="Safe model identifier if loaded")
    detail: Optional[str] = Field(default=None, description="Actionable readiness status message")


class ModelInfoResponse(BaseModel):
    """Safe model metadata response without exposing secrets or filesystem paths."""

    model_name: str = Field(..., description="Model architecture and calibration specification")
    base_estimator: str = Field(..., description="Underlying ensemble or classifier family")
    decision_threshold: float = Field(..., description="Frozen optimal classification decision threshold")
    threshold_strategy: str = Field(..., description="Method used to derive decision threshold")
    calibration_method: str = Field(default="Platt Scaling (Sigmoid)", description="Probability calibration method")
    brier_score_calibrated: Optional[float] = Field(default=None, description="Post-calibration Brier reliability score")
    cv_mean_pr_auc: Optional[float] = Field(default=None, description="Training CV Average Precision (PR-AUC)")
    cv_std_pr_auc: Optional[float] = Field(default=None, description="Standard deviation across CV folds")
    top_features: List[str] = Field(default_factory=list, description="Top predictive drivers identified by TreeSHAP")
    api_version: str = Field(default="0.1.0", description="API package version")


class ErrorResponse(BaseModel):
    """Standardized error response schema."""

    error: str = Field(..., description="Error category name")
    detail: str = Field(..., description="Safe, descriptive error message")
    timestamp: str = Field(..., description="UTC timestamp of occurrence")
