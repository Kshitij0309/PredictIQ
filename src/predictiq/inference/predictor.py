"""Inference engine and prediction interface for PredictIQ.

Provides safe scoring of customer churn records, input validation, risk stratification,
and graceful handling of unseen categories and missing optional features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from predictiq.models.persistence import load_model_pipeline

logger = logging.getLogger(__name__)

# Core raw features expected for prediction (excluding target and ID)
EXPECTED_RAW_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
]


class InferenceValidationError(ValueError):
    """Raised when inference input data is malformed, missing required fields, or empty."""
    pass


class ChurnPredictor:
    """Production-grade inference interface for PredictIQ Churn models."""

    def __init__(
        self,
        pipeline: Pipeline,
        metadata: Optional[Dict[str, Any]] = None,
        decision_threshold: float = 0.50,
        risk_threshold_medium: float = 0.30,
        risk_threshold_high: float = 0.60,
    ) -> None:
        self.pipeline = pipeline
        self.metadata = metadata or {}
        self.decision_threshold = decision_threshold
        self.risk_threshold_medium = risk_threshold_medium
        self.risk_threshold_high = risk_threshold_high

    @classmethod
    def from_saved_model(
        cls,
        model_path: Optional[Path | str] = None,
        metadata_path: Optional[Path | str] = None,
    ) -> ChurnPredictor:
        """Instantiate predictor by loading serialized pipeline from disk."""
        pipeline, metadata = load_model_pipeline(model_path=model_path, metadata_path=metadata_path)
        thresh = float(metadata.get("optimal_decision_threshold", 0.50))
        return cls(pipeline=pipeline, metadata=metadata, decision_threshold=thresh)

    def _normalize_and_validate_input(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """Validate input structure, enforce schema expectations, and handle edge cases."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            if len(data) == 0:
                raise InferenceValidationError("Input record list is empty.")
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                raise InferenceValidationError("Input DataFrame contains zero records.")
            df = data.copy()
        else:
            raise InferenceValidationError(
                f"Unsupported input type: {type(data)}. Must be DataFrame, dict, or list of dicts."
            )

        # Check for presence of core raw features
        missing = [col for col in EXPECTED_RAW_FEATURES if col not in df.columns]
        if missing:
            raise InferenceValidationError(
                f"Input data is missing {len(missing)} required features: {missing}"
            )

        # Remove identifiers or target if present in input
        cols_to_drop = [c for c in ["customerID", "Churn"] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Strip whitespace from strings
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()

        # Handle TotalCharges whitespace if raw strings passed
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"].replace("", "0.0"), errors="coerce"
            ).fillna(0.0)

        if "tenure" in df.columns:
            df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)

        if "MonthlyCharges" in df.columns:
            df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)

        return df

    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> np.ndarray:
        """Generate binary churn predictions using the configured decision threshold."""
        df = self._normalize_and_validate_input(data)
        if hasattr(self.pipeline, "predict_proba"):
            probas = self.predict_proba(df)
            return (probas >= self.decision_threshold).astype(int)
        return self.pipeline.predict(df).astype(int)

    def predict_proba(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> np.ndarray:
        """Generate probability estimates for customer churn."""
        df = self._normalize_and_validate_input(data)
        if not hasattr(self.pipeline, "predict_proba"):
            raise RuntimeError("Underlying model pipeline does not support probability estimation.")

        probas = self.pipeline.predict_proba(df)
        if probas.shape[1] >= 2:
            return probas[:, 1]
        return probas[:, 0]

    def predict_detailed(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """Produce rich prediction dataframe with churn labels, probabilities, and risk tiers."""
        raw_df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        clean_df = self._normalize_and_validate_input(raw_df)

        has_proba = hasattr(self.pipeline, "predict_proba")
        if has_proba:
            probas = self.predict_proba(clean_df)
            preds = (probas >= self.decision_threshold).astype(int)
        else:
            probas = np.full(len(clean_df), np.nan)
            preds = self.pipeline.predict(clean_df).astype(int)

        results = raw_df.copy()
        results["predicted_churn"] = np.where(preds == 1, "Yes", "No")
        results["churn_probability"] = np.round(probas, 4)
        results["decision_threshold"] = round(self.decision_threshold, 4)

        # Risk tier assignment
        def assign_risk(p: float) -> str:
            if np.isnan(p):
                return "Unknown"
            if p >= self.risk_threshold_high:
                return "High Risk"
            elif p >= self.risk_threshold_medium:
                return "Medium Risk"
            else:
                return "Low Risk"

        results["risk_tier"] = [assign_risk(p) for p in probas]
        return results
