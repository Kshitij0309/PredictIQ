"""Domain-specific feature engineering transformers for PredictIQ.

Implements leakage-safe, scikit-learn compatible transformers for customer churn.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

SERVICE_COLUMNS: List[str] = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineers business-grounded features for customer churn prediction.

    Engineered features:
    - service_count: Total count of active subscribed services (0 to 9).
    - streaming_services_count: Count of active entertainment services (StreamingTV + StreamingMovies).
    - has_tech_support: Binary flag for high-retention technical assistance (reduces churn ~2.7x).
    - has_online_security: Binary flag for online security subscription.
    - is_month_to_month: Binary flag for highest-risk contract commitment.
    - is_fiber_optic: Binary flag for Fiber Optic internet (high charge / high friction segment).
    - is_electronic_check: Binary flag for electronic check payment method.
    - tenure_group: Categorical cohort binning tenure into [0-12, 13-24, 25-48, 49-72] months.

    Design Rationale for Omitted Features:
    - TotalCharges / MonthlyCharges ratio is deliberately OMITTED.
      In subscription telecom services, TotalCharges is approximately MonthlyCharges * tenure.
      Dividing them yields a noisy proxy for tenure (correlation r > 0.98), introduces division-by-zero
      risk for unbilled accounts, and produces severe multicollinearity without independent predictive value.
    """

    def __init__(
        self,
        create_service_counts: bool = True,
        create_contract_flags: bool = True,
        create_tenure_cohorts: bool = True,
    ) -> None:
        self.create_service_counts = create_service_counts
        self.create_contract_flags = create_contract_flags
        self.create_tenure_cohorts = create_tenure_cohorts
        self.feature_names_out_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series | np.ndarray] = None) -> DomainFeatureEngineer:
        """Fit transformer (stateless transformation; records output column structure)."""
        if isinstance(X, np.ndarray):
            # If array passed, nothing to infer column names from
            self.feature_names_out_ = None
            return self

        # Generate sample transformed output to learn column structure
        sample_transformed = self._transform_dataframe(X.head(2))
        self.feature_names_out_ = list(sample_transformed.columns)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Apply feature engineering transformations.

        Args:
            X: Input dataframe or array.

        Returns:
            pd.DataFrame: Transformed dataframe containing original and engineered features.
        """
        if isinstance(X, np.ndarray):
            raise TypeError("DomainFeatureEngineer requires a pandas DataFrame with column names.")

        return self._transform_dataframe(X)

    def _transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal transform logic operating on DataFrame."""
        out = df.copy()

        # 1. Active service count features
        if self.create_service_counts:
            service_indicators = []

            if "PhoneService" in out.columns:
                service_indicators.append((out["PhoneService"].astype(str) == "Yes").astype(int))
            if "MultipleLines" in out.columns:
                service_indicators.append((out["MultipleLines"].astype(str) == "Yes").astype(int))
            if "InternetService" in out.columns:
                service_indicators.append(
                    out["InternetService"].astype(str).isin(["DSL", "Fiber optic"]).astype(int)
                )
            if "OnlineSecurity" in out.columns:
                sec = (out["OnlineSecurity"].astype(str) == "Yes").astype(int)
                service_indicators.append(sec)
                out["has_online_security"] = sec
            if "OnlineBackup" in out.columns:
                service_indicators.append((out["OnlineBackup"].astype(str) == "Yes").astype(int))
            if "DeviceProtection" in out.columns:
                service_indicators.append((out["DeviceProtection"].astype(str) == "Yes").astype(int))
            if "TechSupport" in out.columns:
                tech = (out["TechSupport"].astype(str) == "Yes").astype(int)
                service_indicators.append(tech)
                out["has_tech_support"] = tech

            streaming_count = pd.Series(0, index=out.index, dtype=int)
            if "StreamingTV" in out.columns:
                tv = (out["StreamingTV"].astype(str) == "Yes").astype(int)
                service_indicators.append(tv)
                streaming_count += tv
            if "StreamingMovies" in out.columns:
                mov = (out["StreamingMovies"].astype(str) == "Yes").astype(int)
                service_indicators.append(mov)
                streaming_count += mov

            out["streaming_services_count"] = streaming_count

            if service_indicators:
                total_services = pd.Series(0, index=out.index, dtype=int)
                for ind in service_indicators:
                    total_services += ind
                out["service_count"] = total_services
            else:
                out["service_count"] = 0

        # 2. Risk flags (Contract, Payment, Internet)
        if self.create_contract_flags:
            if "Contract" in out.columns:
                out["is_month_to_month"] = (
                    out["Contract"].astype(str) == "Month-to-month"
                ).astype(int)
            if "InternetService" in out.columns:
                out["is_fiber_optic"] = (
                    out["InternetService"].astype(str) == "Fiber optic"
                ).astype(int)
            if "PaymentMethod" in out.columns:
                out["is_electronic_check"] = (
                    out["PaymentMethod"].astype(str) == "Electronic check"
                ).astype(int)

        # 3. Tenure Cohorts
        if self.create_tenure_cohorts and "tenure" in out.columns:
            tenure_num = pd.to_numeric(out["tenure"], errors="coerce").fillna(0)
            bins = [-1, 12, 24, 48, 120]
            labels = ["0-12 mos", "13-24 mos", "25-48 mos", "49-72+ mos"]
            out["tenure_group"] = pd.cut(tenure_num, bins=bins, labels=labels).astype(str)

        return out

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Return output feature names for scikit-learn pipeline inspection."""
        if self.feature_names_out_ is not None:
            return np.array(self.feature_names_out_)
        if input_features is not None:
            return np.array(input_features)
        return np.array([])
