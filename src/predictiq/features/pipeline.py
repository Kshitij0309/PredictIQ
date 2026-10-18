"""Feature preprocessing and end-to-end pipeline creation for PredictIQ.

Constructs scikit-learn ColumnTransformers and Pipelines ensuring strict leakage prevention,
explicit categorical encoding with unknown value handling, and robust missing value imputation.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictiq.features.transformers import DomainFeatureEngineer

logger = logging.getLogger(__name__)

# Base categorical columns from raw/cleaned dataset
BASE_CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
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
    "tenure_group",  # Added by DomainFeatureEngineer
]

# Base continuous and engineered numeric features
BASE_NUMERIC_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "service_count",
    "streaming_services_count",
    "has_tech_support",
    "has_online_security",
    "is_month_to_month",
    "is_fiber_optic",
    "is_electronic_check",
]


def create_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Create a ColumnTransformer for preprocessing numeric and categorical columns.

    Leakage & Safety Controls:
    - Numeric: Imputed with median, optionally scaled with StandardScaler.
    - Categorical: Imputed with most frequent, encoded with OneHotEncoder(handle_unknown='ignore').
      Unseen categories at test or inference time will produce all-zero indicator rows without errors.
    - Remainder: 'drop' to prevent arbitrary or uninspected columns from passing into models.

    Args:
        numeric_features: List of numeric column names. Defaults to BASE_NUMERIC_FEATURES.
        categorical_features: List of categorical column names. Defaults to BASE_CATEGORICAL_FEATURES.
        scale_numeric: Whether to apply StandardScaler to numerical features.

    Returns:
        ColumnTransformer: Configured preprocessor.
    """
    num_cols = numeric_features if numeric_features is not None else list(BASE_NUMERIC_FEATURES)
    cat_cols = (
        categorical_features if categorical_features is not None else list(BASE_CATEGORICAL_FEATURES)
    )

    # Numeric sub-pipeline
    num_steps: list = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=num_steps)

    # Categorical sub-pipeline
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_model_pipeline(
    classifier: BaseEstimator,
    scale_numeric: bool = True,
    create_service_counts: bool = True,
    create_contract_flags: bool = True,
    create_tenure_cohorts: bool = True,
) -> Pipeline:
    """Assemble a full, leakage-safe pipeline from raw features to model predictions.

    Pipeline structure:
    1. 'features': DomainFeatureEngineer (stateless feature augmentation)
    2. 'preprocessor': ColumnTransformer (learns medians, scalers, and OHE categories from train data)
    3. 'classifier': Scikit-learn estimator / classifier

    Args:
        classifier: Scikit-learn estimator implementing fit, predict (and optionally predict_proba).
        scale_numeric: Whether to standardize continuous features.
        create_service_counts: Enable service count feature engineering.
        create_contract_flags: Enable contract risk flags.
        create_tenure_cohorts: Enable tenure cohort categorization.

    Returns:
        Pipeline: Complete executable scikit-learn pipeline.
    """
    feature_engineer = DomainFeatureEngineer(
        create_service_counts=create_service_counts,
        create_contract_flags=create_contract_flags,
        create_tenure_cohorts=create_tenure_cohorts,
    )

    preprocessor = create_preprocessor(scale_numeric=scale_numeric)

    pipeline = Pipeline(
        steps=[
            ("features", feature_engineer),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline
