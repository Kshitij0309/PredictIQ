"""MLflow experiment tracking manager for PredictIQ.

Supports both remote MLflow tracking servers and automatic local file-based fallback (./mlruns).
Logs parameters, metrics, evaluation artifacts, model signatures, and input examples.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline

from predictiq.config import get_config
from predictiq.models.evaluation import ModelEvaluationResult

logger = logging.getLogger(__name__)

# Silence MLflow assistant hint in Python 3.13 if not set
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# Allow local filesystem / SQLite fallback tracking without deprecation aborts
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


@dataclass
class MLflowRunResult:
    """Outcome of an MLflow logging operation."""
    run_id: str
    experiment_id: str
    experiment_name: str
    tracking_uri: str
    tracking_mode: str  # "remote_server" or "local_directory"
    status: str  # "logged" or "failed"
    error_message: Optional[str] = None


class MLflowTrackingManager:
    """Manages MLflow lifecycle, run creation, and artifact logging with resilient fallbacks."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        local_fallback_dir: Optional[Path | str] = None,
    ) -> None:
        config = get_config()

        # 1. Resolve tracking URI
        env_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.configured_uri = tracking_uri or env_uri or config.mlflow.tracking_uri
        self.experiment_name = (
            experiment_name
            or os.getenv("MLFLOW_EXPERIMENT_NAME")
            or "predictiq-churn-baselines"
        )

        # 2. Determine local directory fallback
        root_dir = config.paths.root_dir
        self.local_dir = (
            Path(local_fallback_dir).resolve()
            if local_fallback_dir
            else (root_dir / "mlruns").resolve()
        )

        # 3. Determine active tracking mode
        self.active_tracking_uri, self.tracking_mode = self._resolve_active_tracking_target(
            self.configured_uri, self.local_dir
        )

        # 4. Initialize MLflow
        self._initialize_mlflow()

    @staticmethod
    def _is_server_reachable(uri: str, timeout_sec: float = 1.0) -> bool:
        """Check whether a remote HTTP/HTTPS MLflow server is responding."""
        if not (uri.startswith("http://") or uri.startswith("https://")):
            return False
        try:
            req = urllib.request.Request(
                uri,
                headers={"User-Agent": "PredictIQ-HealthCheck"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=timeout_sec):
                return True
        except Exception:
            return False

    def _resolve_active_tracking_target(
        self,
        configured_uri: str,
        local_dir: Path,
    ) -> tuple[str, str]:
        """Determine whether to use configured server or fallback to local directory."""
        if configured_uri.startswith("http://") or configured_uri.startswith("https://"):
            if self._is_server_reachable(configured_uri):
                logger.info(f"Connected to remote MLflow tracking server at: {configured_uri}")
                return configured_uri, "remote_server"
            else:
                logger.warning(
                    f"MLflow server at '{configured_uri}' is unreachable. "
                    f"Falling back to local file tracking at: {local_dir}"
                )
                local_dir.mkdir(parents=True, exist_ok=True)
                local_uri = local_dir.as_uri()
                return local_uri, "local_directory"
        else:
            # File or SQLite URI
            return configured_uri, "local_directory"

    def _initialize_mlflow(self) -> None:
        """Configure MLflow global tracking URI and experiment."""
        try:
            mlflow.set_tracking_uri(self.active_tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(
                f"MLflow initialized: experiment='{self.experiment_name}', "
                f"mode='{self.tracking_mode}', uri='{self.active_tracking_uri}'"
            )
        except Exception as err:
            logger.warning(f"Failed to initialize MLflow experiment: {err}")

    def log_model_run(
        self,
        model_name: str,
        pipeline: Pipeline,
        eval_result: ModelEvaluationResult,
        hyperparameters: Dict[str, Any],
        X_sample: pd.DataFrame,
        dataset_meta: Optional[Dict[str, Any]] = None,
        artifact_paths: Optional[Dict[str, Path]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> MLflowRunResult:
        """Log a complete training run to MLflow."""
        run_name = f"{model_name.replace(' ', '_').lower()}_{int(np.random.randint(1000, 9999))}"

        try:
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                exp_id = run.info.experiment_id

                # 1. Tags
                base_tags = {
                    "model_name": model_name,
                    "model_type": eval_result.model_type,
                    "tracking_mode": self.tracking_mode,
                    "platform": "PredictIQ",
                    "target": "Churn",
                    "task": "binary_classification",
                }
                if tags:
                    base_tags.update(tags)
                mlflow.set_tags(base_tags)

                # 2. Hyperparameters
                clean_params: Dict[str, Any] = {}
                for k, v in hyperparameters.items():
                    # Flatten complex values into strings for MLflow param limits
                    clean_params[str(k)] = str(v) if not isinstance(v, (int, float, bool, str)) else v
                mlflow.log_params(clean_params)

                # 3. Dataset metadata
                if dataset_meta:
                    for k, v in dataset_meta.items():
                        if isinstance(v, (str, int, float, bool)):
                            mlflow.set_tag(f"data.{k}", str(v))

                # 4. Metrics
                metrics_to_log = {
                    "test_accuracy": eval_result.accuracy,
                    "test_precision": eval_result.precision,
                    "test_recall": eval_result.recall,
                    "test_f1": eval_result.f1_score,
                }
                if eval_result.roc_auc is not None:
                    metrics_to_log["test_roc_auc"] = eval_result.roc_auc
                if eval_result.average_precision is not None:
                    metrics_to_log["test_average_precision_pr_auc"] = eval_result.average_precision
                if eval_result.brier_score is not None:
                    metrics_to_log["test_brier_score"] = eval_result.brier_score
                if eval_result.cv_mean_pr_auc is not None:
                    metrics_to_log["cv_mean_pr_auc"] = eval_result.cv_mean_pr_auc
                if eval_result.cv_std_pr_auc is not None:
                    metrics_to_log["cv_std_pr_auc"] = eval_result.cv_std_pr_auc
                if eval_result.cv_mean_roc_auc is not None:
                    metrics_to_log["cv_mean_roc_auc"] = eval_result.cv_mean_roc_auc
                if eval_result.cv_std_roc_auc is not None:
                    metrics_to_log["cv_std_roc_auc"] = eval_result.cv_std_roc_auc

                mlflow.log_metrics(metrics_to_log)

                # 5. Artifacts (plots and json)
                if artifact_paths:
                    for art_name, art_path in artifact_paths.items():
                        p = Path(art_path)
                        if p.is_file():
                            mlflow.log_artifact(str(p), artifact_path="evaluation_artifacts")

                # 6. Model signature & model logging
                try:
                    sample_input = X_sample.head(3)
                    sample_pred = pipeline.predict(sample_input)
                    signature = infer_signature(sample_input, sample_pred)

                    mlflow.sklearn.log_model(
                        sk_model=pipeline,
                        name="model",
                        serialization_format="cloudpickle",
                        signature=signature,
                        input_example=sample_input,
                    )
                except Exception as log_err:
                    logger.warning(f"Could not log sklearn model artifact: {log_err}")

                logger.info(f"Successfully logged run '{run_name}' (ID: {run_id}) to MLflow")
                return MLflowRunResult(
                    run_id=run_id,
                    experiment_id=exp_id,
                    experiment_name=self.experiment_name,
                    tracking_uri=self.active_tracking_uri,
                    tracking_mode=self.tracking_mode,
                    status="logged",
                )

        except Exception as err:
            logger.error(f"Failed to log run for model '{model_name}': {err}")
            return MLflowRunResult(
                run_id="",
                experiment_id="",
                experiment_name=self.experiment_name,
                tracking_uri=self.active_tracking_uri,
                tracking_mode=self.tracking_mode,
                status="failed",
                error_message=str(err),
            )
