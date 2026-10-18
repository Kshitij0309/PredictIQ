"""PredictIQ Model Layer: training, evaluation, baselines, and persistence."""

from predictiq.models.baselines import (
    build_dummy_baseline,
    build_logistic_regression_baseline,
    build_random_forest_baseline,
    get_all_baselines,
)
from predictiq.models.evaluation import (
    ModelEvaluationResult,
    build_comparison_dataframe,
    evaluate_model,
    plot_confusion_matrices,
    plot_pr_curves,
    plot_roc_curves,
    save_evaluation_artifacts,
)
from predictiq.models.persistence import (
    ModelPersistenceError,
    load_model_pipeline,
    save_model_pipeline,
)
from predictiq.models.train import (
    run_baseline_training_pipeline,
    run_cross_validation_evaluation,
)

__all__ = [
    "build_dummy_baseline",
    "build_logistic_regression_baseline",
    "build_random_forest_baseline",
    "get_all_baselines",
    "evaluate_model",
    "ModelEvaluationResult",
    "build_comparison_dataframe",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_confusion_matrices",
    "save_evaluation_artifacts",
    "save_model_pipeline",
    "load_model_pipeline",
    "ModelPersistenceError",
    "run_baseline_training_pipeline",
    "run_cross_validation_evaluation",
]
