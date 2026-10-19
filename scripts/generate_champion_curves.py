"""Generate champion model held-out evaluation curves."""
from pathlib import Path

from predictiq.data.loader import load_and_split_isolated_evaluation_data
from predictiq.models.evaluation import plot_champion_heldout_curves
from predictiq.models.persistence import load_model_pipeline


def generate_curves():
    splits = load_and_split_isolated_evaluation_data()
    pipeline, meta = load_model_pipeline("models/churn_champion_model.joblib")
    probas = pipeline.predict_proba(splits.X_test)[:, 1]
    optimal_threshold = float(meta.get("optimal_decision_threshold", 0.34))

    out_path = Path("reports/figures/champion_heldout_curves.png")
    plot_champion_heldout_curves(
        y_true=splits.y_test,
        y_proba=probas,
        optimal_threshold=optimal_threshold,
        output_path=out_path,
        model_name=meta.get("model_name", "Random Forest (Calibrated)"),
        split_desc="Held-Out Test (Seed 2026, N=1,409)",
    )
    print(f"Generated champion curves at: {out_path.resolve()}")

if __name__ == "__main__":
    generate_curves()
