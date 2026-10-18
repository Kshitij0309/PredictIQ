"""Executable script to train and evaluate PredictIQ baseline models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project src to sys.path if running as standalone script
project_root = Path(__file__).resolve().parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from predictiq.models.train import run_baseline_training_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate PredictIQ baseline models.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional custom path to Telco churn CSV dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proportion of dataset reserved for held-out test set (default: 0.20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data splitting and model training (default: 42).",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds on training data (default: 5).",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow experiment logging.",
    )

    args = parser.parse_args()

    results = run_baseline_training_pipeline(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.seed,
        cv_splits=args.cv_splits,
        log_to_mlflow=not args.no_mlflow,
    )

    print("\n" + "=" * 70)
    print("PREDICTIQ BASELINE MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(results["comparison_dataframe"].to_string(index=False))
    print("=" * 70)
    print(f"Selected Best Model: {results['best_model_name']}")
    if results.get("saved_model_paths"):
        print(f"Model Artifact:      {results['saved_model_paths'][0]}")
        print(f"Metadata Artifact:   {results['saved_model_paths'][1]}")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
