# PredictIQ — Phase 3 Implementation Report
## Feature Engineering, Baseline Modeling & MLflow Experiment Tracking

**Status**: Verified & Complete  
**Date**: 2026-10-18  
**Author**: PredictIQ Machine Learning Engineering Team  
**Prior Milestone**: Phase 2 Verified Complete (`commit 1f6447a`)  
**Active Test Suite**: 86 passing tests in 10.7s  

---

## 1. Initial Repository State

Prior to Phase 3 implementation, PredictIQ possessed:
- Verified Python package scaffold with typed configuration (`src/predictiq/config.py`).
- Data acquisition, non-destructive schema validation, and cleaning modules (`src/predictiq/data/`).
- Verified raw IBM Telco Customer Churn dataset (970,457 bytes, SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`).
- Interim cleaned dataset (`data/interim/Telco-Customer-Churn-cleaned.csv`, 7,043 rows $\times$ 21 columns).
- Fully executed exploratory analysis notebook (`notebooks/01_eda_telco_churn.ipynb`) and Dataset Card (`docs/data/DATASET_CARD.md`).
- 63 passing unit and integration tests across configuration, ingestion, and validation.
- Docker Desktop host daemon stopped; remote infrastructure offline.

---

## 2. Files Added and Modified

### Source Modules (`src/predictiq/`)
- [`src/predictiq/data/loader.py`](file:///D:/Projects/PredictIQ/src/predictiq/data/loader.py): Structured dataset loader, target extraction (`Churn` mapped to 1/0), identifier exclusion (`customerID`), and reproducible stratified train/test splitting (`DatasetSplits`).
- [`src/predictiq/features/transformers.py`](file:///D:/Projects/PredictIQ/src/predictiq/features/transformers.py): `DomainFeatureEngineer` implementing active service counts, streaming counts, high-retention feature flags (`has_tech_support`, `has_online_security`), contract risk flags (`is_month_to_month`, `is_fiber_optic`), and tenure cohorts.
- [`src/predictiq/features/pipeline.py`](file:///D:/Projects/PredictIQ/src/predictiq/features/pipeline.py): `create_preprocessor` and `build_model_pipeline` chaining imputer, standard scaler, `OneHotEncoder(handle_unknown='ignore')`, and classifier.
- [`src/predictiq/models/baselines.py`](file:///D:/Projects/PredictIQ/src/predictiq/models/baselines.py): Constructors for naive `DummyClassifier`, interpretable `LogisticRegression`, and `RandomForestClassifier`.
- [`src/predictiq/models/evaluation.py`](file:///D:/Projects/PredictIQ/src/predictiq/models/evaluation.py): `evaluate_model`, `ModelEvaluationResult`, comparison table generator, and plotting routines for ROC curves, PR curves, and confusion matrices.
- [`src/predictiq/models/persistence.py`](file:///D:/Projects/PredictIQ/src/predictiq/models/persistence.py): Joblib serialization and metadata extraction for fitted pipelines.
- [`src/predictiq/models/train.py`](file:///D:/Projects/PredictIQ/src/predictiq/models/train.py): End-to-end training orchestrator executing 5-fold CV, held-out evaluation, artifact generation, MLflow logging, and best-model selection.
- [`src/predictiq/tracking/manager.py`](file:///D:/Projects/PredictIQ/src/predictiq/tracking/manager.py): Resilient MLflow tracking manager supporting remote server connection with automatic local file-based fallback (`./mlruns`).
- [`src/predictiq/inference/predictor.py`](file:///D:/Projects/PredictIQ/src/predictiq/inference/predictor.py): Production inference class `ChurnPredictor` with strict input schema validation, batch scoring, and business risk tiering.

### Scripts & Entry Points
- [`scripts/train_baseline.py`](file:///D:/Projects/PredictIQ/scripts/train_baseline.py): CLI interface for training, cross-validation, and artifact generation.

### Automated Test Suite (`tests/`)
- [`tests/test_data_loader.py`](file:///D:/Projects/PredictIQ/tests/test_data_loader.py): 7 tests verifying split reproducibility, stratification balance, zero index overlap, and target validation.
- [`tests/test_features.py`](file:///D:/Projects/PredictIQ/tests/test_features.py): 6 tests verifying feature engineering outputs, service count logic, unseen category handling, and imputer behavior.
- [`tests/test_model_evaluation.py`](file:///D:/Projects/PredictIQ/tests/test_model_evaluation.py): 4 tests verifying metric extraction, probability checks, and artifact export.
- [`tests/test_inference.py`](file:///D:/Projects/PredictIQ/tests/test_inference.py): 4 tests verifying model serialization bit-level consistency, input schema validation, and risk tier assignments.
- [`tests/test_tracking.py`](file:///D:/Projects/PredictIQ/tests/test_tracking.py): 2 tests verifying MLflow tracking initialization, local directory fallback, and offline run logging.

### Documentation & Configuration
- [`pyproject.toml`](file:///D:/Projects/PredictIQ/pyproject.toml): Added `scikit-learn>=1.4.0`, `joblib>=1.3.0`, `mlflow>=2.10.0` to core dependencies; `ruff>=0.4.0` to dev; database drivers to `db` extra.
- Updated [`README.md`](file:///D:/Projects/PredictIQ/README.md), [`docs/ARCHITECTURE.md`](file:///D:/Projects/PredictIQ/docs/ARCHITECTURE.md), [`docs/CHANGELOG.md`](file:///D:/Projects/PredictIQ/docs/CHANGELOG.md), [`CURRENT_STATE.md`](file:///D:/Projects/PredictIQ/CURRENT_STATE.md), and [`NEXT_TASK.md`](file:///D:/Projects/PredictIQ/NEXT_TASK.md).

---

## 3. Feature Engineering Decisions and Rationale

The feature engineering layer was implemented in [`DomainFeatureEngineer`](file:///D:/Projects/PredictIQ/src/predictiq/features/transformers.py) based on empirical findings from Phase 2 EDA:

| Feature Name | Type | Definition | Business / Empirical Rationale |
| :--- | :--- | :--- | :--- |
| `service_count` | Integer [0–9] | Sum of 9 active service indicators | Accounts subscribing to multiple services exhibit substantially higher retention and switching costs. |
| `streaming_services_count` | Integer [0–2] | `StreamingTV` + `StreamingMovies` | Quantifies entertainment bundle depth. |
| `has_tech_support` | Binary [0/1] | `TechSupport == 'Yes'` | In Phase 2 EDA, tech support reduced churn by ~2.7x (15.16% vs 41.64%). |
| `has_online_security` | Binary [0/1] | `OnlineSecurity == 'Yes'` | Reduced churn rate from 41.77% to 14.61%. |
| `is_month_to_month` | Binary [0/1] | `Contract == 'Month-to-month'` | The strongest single bivariate driver ($\chi^2 = 1184.60, \text{Cramér's } V = 0.410$, churn rate 42.71% vs 2.83% for 2-year). |
| `is_fiber_optic` | Binary [0/1] | `InternetService == 'Fiber optic'` | Highest-churn product segment (41.89%) characterized by elevated monthly charges ($91.50). |
| `is_electronic_check` | Binary [0/1] | `PaymentMethod == 'Electronic check'` | Payment friction indicator (45.3% churn rate). |
| `tenure_group` | Categorical | Binned `tenure` into [0–12, 13–24, 25–48, 49–72+] | Captures non-linear lifecycle risk tiers. |

### Deliberate Omission of `TotalCharges / MonthlyCharges`
The prompt instructed: *"In particular, `TotalCharges / MonthlyCharges` is approximately a proxy for tenure in this dataset and may be redundant or unstable for zero monthly charges. Do not add it without explicitly evaluating its validity and incremental value."*

**Evaluation Finding**:
In subscription telecom billing, total revenue accrued is governed by $\text{TotalCharges} \approx \text{MonthlyCharges} \times \text{tenure}$. Computing their quotient results in an empirical proxy for `tenure` (Pearson correlation $r > 0.98$). Furthermore, for brand-new accounts with $\text{tenure} = 0$, `MonthlyCharges` can create mathematical instability, while offering zero independent orthogonal variance. It was therefore deliberately excluded.

---

## 4. Split Strategy and Leakage Safeguards

1. **Stratified Split**:
   - Split ratio: 80% train ($N = 5,634$), 20% held-out test ($N = 1,409$).
   - Random seed: Fixed at `42`.
   - Stratification target: `Churn` binary vector.
   - Resulting churn prevalence: Train = **26.54%** ($N = 1,495$), Test = **26.54%** ($N = 374$). Class balance was preserved with zero skew.
2. **Identifier Exclusion**:
   - `customerID` is stripped from feature matrices before pipeline ingestion, preventing arbitrary key memorization.
3. **Leakage Prevention**:
   - Feature imputers (`median`, `most_frequent`), standard scalers (`mean`, `std`), and categorical encoders (`OneHotEncoder`) are fit **strictly on training folds**.
   - Held-out test set remained isolated until final evaluation.

---

## 5. Model Configurations

All models were evaluated on the identical stratified training and test partitions:

1. **Naive Baseline (`DummyClassifier`)**:
   - Strategy: `most_frequent`
   - Purpose: Establishes majority-class empirical floor.
2. **Linear Baseline (`LogisticRegression`)**:
   - Solver: `lbfgs`, Max iterations: `1000`, Regularization: $C = 1.0$ (L2 penalty)
   - Preprocessing: `StandardScaler` on continuous features, `OneHotEncoder(handle_unknown='ignore')` on categoricals
   - Class weight: `balanced` (adjusts loss penalty inversely proportional to class frequencies)
   - Random state: `42`
3. **Tree-Based Baseline (`RandomForestClassifier`)**:
   - Estimators: `100`, Max depth: `8`, Min samples leaf: `5`, Class weight: `balanced`
   - Bounded depth and leaf size explicitly selected to prevent leaf-level memorization on 5,634 training rows.
   - Random state: `42`

---

## 6. Evaluation Framework & Metric Definitions

Because customer churn is an imbalanced binary classification problem (positive class prevalence: 26.54%), standard accuracy is misleading (a naive model predicting all "No" achieves 73.46% accuracy with zero utility).

- **Primary Comparison Metric**: **Average Precision / PR-AUC** (`average_precision_score`). Evaluates precision across all decision thresholds for the minority churn class.
- **Secondary Metrics**:
  - **ROC-AUC**: Overall ranking capability across false positive rates.
  - **Recall**: Proportion of actual churners correctly detected ($TP / (TP + FN)$).
  - **Precision**: Proportion of predicted churners that actually churned ($TP / (TP + FP)$).
  - **F1-Score**: Harmonic mean of precision and recall.
  - **Brier Score Loss**: Mean squared probability error (calibration accuracy).

---

## 7. Actual Measured Model Comparison Results

### Held-Out Test Set Performance ($N = 1,409$, Positive Cases = 374, Negative Cases = 1,035)

| Model Name | Model Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Average Precision (PR-AUC)* | Brier Score | 5-Fold CV PR-AUC | 5-Fold CV ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (Champion) | `RandomForestClassifier` | **0.7502** | **0.5194** | 0.7888 | **0.6263** | **0.8420** | **0.6471** | **0.1630** | 0.6633 $\pm$ 0.016 | **0.8480** |
| **Logistic Regression** | `LogisticRegression` | 0.7339 | 0.4992 | **0.7941** | 0.6130 | 0.8417 | 0.6348 | 0.1675 | **0.6639** $\pm$ 0.013 | 0.8463 |
| **Dummy (Most Frequent)** | `DummyClassifier` | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2654 | 0.2654 | 0.2654 $\pm$ 0.000 | 0.5000 |

*\*Primary metric for model selection.*

### Confusion Matrix Breakdown (Test Set)
- **Dummy Classifier**:
  - True Negatives: 1,035 | False Positives: 0
  - False Negatives: 374 | True Positives: 0
- **Logistic Regression**:
  - True Negatives: 737 | False Positives: 298
  - False Negatives: 77 | True Positives: **297** (catches 79.4% of all churners)
- **Random Forest**:
  - True Negatives: 762 | False Positives: 273
  - False Negatives: 79 | True Positives: **295** (catches 78.9% of all churners with fewer false alarms)

### Key Observations:
1. **Surpassing Naive Baseline**:
   - Both machine learning models surpass the naive baseline by **+0.38 PR-AUC** (a **2.44x** improvement over the 0.2654 baseline prevalence).
2. **Best Model Selection**:
   - **Random Forest** achieved the highest test Average Precision (**0.6471**), the highest F1-score (**0.6263**), the highest accuracy (**75.02%**), and the lowest Brier score (**0.1630**).
3. **Precision / Recall Trade-off**:
   - With `class_weight='balanced'`, both models prioritize recall (~79%), ensuring approximately 4 out of 5 at-risk customers are flagged for retention campaigns at a precision of ~52%.

---

## 8. MLflow Experiment Tracking Status

- **Tracking Mode**: `local_directory`
- **Active Tracking URI**: `file:///D:/Projects/PredictIQ/mlruns`
- **Experiment Name**: `predictiq-churn-baselines`
- **Logged Runs**:
  - Run 1 (`Dummy Classifier`): ID `543579e7c0d341418320906336ecd17b`
  - Run 2 (`Logistic Regression`): ID `f67056aa6fb8427e82a61dd31065d73e`
  - Run 3 (`Random Forest`): ID `3b4331791a1142dab725e1f4bd499841`
- **Logged Entities**:
  - Model hyperparameters (`n_estimators`, `max_depth`, `C`, `class_weight`, `scale_numeric`).
  - Evaluation metrics (`test_accuracy`, `test_precision`, `test_recall`, `test_f1`, `test_roc_auc`, `test_average_precision_pr_auc`, `cv_mean_pr_auc`, etc.).
  - Artifacts: Comparison CSV, JSON, ROC curve plot, PR curve plot, confusion matrix plot.
  - Model artifacts: Scikit-learn pipelines with inferred signatures and input examples.

---

## 9. Model Persistence and Inference Smoke Test

The winning `RandomForestClassifier` pipeline was persisted using joblib:
- Model Binary: [`models/churn_baseline_model.joblib`](file:///D:/Projects/PredictIQ/models/churn_baseline_model.joblib)
- Model Metadata: [`models/churn_baseline_model_metadata.json`](file:///D:/Projects/PredictIQ/models/churn_baseline_model_metadata.json)

### Inference Smoke Test Results
Executed via `ChurnPredictor.from_saved_model()`:
- **Test Record 1** (High Risk profile: Month-to-month, Fiber Optic, 2 months tenure, electronic check):
  - Predicted Churn: **Yes**
  - Churn Probability: **0.8777**
  - Assigned Risk Tier: **High Risk**
- **Test Record 2** (Low Risk profile: Two-year contract, DSL, 65 months tenure, tech support, autopay):
  - Predicted Churn: **No**
  - Churn Probability: **0.0437**
  - Assigned Risk Tier: **Low Risk**

---

## 10. Automated Test Results

Executed complete test suite across all project modules:
```powershell
& "D:\Projects\PredictIQ\.venv\Scripts\python.exe" -m pytest -v
```
**Outcome**:
```
collected 86 items

tests\test_config.py ..........................................          [ 48%]
tests\test_data_ingestion.py .........                                   [ 59%]
tests\test_data_loader.py .......                                        [ 67%]
tests\test_data_validation.py ............                               [ 81%]
tests\test_features.py ......                                            [ 88%]
tests\test_inference.py ....                                             [ 93%]
tests\test_model_evaluation.py ....                                      [ 97%]
tests\test_tracking.py ..                                                [100%]

======================= 86 passed, 1 warning in 10.69s ========================
```

Executed Ruff linter:
```powershell
& "D:\Projects\PredictIQ\.venv\Scripts\ruff.exe" check .
```
**Outcome**: `All checks passed!`

---

## 11. Known Limitations

1. **Docker Daemon Offline**: Remote PostgreSQL and MLflow containers were not run on the host. Local directory tracking verified and functional.
2. **Threshold Tuning**: Baselines currently operate at standard probability threshold $p = 0.50$ with balanced weighting. Business decision threshold tuning belongs in Phase 4.
3. **Cross-Sectional Dataset**: Churn timing cannot be modeled with time-to-event survival analysis without transactional event timestamp streams.

---

## 12. Reproduction Commands (Windows PowerShell)

```powershell
# 1. Activate virtual environment
& "D:\Projects\PredictIQ\.venv\Scripts\Activate.ps1"

# 2. Run automated test suite
pytest -v

# 3. Execute baseline model training and MLflow tracking
python scripts/train_baseline.py

# 4. Run inference smoke test on saved champion model
python -c "
from predictiq.inference import ChurnPredictor
predictor = ChurnPredictor.from_saved_model()
print('Predictor loaded successfully!')
"
```

---

## 13. Recommended Next Phase: Phase 4

**Phase 4: Hyperparameter Optimization, Threshold Calibration & Champion Model Selection**
- Systematic search across gradient boosting candidates (LightGBM, XGBoost, CatBoost).
- Decision threshold calibration optimizing financial cost matrices (retention campaign cost vs. customer lifetime value salvage).
- Model explainability (SHAP global importance and local waterfall plots).
