# PredictIQ — Phase 4 Implementation Report
## Evaluation Protocol Correction, Advanced Modeling, Calibration, Threshold Optimization & Explainability

**Status**: Verified & Complete  
**Date**: 2026-10-18  
**Author**: PredictIQ Machine Learning Engineering Team  
**Prior Milestone**: Phase 3 Verified Complete (`commit b60e564`)  
**Active Test Suite**: 97 passing tests in 20.40s (100% pass rate)  
**Code Quality**: Ruff linter clean (0 errors)  
**Python Runtime**: Python 3.13.14 (AMD64, Windows)  

---

## 1. Executive Summary

Phase 4 of PredictIQ successfully addressed two core mandates:
1. **Part A (Evaluation Protocol Correction)**: Audited the Phase 3 evaluation workflow, retroactively classified prior test metrics as exploratory historical artifacts, and established strict evaluation independence. Model selection is now strictly driven by training-only Stratified 5-Fold Cross-Validation optimizing Average Precision (PR-AUC). A fresh, unobserved held-out evaluation split (Seed 2026, $N=1,409$) was locked and evaluated strictly once after model selection, probability calibration, and threshold optimization were frozen.
2. **Part B (Advanced Modeling & Production Optimization)**: Verified modern gradient boosting on Python 3.13 (LightGBM 4.7.0), executed hyperparameter optimization across candidate model families, applied Platt sigmoid probability calibration (reducing Brier score by 15.42%), derived an optimal decision threshold ($p^* = 0.3400$) from training out-of-fold predictions, and integrated TreeSHAP global and local explainability.

The resulting champion pipeline (`RandomForestClassifier` with Platt sigmoid calibration and frozen threshold $p^* = 0.3400$) achieved a final held-out **PR-AUC of 0.6684**, **ROC-AUC of 0.8428**, and **F1-score of 0.6231** (capturing **71.39% of churners** vs 57.22% under the default 0.50 threshold), serialized alongside complete runtime metadata to `models/churn_champion_model.joblib`.

---

## 2. Part A — Evaluation Protocol Audit and Correction

### 2.1 Audit Findings from Phase 3
In Phase 3, the training pipeline compared three baseline models (Dummy, Logistic Regression, and Random Forest) using both 5-fold cross-validation and evaluation on a held-out test split (Seed 42). The selection logic logged and compared test-set Average Precision alongside CV metrics. While Random Forest scored highest on both, choosing a champion model using held-out test metrics violates statistical independence and introduces subtle test-set overfitting (data snooping).

### 2.2 Corrective Protocol Implementation
To restore rigorous scientific and engineering separation:
1. **Exploratory Status**: All previously reported test metrics from Phase 3 (using Seed 42) are formally classified as exploratory historical benchmarks.
2. **Training-Only Model Selection**: Model selection is governed exclusively by the mean Average Precision (PR-AUC) across identical 5-fold stratified splits of the training partition (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
3. **Fresh Isolated Evaluation Split**: A fresh, documented held-out split seed (`ISOLATED_EVALUATION_SPLIT_SEED = 2026`) was created in [`src/predictiq/data/loader.py`](file:///D:/Projects/PredictIQ/src/predictiq/data/loader.py) via `load_and_split_isolated_evaluation_data()`.
   - Dataset: 7,043 total rows $\to$ 5,634 training rows (80.0%), 1,409 evaluation rows (20.0%).
   - Target distribution: Exactly preserved at 26.54% churn (1,495 positive train, 374 positive test).
   - Zero overlap: Enforced via index validation.
4. **Pre-Test Freezing**: Model architecture, hyperparameters, calibration parameters, and decision threshold are frozen on training data before evaluating on the isolated test set.
5. **Single Evaluation**: The test set is scored exactly once for final reporting. No tuning, feature pruning, or threshold adjustments use test data.
6. **Automated Protocol Enforcement**: Added dedicated tests in [`tests/test_evaluation_protocol.py`](file:///D:/Projects/PredictIQ/tests/test_evaluation_protocol.py) asserting split seed separation, identical fold comparisons, and data isolation.

---

## 3. Part B — Modeling, Tuning, Calibration, Thresholding & Explainability

### 3.1 Environment Verification & Candidate Models
Compatibility testing on Python 3.13.14 verified that `lightgbm>=4.0.0` and `shap>=0.45.0` run natively without wheels compilation issues.
Candidate models evaluated:
1. **Logistic Regression (ElasticNet / L2)**: Linear baseline with saga solver, serving as a transparent parametric benchmark.
2. **Random Forest Classifier**: Non-linear ensemble baseline with balanced class weights.
3. **LightGBM Classifier**: Gradient-boosted decision tree ensemble using histogram-based splitting, bounded depth, and leaf-wise growth.

Constructors were integrated into [`src/predictiq/models/baselines.py`](file:///D:/Projects/PredictIQ/src/predictiq/models/baselines.py).

### 3.2 Hyperparameter Optimization (`src/predictiq/models/tuning.py`)
Hyperparameter optimization was executed using `RandomizedSearchCV` across identical 5-fold stratified cross-validation splits on the 5,634 training records, optimizing Average Precision (`average_precision`). Final test data remained completely untouched.

#### Hyperparameter Search Spaces:
- **Logistic Regression**: `C` $\in [0.01, 10.0]$, `solver` $\in$ `['lbfgs', 'saga']`.
- **Random Forest**: `n_estimators` $\in [80, 200]$, `max_depth` $\in [4, 12]$, `min_samples_leaf` $\in [2, 10]$, `max_features` $\in ['sqrt', 'log2']$.
- **LightGBM**: `num_leaves` $\in [15, 63]$, `max_depth` $\in [3, 8]$, `learning_rate` $\in [0.01, 0.1]$, `n_estimators` $\in [80, 200]$, `min_child_samples` $\in [10, 40]$.

#### Cross-Validation Model Comparison Table:
| Candidate Model | CV PR-AUC (Mean) | CV PR-AUC (Std) | Tuning Runtime | Selected Hyperparameters |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest** | **0.6607** | **$\pm$ 0.0220** | 6.5s | `n_estimators=120, max_depth=8, min_samples_leaf=3, max_features='sqrt'` |
| **LightGBM** | 0.6598 | $\pm$ 0.0287 | 6.0s | `n_estimators=140, max_depth=4, num_leaves=15, min_child_samples=15, learning_rate=0.03` |
| **Logistic Regression** | 0.6559 | $\pm$ 0.0221 | 6.3s | `C=5.0, solver='saga'` |

**Selection Decision**: Random Forest achieved the highest cross-validation PR-AUC (0.6607) with lower fold variance ($\pm 0.0220$) than LightGBM ($\pm 0.0287$). In accordance with protocol Part A, **Random Forest was frozen as the champion model before final testing**.

---

### 3.3 Probability Calibration (`src/predictiq/models/calibration.py`)
Tree-based ensembles with `class_weight='balanced'` artificially displace raw predicted probabilities toward higher values to penalize false negatives, distorting confidence estimates.

Platt scaling was applied via `CalibratedClassifierCV(method='sigmoid', cv=5)` fitted strictly on training data folds.
- **Uncalibrated Brier Score**: 0.1602
- **Calibrated Brier Score**: 0.1354
- **Brier Score Improvement**: **+15.42% reduction in mean squared probability error**
- **Artifact**: Diagnostic reliability curve generated at [`reports/figures/calibration_curve.png`](file:///D:/Projects/PredictIQ/reports/figures/calibration_curve.png).

---

### 3.4 Decision Threshold Optimization (`src/predictiq/models/thresholds.py`)
Default classification thresholds ($p=0.50$) are arbitrary and ill-suited for imbalanced churn problems where missing a churner carries higher business risk than contacting a satisfied customer.

Using training out-of-fold (OOF) cross-validation predictions:
- Threshold sweep evaluated 99 candidate thresholds from $0.01$ to $0.99$.
- Optimization selected $p^* = 0.3400$, maximizing the empirical F1-score on training OOF predictions.
- **Training OOF Metrics at $p^* = 0.3400$**: F1 = 0.6381, Precision = 0.5653, Recall = 0.7324 (vs F1 = 0.6120, Recall = 0.5840 at $p=0.50$).
- **Cost Matrix Support**: The module implements [`BusinessCostMatrix`](file:///D:/Projects/PredictIQ/src/predictiq/models/thresholds.py) ($Cost = FP \times C_{FP} + FN \times C_{FN}$). In the absence of confirmed customer retention economics, no speculative dollar figures were fabricated; transparent trade-off curves were generated instead.
- **Artifact**: Threshold trade-off curve generated at [`reports/figures/threshold_tradeoff.png`](file:///D:/Projects/PredictIQ/reports/figures/threshold_tradeoff.png).
- **Frozen Threshold**: $p^* = 0.3400$ was locked prior to final test evaluation.

---

### 3.5 Single Final Evaluation on Isolated Held-Out Split (Seed 2026)

Evaluating the frozen calibrated champion model once on the locked held-out test set ($N=1,409$):

| Metric | Historical Exploratory (Seed 42, Uncalibrated) | Final Frozen Evaluation (Seed 2026, Calibrated) |
| :--- | :---: | :---: |
| **Split Seed** | 42 (Exploratory) | 2026 (Isolated & Locked) |
| **Test Set Size** | 1,409 | 1,409 |
| **PR-AUC (Average Precision)** | 0.6657 | **0.6684** |
| **ROC-AUC** | 0.8493 | **0.8428** |
| **Brier Score Loss** | 0.1610 | **0.1377** |
| **Decision Threshold** | 0.50 (Default) | **0.3400 (Optimized)** |
| **Recall (Churn Catch Rate)** | 59.89% | **71.39% (+11.5 pp)** |
| **Precision** | 65.69% | **55.28%** |
| **F1-Score** | 0.6266 | **0.6231** |

#### Comparison of Threshold Impacts on Final Held-Out Test Set:
| Threshold Policy | Threshold | Test Precision | Test Recall | Test F1-Score | True Positives | False Negatives | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Default | 0.5000 | 0.6350 | 0.5722 | 0.6020 | 214 | 160 | 123 |
| **Optimized (Frozen)** | **0.3400** | **0.5528** | **0.7139** | **0.6231** | **267** | **107** | **216** |
| **Net Difference** | **-0.1600** | **-8.22 pp** | **+14.17 pp** | **+2.11 pp** | **+53 Churners Caught** | **-53 Misses** | **+93 Inquiries** |

*Conclusion*: Lowering the threshold to $p^*=0.3400$ enables retention teams to intercept 53 additional churners out of 374 (a 24.8% relative reduction in missed churners), demonstrating the utility of data-driven threshold selection.

---

### 3.6 Model Explainability (`src/predictiq/models/explainability.py`)
TreeSHAP was applied to explain the model's predictive structure without exposing customer identifiers or claiming causal influence.

#### Top 10 Global Predictive Features (by Mean Absolute SHAP Value):
1. `is_month_to_month` (Domain feature: Contract == Month-to-month)
2. `Contract_Month-to-month` (Categorical one-hot)
3. `tenure` (Customer lifetime in months; longer tenure reduces churn risk)
4. `InternetService_Fiber optic` (Higher churn propensity associated with service delivery issues)
5. `is_fiber_optic` (Domain feature flag)
6. `OnlineSecurity_No` (Lack of cybersecurity add-on increases churn)
7. `Contract_Two year` (Long-term commitment strongly protects retention)
8. `tenure_group_0-12 mos` (New customer onboarding risk window)
9. `TotalCharges` (Cumulative spend proxy)
10. `TechSupport_No` (Absence of technical support increases churn probability)

#### Explainability Visual Artifacts:
- **Global Summary Plot**: [`reports/figures/shap_summary.png`](file:///D:/Projects/PredictIQ/reports/figures/shap_summary.png) showing feature attribution distributions and directional impact.
- **High-Risk Archetype Waterfall**: [`reports/figures/shap_local_high_risk.png`](file:///D:/Projects/PredictIQ/reports/figures/shap_local_high_risk.png) illustrating a month-to-month fiber optic customer with no tech support ($P(\text{churn}) = 0.81$).
- **Low-Risk Archetype Waterfall**: [`reports/figures/shap_local_low_risk.png`](file:///D:/Projects/PredictIQ/reports/figures/shap_local_low_risk.png) illustrating a two-year contract DSL customer with multi-service bundle ($P(\text{churn}) = 0.04$).

---

## 4. Engineering Quality & Verification

### 4.1 Production Model Persistence & Inference Integration
The champion model was saved using the unified packaging convention:
- **Pipeline Binary**: [`models/churn_champion_model.joblib`](file:///D:/Projects/PredictIQ/models/churn_champion_model.joblib) (Fitted preprocessor + CalibratedClassifierCV).
- **Metadata**: [`models/churn_champion_model_metadata.json`](file:///D:/Projects/PredictIQ/models/churn_champion_model_metadata.json) recording optimal threshold ($p^*=0.3400$), calibration statistics, SHAP feature rankings, and evaluation seed.
- **Inference Engine**: [`ChurnPredictor`](file:///D:/Projects/PredictIQ/src/predictiq/inference/predictor.py) dynamically loads `decision_threshold` from metadata, ensuring scoring parity between training and production.

### 4.2 Automated Test Suite
Pytest suite expanded from 86 to **97 tests** across 13 test modules:
- `tests/test_evaluation_protocol.py` (2 tests): Asserts seed independence and identical fold comparisons.
- `tests/test_tuning.py` (3 tests): Verifies search spaces, CV execution, and determinism.
- `tests/test_calibration.py` (2 tests): Tests Platt scaling, Brier score reduction, and reliability plots.
- `tests/test_thresholds.py` (3 tests): Tests OOF sweep, cost matrix calculation, and trade-off curves.
- `tests/test_explainability.py` (1 test): Verifies SHAP tree attribution and archetype generation.
- All Phase 1, 2, and 3 regression tests maintained (100% pass rate).

### 4.3 Static Analysis & Code Quality
- **Ruff Linter**: Executed across all modules. 0 errors, 0 warnings.
- **Headless Plotting**: Matplotlib backend explicitly forced to `Agg` across all modules to eliminate Windows Tkinter dependencies.

---

## 5. Experiment Tracking & Infrastructure Status
- **MLflow Server**: Local tracking mode (`file:///D:/Projects/PredictIQ/mlruns`). Remote server at `http://localhost:5000` is offline (Docker host not running).
- **Logged Experiment**: `predictiq-churn-baselines`
- **Logged Run**: `champion_random_forest_1783` (ID: `97b68ccc677d4eb38298f2e7d8a64d2f`).
- **Logged Artifacts**: Model pipeline, metadata dictionary, CV comparison CSV, evaluation plots, and SHAP summaries.

---

## 6. Limitations and Next Steps (Phase 5)

### Identified Limitations:
1. **Explainability vs Causality**: SHAP values indicate associative predictive weight within the ensemble model. They do not represent causal effects (e.g., forcing a customer onto a two-year contract will not necessarily reduce their underlying churn intent).
2. **Static Thresholding**: The threshold $p^* = 0.3400$ was optimized for maximum F1-score across all customer segments. Segment-specific thresholds (e.g., differentiating high-LTV vs low-LTV customers) may yield superior business value once financial data is integrated.
3. **Infrastructure**: MLflow tracking operated in local file mode due to inactive Docker daemon.

### Phase 5 Focus:
- Production REST API service using FastAPI.
- Containerized deployment with Docker Compose.
- Automated drift detection and monitoring endpoints.
