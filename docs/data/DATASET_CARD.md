# PredictIQ Dataset Card: IBM Telco Customer Churn

## 1. Dataset Overview
- **Dataset Name:** IBM Telco Customer Churn
- **Domain:** Telecommunications / Predictive Intelligence / Customer Retention
- **Primary Task:** Binary classification predicting customer churn (`Churn` = 'Yes' / 'No')
- **Format:** Comma-Separated Values (`.csv`)
- **Dimensions:** 7,043 observations × 21 variables (20 features + 1 target)
- **Class Balance:**
  - Retained (`No`): 5,174 (73.46%)
  - Churned (`Yes`): 1,869 (26.54%)
  - Imbalance Ratio: ~2.77:1

---

## 2. Provenance & Cryptographic Identity
- **Upstream Repository:** [IBM Telco Customer Churn on ICP4D](https://github.com/IBM/telco-customer-churn-on-icp4d)
- **Source URL:** `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Retrieval Date:** 2026-10-18
- **License:** Apache License 2.0 (Permissive for commercial and non-commercial research/application use)
- **File Size:** 970,457 bytes (~947.7 KB)
- **Cryptographic Digest (SHA-256):** `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- **Storage Policy:** Stored locally under `data/raw/Telco-Customer-Churn.csv` (strictly excluded from Git tracking via `.gitignore`). Provenance metadata tracked under `data/metadata/Telco-Customer-Churn_metadata.json`.

---

## 3. Variable Dictionary & Data Types

| Variable Name | Storage Type | Canonical Type | Description | Permitted / Observed Values |
|---|---|---|---|---|
| `customerID` | String | Identifier | Unique customer identifier | e.g. `7590-VHVEG` (Unique across all 7,043 rows) |
| `gender` | Categorical | Demographic | Customer gender identity | `Female`, `Male` |
| `SeniorCitizen` | Integer/Binary | Demographic | Whether customer is a senior citizen | `0` (No, 83.79%), `1` (Yes, 16.21%) |
| `Partner` | Categorical | Demographic | Whether customer has a domestic partner | `Yes`, `No` |
| `Dependents` | Categorical | Demographic | Whether customer lives with dependents | `Yes`, `No` |
| `tenure` | Integer | Account | Number of months account has remained active | Range `0` to `72` months (median: `29` months) |
| `PhoneService` | Categorical | Service | Whether customer has telephone service | `Yes`, `No` |
| `MultipleLines` | Categorical | Service | Whether customer has multiple phone lines | `No`, `Yes`, `No phone service` |
| `InternetService` | Categorical | Service | Internet connectivity infrastructure type | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | Categorical | Add-on | Whether online security add-on is active | `Yes`, `No`, `No internet service` |
| `OnlineBackup` | Categorical | Add-on | Whether cloud backup add-on is active | `Yes`, `No`, `No internet service` |
| `DeviceProtection` | Categorical | Add-on | Whether hardware warranty/protection is active | `Yes`, `No`, `No internet service` |
| `TechSupport` | Categorical | Add-on | Whether premium technical support is active | `Yes`, `No`, `No internet service` |
| `StreamingTV` | Categorical | Add-on | Whether streaming television package is active | `Yes`, `No`, `No internet service` |
| `StreamingMovies`| Categorical | Add-on | Whether streaming movies package is active | `Yes`, `No`, `No internet service` |
| `Contract` | Categorical | Account | Contract billing commitment terms | `Month-to-month` (55.0%), `One year` (20.9%), `Two year` (24.1%) |
| `PaperlessBilling` | Categorical | Account | Whether digital paperless invoices are used | `Yes`, `No` |
| `PaymentMethod` | Categorical | Account | Automatic or manual payment channel | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `MonthlyCharges` | Float | Financial | Current recurring monthly bill in USD | Range `$18.25` to `$118.75` (mean: `$64.76`) |
| `TotalCharges` | Float (Raw String) | Financial | Cumulative lifetime charges billed in USD | Range `$18.80` to `$8684.80` (mean: `$2283.30`) |
| `Churn` | Categorical/Binary | **Target** | Whether customer cancelled service within past month | `Yes` (26.54%), `No` (73.46%) |

---

## 4. Data Quality Audit & Handling Rules

1. **Missing Data in `TotalCharges` (11 Records):**
   - **Observation:** Exactly 11 rows contain blank whitespace strings `""` instead of numerical floats.
   - **Root Cause Audit:** Every single one of these 11 accounts has `tenure == 0`, representing brand-new subscribers who joined recently and have not yet accrued a monthly billing cycle. All 11 records have `Churn == 'No'`.
   - **Recommended Handling:** Impute `TotalCharges = 0.0` rather than dropping the records, preserving new account representations.
2. **Duplicate Identifiers:**
   - **Result:** Exactly 0 duplicates in `customerID`.
3. **Out-of-Bounds Values:**
   - Zero negative tenure values or negative charges detected.
4. **Domain Consistency:**
   - Perfectly consistent relations between `PhoneService == 'No'` and `MultipleLines == 'No phone service'`, as well as `InternetService == 'No'` and sub-services.

---

## 5. Statistical Investigation Summary

### Hypothesis 1: Contract Structure vs. Churn
- **Null Hypothesis ($H_0$):** Customer churn is independent of contract commitment type.
- **Result:** $\chi^2 = 1184.60, \text{df}=2, p = 5.86 \times 10^{-258}$. Cramér's $V = 0.410$ (large effect size).
- **Finding:** We reject $H_0$. Month-to-month contract customers churn at **42.71%**, compared to **11.27%** for one-year contracts and **2.83%** for two-year contracts.

### Hypothesis 2: Monthly Financial Burden vs. Churn
- **Null Hypothesis ($H_0$):** There is no difference in mean monthly charges between churned and retained customers.
- **Result:** Welch's $t = 18.41, p = 8.59 \times 10^{-73}$; Mann-Whitney $U = 3,311,628, p = 3.31 \times 10^{-54}$. Cohen's $d = 0.446$ (moderate effect size).
- **Finding:** We reject $H_0$. Churned customers paid an average of **$74.44/month** vs. **$61.27/month** for retained customers (mean difference of **+$13.18/month**, 95% CI: [+$11.77, +$14.58]).

---

## 6. Intended Use, Limitations & Ethical Guardrails

- **Intended Use:** Serves as the primary tabular dataset for building, evaluating, and serving the PredictIQ predictive intelligence pipeline.
- **Cross-Sectional Limitation:** The dataset reflects a single historical snapshot. It lacks event-level timestamped telemetry (e.g., ticket creation events, network latency spikes, billing disputes).
- **Confounding Caution:** Fiber Optic subscribers exhibit the highest churn (41.89%) and the highest charges ($91.50). Correlation does not establish that high prices cause churn directly; service quality and competitive fiber alternatives are unmeasured confounders.
- **Fairness & Governance:** Gender has no predictive association with churn ($26.92\%$ Female vs $26.16\%$ Male). Senior Citizen status ($41.68\%$ churn) must be monitored in downstream modeling to ensure automated retention interventions do not violate age discrimination regulations.
