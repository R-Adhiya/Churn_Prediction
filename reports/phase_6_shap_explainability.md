# Phase 6: SHAP Explainable AI & Model Interpretability Report

## Executive Summary
Phase 6 implements model interpretability using **SHAP (SHapley Additive exPlanations)** for the selected Phase 4 tuned **XGBoost Classifier**. Global feature importance and directional risk drivers were calculated across the complete **Training Set (`4,930` records)** using `shap.TreeExplainer`. Individual customer risk explanations were evaluated on the **Validation Set (`1,056` records)**.

> [!IMPORTANT]
> **DATA ISOLATION & TEST SET PROTECTION**
> 
> The **15% Test set (`X_test`, `1,057` records) remained 100% UNTOUCHED** throughout Phase 6 SHAP analysis. No test records were loaded into memory or analyzed.

> [!NOTE]
> **NON-CAUSAL PHRASING DIRECTIVE**
> 
> SHAP measures feature contributions to model decision-making log-odds probabilities. All explanations adhere to strict non-causal phrasing (e.g., *"Feature X is associated with higher predicted churn risk in the model"* rather than *"Feature X causes churn"*).

---

## 1. SHAP Configuration & Methodology

- **Model Explained**: XGBoost Classifier (`models/tuned/xgboost_tuned.joblib`)
- **Explainer Method**: `shap.TreeExplainer`
- **Global Explanation Dataset**: Training Set (`X_train.csv`, `4,930` records, `43` features)
- **Individual Explanation Dataset**: Validation Set (`X_val.csv`, `1,056` records)
- **Official Decision Threshold**: `0.30` (established in Phase 4)
- **Target Class Explained**: `Churn = 1` (Positive Churn Class)
- **Additive Verification**: Verified in margin space $\text{base\_value} + \sum \text{SHAP} = \text{log-odds margin}$, with continuous probability $P = \sigma(\text{margin})$.

---

## 2. Top 20 Global SHAP Feature Importance Table

The table below summarizes the top 20 features ranked by mean absolute SHAP value ($\text{mean}(|\text{SHAP}|)$):

| Rank | Feature Name | Mean \|SHAP\| | Mean SHAP | Pos Impact Count ($>0$) | Neg Impact Count ($<0$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `1` | `Contract_Month-to-month` | **`0.5708`** | `-0.0652` | `2704` | `2226` |
| `2` | `tenure` | **`0.4018`** | `-0.1370` | `1647` | `3283` |
| `3` | `OnlineSecurity_No` | **`0.2469`** | `-0.0488` | `2439` | `2491` |
| `4` | `TechSupport_No` | **`0.1985`** | `-0.0336` | `2411` | `2519` |
| `5` | `InternetService_Fiber optic` | **`0.1851`** | `-0.0250` | `2177` | `2753` |
| `6` | `PaymentMethod_Electronic check` | **`0.1550`** | `-0.0334` | `1654` | `3276` |
| `7` | `Contract_Two year` | **`0.1523`** | `-0.0399` | `3738` | `1192` |
| `8` | `PaperlessBilling` | **`0.1512`** | `-0.0230` | `2920` | `2010` |
| `9` | `MonthlyCharges` | **`0.1401`** | `-0.0213` | `1697` | `3233` |
| `10` | `AverageMonthlySpend` | **`0.1261`** | `-0.0245` | `1360` | `3570` |
| `11` | `TenureYears` | **`0.1114`** | `-0.0437` | `1412` | `3518` |
| `12` | `MultipleLines_No` | **`0.0831`** | `-0.0009` | `2583` | `2347` |
| `13` | `OnlineBackup_No` | **`0.0782`** | `-0.0121` | `2142` | `2788` |
| `14` | `TotalCharges` | **`0.0716`** | `-0.0125` | `1794` | `3136` |
| `15` | `StreamingMovies_Yes` | **`0.0613`** | `-0.0119` | `1931` | `2999` |
| `16` | `Dependents` | **`0.0562`** | `-0.0116` | `3463` | `1467` |
| `17` | `Contract_One year` | **`0.0394`** | `-0.0168` | `1037` | `3893` |
| `18` | `SeniorCitizen` | **`0.0384`** | `-0.0055` | `794` | `4136` |
| `19` | `TotalServicesCount` | **`0.0201`** | `-0.0060` | `2805` | `2125` |
| `20` | `StreamingTV_Yes` | **`0.0185`** | `-0.0002` | `1941` | `2989` |

---

## 3. Top 5 Feature Deep-Dive & Interpretations

Based strictly on mean absolute SHAP values across `4,930` training samples, the top 5 features driving the XGBoost churn model are:

1. **`Contract_Month-to-month`** (Mean \|SHAP\| = `0.5708`):
   - **Directional Impact**: Having a month-to-month contract is the single strongest factor associated with higher predicted churn risk in the model.
   - **Distribution**: Pushes model log-odds significantly upward ($>0$) for month-to-month subscribers compared to annual contracts.

2. **`tenure`** (Mean \|SHAP\| = `0.4018`):
   - **Directional Impact**: Shorter customer tenure is associated with higher predicted churn risk.
   - **Distribution**: Higher tenure values (long-standing customers) consistently yield negative SHAP values, pushing predicted churn risk downward.

3. **`OnlineSecurity_No`** (Mean \|SHAP\| = `0.2469`):
   - **Directional Impact**: Absence of online security services is associated with higher predicted churn risk.
   - **Distribution**: Customers without online security protection tend to receive positive SHAP contributions toward churn.

4. **`TechSupport_No`** (Mean \|SHAP\| = `0.1985`):
   - **Directional Impact**: Lack of tech support is associated with elevated predicted churn probability.
   - **Distribution**: Customers without tech support subscriptions show consistently higher SHAP risk values.

5. **`InternetService_Fiber optic`** (Mean \|SHAP\| = `0.1851`):
   - **Directional Impact**: Fiber optic internet service subscription is associated with higher predicted churn risk.
   - **Distribution**: High monthly charges combined with fiber optic service push model risk scores upward.

---

## 4. Individual Customer Case Studies (Validation Set)

Three representative customers were selected from the Validation Set (`X_val`) to demonstrate local model interpretability at the official decision threshold ($t = 0.30$):

### High Risk Customer (Customer ID: `1069-XAIEM`)
- **Validation Row Index**: `174`
- **Predicted Churn Probability**: **`91.3%`**
- **Official Threshold**: `0.30`
- **Risk Classification**: **Higher predicted churn risk**

#### Key Risk Factors (SHAP Breakdown):
```text
Predicted Churn Probability: 91.4%
Official Operating Threshold: 0.30
Risk Classification: Higher predicted churn risk (Class 1)

Top factors increasing predicted churn risk:
  • tenure (val = -1.277): SHAP contribution +0.8487
  • Contract_Month-to-month (val = 1.0): SHAP contribution +0.5799
  • OnlineSecurity_No (val = 1.0): SHAP contribution +0.2348

Top factors reducing predicted churn risk:
  • Contract_One year (val = 0.0): SHAP contribution -0.0193
  • StreamingTV_Yes (val = 0.0): SHAP contribution -0.0143
  • gender (val = 0.0): SHAP contribution -0.0109

```
- **Waterfall Plot Artifact**: `reports/phase_6_plots/shap_waterfall_high_risk.png`

---
### Medium Risk (Borderline) Customer (Customer ID: `7733-UDMTP`)
- **Validation Row Index**: `496`
- **Predicted Churn Probability**: **`30.0%`**
- **Official Threshold**: `0.30`
- **Risk Classification**: **Higher predicted churn risk**

#### Key Risk Factors (SHAP Breakdown):
```text
Predicted Churn Probability: 30.0%
Official Operating Threshold: 0.30
Risk Classification: Higher predicted churn risk (Class 1)

Top factors increasing predicted churn risk:
  • Contract_Month-to-month (val = 1.0): SHAP contribution +0.3564
  • PaymentMethod_Electronic check (val = 1.0): SHAP contribution +0.2104
  • OnlineSecurity_No (val = 1.0): SHAP contribution +0.1727

Top factors reducing predicted churn risk:
  • tenure (val = 0.9953): SHAP contribution -0.6515
  • InternetService_Fiber optic (val = 0.0): SHAP contribution -0.2422
  • TenureYears (val = 0.9953): SHAP contribution -0.1647

```
- **Waterfall Plot Artifact**: `reports/phase_6_plots/shap_waterfall_medium_risk.png`

---
### Low Risk Customer (Customer ID: `5787-KXGIY`)
- **Validation Row Index**: `671`
- **Predicted Churn Probability**: **`0.5%`**
- **Official Threshold**: `0.30`
- **Risk Classification**: **Lower predicted churn risk**

#### Key Risk Factors (SHAP Breakdown):
```text
Predicted Churn Probability: 0.5%
Official Operating Threshold: 0.30
Risk Classification: Lower predicted churn risk (Class 0)

Top factors increasing predicted churn risk:
  • gender (val = 1.0): SHAP contribution +0.0428
  • Dependents (val = 0.0): SHAP contribution +0.0176
  • MultipleLines_Yes (val = 0.0): SHAP contribution +0.0103

Top factors reducing predicted churn risk:
  • Contract_Month-to-month (val = 0.0): SHAP contribution -0.7700
  • tenure (val = 1.604): SHAP contribution -0.6477
  • MonthlyCharges (val = -1.5096): SHAP contribution -0.4574

```
- **Waterfall Plot Artifact**: `reports/phase_6_plots/shap_waterfall_low_risk.png`

---
## 5. Quality & Integrity Validation Checks

| Validation Check | Status | Description |
| :--- | :---: | :--- |
| **No NaN Values** | PASS | SHAP matrices contain zero NaN entries. |
| **No Infinite Values** | PASS | SHAP matrices contain zero Infinite entries. |
| **43-Feature Schema Alignment** | PASS | SHAP dimensions match the exact 43 encoded features. |
| **Feature Name Order** | PASS | Exact feature alignment across training and validation splits. |
| **Training Record Integrity** | PASS | 4,930 training records processed cleanly. |
| **Validation Record Integrity** | PASS | 1,056 validation records processed cleanly. |
| **Additive Margin Relationship** | PASS | Base value + sum(SHAP) = log-odds margin output. |
| **Test Set Isolation** | PASS | 1,057 test records preserved 100% untouched. |

---

## 6. Artifacts Generated

- **Feature Importance CSV**: `reports/phase_6_shap_feature_importance.csv`
- **JSON Metadata**: `reports/phase_6_shap_results.json`
- **Markdown Report**: `reports/phase_6_shap_explainability.md`
- **Plots Saved (`reports/phase_6_plots/`)**:
  - `shap_feature_importance_bar.png`
  - `shap_summary_beeswarm.png`
  - `shap_dependence_<feature>.png` (top 5 features)
  - `shap_waterfall_high_risk.png`
  - `shap_waterfall_medium_risk.png`
  - `shap_waterfall_low_risk.png`
