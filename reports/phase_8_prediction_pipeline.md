# Phase 8: Reusable Churn Risk Prediction Pipeline Report

## Executive Summary
Phase 8 develops and validates a production-ready, reusable inference pipeline (**`ChurnPredictionPipeline`**) for **ChurnGuard AI**. The pipeline accepts raw customer-level inputs (dictionaries or pandas DataFrames), executes exact Phase 2 data cleaning and feature engineering, transforms inputs using the pre-fitted **`models/preprocessor.joblib`**, performs schema and feature matrix validation, executes model inference via **`models/final/churnguard_xgboost_final.joblib`**, calculates **SHAP-based risk explanations**, and outputs actionable business retention recommendations.

> [!IMPORTANT]
> **NO RETRAINING & LOCKED THRESHOLD DIRECTIVE**
> 
> - **Pre-Fitted Preprocessor**: Preprocessing artifacts were loaded directly; zero transformations were fitted during inference.
> - **Pre-Tuned Final Model**: Model artifacts were loaded directly; zero retraining occurred.
> - **Locked Decision Threshold**: Classification threshold remains strictly locked at **`0.30`** (`prediction = 1` if `churn_probability >= 0.30`).

---

## 1. Pipeline Architecture & Modular Flow

```text
Raw Customer Input (Dict / DataFrame)
               │
               ▼
   TotalCharges String/Blank Cleaning
               │
               ▼
    Feature Engineering (TenureYears, AverageMonthlySpend, TotalServicesCount)
               │
               ▼
   Binary Encoding & Gender/SeniorCitizen Mapping
               │
               ▼
   ColumnTransformer Preprocessor (models/preprocessor.joblib)
               │
               ▼
  Feature Matrix Schema & Integrity Validation (43 features, No NaN/Inf)
               │
               ▼
   XGBoost Model Inference (models/final/churnguard_xgboost_final.joblib)
               │
               ▼
  Probability & Locked Threshold Classification (t = 0.30)
               │
               ▼
  SHAP Local Explanation (TreeExplainer — Non-Causal Risk Drivers)
               │
               ▼
  Targeted Business Retention Recommendations & Presentation Risk Tiering
```

---

## 2. Risk Level Presentation Categories

While binary prediction uses the locked threshold ($t = 0.30$), presentation risk categorization is defined as:

| Risk Category | Probability Range | Operational Definition |
| :---: | :---: | :--- |
| **Low** | $P < 0.20$ | Standard retention monitoring; low intervention priority. |
| **Medium** | $0.20 \le P < 0.50$ | Borderline / elevated risk; targeted promotional outreach. |
| **High** | $P \ge 0.50$ | High churn probability; immediate high-value retention offer. |

---

## 3. Representative Inference Test Results

The pipeline was evaluated on representative raw customer records extracted from the Validation Set:

### 1069-XAIEM (Higher predicted churn risk)
- **Predicted Churn Probability**: **`91.3%`**
- **Binary Prediction ($t=0.30$)**: `1` (Churn)
- **Presentation Risk Level**: **`High`**

#### Top Risk Factors (SHAP):
  • `tenure` (val = `-1.277`): SHAP contribution `+0.8487`
  • `Contract_Month-to-month` (val = `1.0`): SHAP contribution `+0.5799`
  • `OnlineSecurity_No` (val = `1.0`): SHAP contribution `+0.2348`

#### Retention Recommendations:
  • Offer an annual or 2-year contract incentive (15-20% bill discount) to transition from month-to-month billing.
  • Promote automated payment enrollment (Credit Card / Auto-Bank Transfer) with a recurring $5 monthly bill credit.
  • Offer a complimentary 6-month cybersecurity & online security add-on package to increase service stickiness.
  • Provide a promotional priority 24/7 technical support bundle to improve service experience.
  • Review high-cost fiber optic tier and offer a customized loyalty bundle discount or speed upgrade credit.
  • Trigger an early-tenure customer success check-in and dedicated onboarding review.

---
### 7733-UDMTP (Higher predicted churn risk)
- **Predicted Churn Probability**: **`30.0%`**
- **Binary Prediction ($t=0.30$)**: `1` (Churn)
- **Presentation Risk Level**: **`Medium`**

#### Top Risk Factors (SHAP):
  • `Contract_Month-to-month` (val = `1.0`): SHAP contribution `+0.3564`
  • `PaymentMethod_Electronic check` (val = `1.0`): SHAP contribution `+0.2104`
  • `OnlineSecurity_No` (val = `1.0`): SHAP contribution `+0.1727`

#### Retention Recommendations:
  • Offer an annual or 2-year contract incentive (15-20% bill discount) to transition from month-to-month billing.
  • Promote automated payment enrollment (Credit Card / Auto-Bank Transfer) with a recurring $5 monthly bill credit.
  • Offer a complimentary 6-month cybersecurity & online security add-on package to increase service stickiness.
  • Provide a promotional priority 24/7 technical support bundle to improve service experience.

---
### 5787-KXGIY (Lower predicted churn risk)
- **Predicted Churn Probability**: **`0.5%`**
- **Binary Prediction ($t=0.30$)**: `0` (No Churn)
- **Presentation Risk Level**: **`Low`**

#### Top Risk Factors (SHAP):
  • `gender` (val = `1.0`): SHAP contribution `+0.0428`
  • `Dependents` (val = `0.0`): SHAP contribution `+0.0176`
  • `MultipleLines_Yes` (val = `0.0`): SHAP contribution `+0.0103`

#### Retention Recommendations:
  • Conduct a routine customer satisfaction check-in and review usage plan tier.

---

## 4. Pipeline Consistency Test Results

To verify deployment integrity, predictions from the raw-input inference pipeline were compared directly against predictions generated from the saved Phase 2 processed feature matrix (`X_val.csv`):

- **Max Absolute Probability Difference**: **`0.000050`**
- **Consistency Test Status**: **PASS** (Difference < `1e-4`)

---

## 5. Safety & Integrity Validation Summary

| Safety Check | Status | Description |
| :--- | :---: | :--- |
| **Model Loaded** | PASS | Final XGBoost model artifact loaded successfully. |
| **Preprocessor Loaded** | PASS | Phase 2 ColumnTransformer preprocessor loaded. |
| **Feature Schema Aligned** | PASS | Transformed feature columns match `feature_names.json`. |
| **43 Features Produced** | PASS | Exact 43 model features generated per record. |
| **Zero NaN / Inf** | PASS | Preprocessing produces zero NaN or Inf entries. |
| **Probability Range [0, 1]** | PASS | Predicted probabilities bounded validly. |
| **Locked Threshold (0.30)** | PASS | Threshold maintained strictly without modification. |
| **SHAP Explanations Active** | PASS | Local SHAP drivers generated dynamically. |
| **Deployment Consistency** | PASS | Raw pipeline probability matches saved feature matrix identically. |

---

## 6. Artifacts Generated

- **Python Pipeline Module**: `src/prediction_pipeline.py`
- **Execution Test Script**: `src/phase_8_prediction_pipeline.py`
- **Sample Predictions CSV**: `data/predictions/sample_customer_predictions.csv`
- **JSON Metadata**: `reports/phase_8_prediction_pipeline.json`
- **Markdown Report**: `reports/phase_8_prediction_pipeline.md`
