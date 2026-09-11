# Phase 9: Streamlit ChurnGuard AI Application Report

## Executive Summary
Phase 9 completes the user-facing web deployment layer for **ChurnGuard AI** by creating an interactive, portfolio-ready **Streamlit Web Application** (`app.py`). The application integrates seamlessly with the reusable Phase 8 inference engine (**`ChurnPredictionPipeline`**), providing real-time customer churn prediction, interactive **SHAP risk explanations**, and rule-based business retention recommendations.

> [!IMPORTANT]
> **NO RETRAINING & PREPROCESSOR INTEGRITY DIRECTIVES**
> 
> - **Zero Model Retraining**: Loads `models/final/churnguard_xgboost_final.joblib` directly.
> - **Zero Preprocessor Refitting**: Loads `models/preprocessor.joblib` directly; no preprocessing logic is duplicated inside the Streamlit application.
> - **Locked Decision Threshold**: Classification threshold remains strictly locked at **`0.30`** (`prediction = 1` if `churn_probability >= 0.30`).

---

## 1. Application Architecture & File Structure

```text
c:\Users\akhil\Desktop\Churn prediction\
├── app.py                          # Main Streamlit Dashboard Entry Point
├── app/
│   ├── __init__.py                 # Package marker
│   ├── styles.py                   # Custom CSS theme, card styling & badges
│   └── ui_components.py            # Plotly SHAP chart, metric cards & recommendation renderers
├── src/
│   ├── prediction_pipeline.py      # Reusable Phase 8 ChurnPredictionPipeline Engine
│   └── phase_8_prediction_pipeline.py # Phase 8 Pipeline test suite
├── models/
│   ├── preprocessor.joblib         # Pre-fitted ColumnTransformer (Phase 2)
│   └── final/
│       └── churnguard_xgboost_final.joblib # Pre-trained final XGBoost model (Phase 7)
├── data/processed/
│   └── feature_names.json          # 43-feature schema definition
└── requirements.txt                # Deployment dependencies
```

---

## 2. Dashboard UI Layout & Key Sections

The dashboard is structured into 6 primary operational sections:

1. **Header Banner**: Project title (**ChurnGuard AI**) and platform subtitle.
2. **Sidebar Controls & Demo Presets**:
   - **Interactive Demonstration Presets**: High Risk (`1069-XAIEM`), Medium Risk (`7733-UDMTP`), Low Risk (`5787-KXGIY`), and Custom Input Form.
   - **Form Widgets**: Full 19 original customer input fields grouped under *Customer Profile*, *Services*, and *Billing & Charges*.
   - **About Platform**: High-level architectural summary of ChurnGuard AI.
3. **KPI Metrics Cards Row**: Displays Customer ID, Churn Probability %, Risk Category Badge (LOW / MEDIUM / HIGH), Prediction Status ("Likely to Churn" / "Unlikely to Churn"), and Locked Classification Threshold (`30%`).
4. **Customer Profile & Model Benchmark Summary**: 2-column layout showing active service add-ons count, contract type, tenure, monthly spend, and expandable benchmark evaluation results from the untouched 15% Test Set.
5. **SHAP Explainability Section ("Why is this customer at risk?")**: Interactive horizontal Plotly bar chart displaying top positive risk factors (pushing toward churn) and negative protective factors (pushing away from churn) with non-causal statistical interpretations.
6. **Retention Strategy Section ("Recommended Retention Actions")**: Actionable rule-based business suggestions (contract transition offers, automated payment discounts, cybersecurity add-ons, early-tenure success check-ins).

---

## 3. Presentation Risk Tiering vs. Classification Threshold

| Risk Category | Probability Range | Operational Definition | Classification Status ($t=0.30$) |
| :---: | :---: | :--- | :---: |
| **Low** | $P < 0.20$ | Standard retention monitoring; low intervention priority. | Unlikely to Churn (`Class 0`) |
| **Medium** | $0.20 \le P < 0.50$ | Borderline / elevated risk; targeted promotional outreach. | Churn (`Class 1`) if $P \ge 0.30$ |
| **High** | $P \ge 0.50$ | High churn probability; immediate high-value retention offer. | Likely to Churn (`Class 1`) |

> [!NOTE]
> Risk level is presentation-only. Binary classification relies strictly on the official decision threshold ($t = 0.30$).

---

## 4. Verification & Interactive Demonstration Results

The application was tested across representative customer profiles:

### 1. High Risk Customer (`1069-XAIEM`)
- **Churn Probability**: **`91.4%`**
- **Prediction ($t=0.30$)**: `Likely to Churn` (`Class 1`)
- **Risk Category**: `HIGH`
- **Top SHAP Risk Factors**: `tenure` (val = `-1.277`), `Contract_Month-to-month` (val = `1.0`), `OnlineSecurity_No` (val = `1.0`).
- **Retention Actions Triggered**: Contract transition discount, automated payment credit, cybersecurity add-on package, early-tenure onboarding check-in.

### 2. Medium Risk Customer (`7733-UDMTP`)
- **Churn Probability**: **`30.0%`**
- **Prediction ($t=0.30$)**: `Likely to Churn` (`Class 1`)
- **Risk Category**: `MEDIUM`
- **Top SHAP Risk Factors**: `Contract_Month-to-month` (val = `1.0`), `PaymentMethod_Electronic check` (val = `1.0`), `OnlineSecurity_No` (val = `1.0`).
- **Retention Actions Triggered**: Contract transition offer, auto-payment $5 credit, 6-month security bundle.

### 3. Low Risk Customer (`5787-KXGIY`)
- **Churn Probability**: **`0.5%`**
- **Prediction ($t=0.30$)**: `Unlikely to Churn` (`Class 0`)
- **Risk Category**: `LOW`
- **Top Protective Factors**: `Contract_Month-to-month` (val = `0.0`), `tenure` (val = `1.604`), `MonthlyCharges` (val = `-1.5096`).
- **Retention Actions Triggered**: Routine customer satisfaction check-in.

---

## 5. Artifacts Created / Modified

- **Main Streamlit Entry Point**: `app.py`
- **Package Init**: `app/__init__.py`
- **CSS Styling Module**: `app/styles.py`
- **UI Renderer Components**: `app/ui_components.py`
- **Dependencies Spec**: `requirements.txt`
- **Markdown Report**: `reports/phase_9_streamlit_app.md`

---

## 6. How to Run the Application

Execute from the project root:

```bash
streamlit run app.py
```
