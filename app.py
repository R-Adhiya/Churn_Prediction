"""
app.py

Main Streamlit Application Entry Point for ChurnGuard AI.
Interactive dashboard providing explainable customer churn prediction, SHAP risk driver charts,
and rule-based business retention recommendations using the reusable Phase 8 inference pipeline.

Execution:
streamlit run app.py
"""

import sys
import os
import warnings
import streamlit as st
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from app.styles import load_custom_css
    from app.ui_components import (
        render_header,
        render_metrics_row,
        render_shap_chart,
        render_retention_recommendations,
        render_customer_summary,
        render_model_info_expander,
    )
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "app")))
    from styles import load_custom_css
    from ui_components import (
        render_header,
        render_metrics_row,
        render_shap_chart,
        render_retention_recommendations,
        render_customer_summary,
        render_model_info_expander,
    )

try:
    from src.prediction_pipeline import ChurnPredictionPipeline
except ModuleNotFoundError:
    from prediction_pipeline import ChurnPredictionPipeline



# Set Streamlit Page Configuration
st.set_page_config(
    page_title="ChurnGuard AI — Explainable Customer Churn Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS Styling
st.markdown(load_custom_css(), unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Caches and returns the reusable Phase 8 ChurnPredictionPipeline instance."""
    return ChurnPredictionPipeline()


# Representative Demo Presets (Identified in Phase 6 & Phase 8)
PRESET_HIGH_RISK = {
    "customerID": "1069-XAIEM",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.30,
    "TotalCharges": 75.30,
}

PRESET_MEDIUM_RISK = {
    "customerID": "7733-UDMTP",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 57,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 55.00,
    "TotalCharges": 3135.00,
}

PRESET_LOW_RISK = {
    "customerID": "5787-KXGIY",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 71,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 25.40,
    "TotalCharges": 1803.40,
}


def main():
    render_header()

    # Load Pipeline with error handling
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Failed to load prediction pipeline artifacts: {e}")
        st.stop()

    # --- SIDEBAR: PRESET SELECTOR & INPUT CONTROLS ---
    st.sidebar.markdown("### 🎛️ Demo Presets & Mode")
    preset_choice = st.sidebar.selectbox(
        "Select Demonstration Preset",
        [
            "High Risk Customer (1069-XAIEM)",
            "Medium Risk Customer (7733-UDMTP)",
            "Low Risk Customer (5787-KXGIY)",
            "Custom Input Form",
        ],
        index=0,
    )

    # Set default values based on preset
    if "High Risk" in preset_choice:
        defaults = PRESET_HIGH_RISK
    elif "Medium Risk" in preset_choice:
        defaults = PRESET_MEDIUM_RISK
    elif "Low Risk" in preset_choice:
        defaults = PRESET_LOW_RISK
    else:
        defaults = PRESET_HIGH_RISK

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Customer Profile")
    cid = st.sidebar.text_input("Customer ID", value=defaults["customerID"])
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"], index=0 if defaults["gender"] == "Female" else 1)
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1], index=int(defaults["SeniorCitizen"]))
    partner = st.sidebar.selectbox("Partner", ["No", "Yes"], index=0 if defaults["Partner"] == "No" else 1)
    dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"], index=0 if defaults["Dependents"] == "No" else 1)
    tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=int(defaults["tenure"]))
    contract = st.sidebar.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"],
        index=["Month-to-month", "One year", "Two year"].index(defaults["Contract"]),
    )
    payment = st.sidebar.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(defaults["PaymentMethod"]),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌐 Services")
    phone = st.sidebar.selectbox("Phone Service", ["No", "Yes"], index=0 if defaults["PhoneService"] == "No" else 1)
    lines = st.sidebar.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"],
        index=["No", "Yes", "No phone service"].index(defaults["MultipleLines"]),
    )
    internet = st.sidebar.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
        index=["DSL", "Fiber optic", "No"].index(defaults["InternetService"]),
    )
    sec = st.sidebar.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"],
        index=["No", "Yes", "No internet service"].index(defaults["OnlineSecurity"]),
    )
    backup = st.sidebar.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"],
        index=["No", "Yes", "No internet service"].index(defaults["OnlineBackup"]),
    )
    protection = st.sidebar.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"],
        index=["No", "Yes", "No internet service"].index(defaults["DeviceProtection"]),
    )
    tech = st.sidebar.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"],
        index=["No", "Yes", "No internet service"].index(defaults["TechSupport"]),
    )
    tv = st.sidebar.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"],
        index=["No", "Yes", "No internet service"].index(defaults["StreamingTV"]),
    )
    movies = st.sidebar.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"],
        index=["No", "Yes", "No internet service"].index(defaults["StreamingMovies"]),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💳 Billing & Charges")
    paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"], index=0 if defaults["PaperlessBilling"] == "No" else 1)
    monthly = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=float(defaults["MonthlyCharges"]), step=1.0)
    total = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(defaults["TotalCharges"]), step=10.0)

    # Sidebar About Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About ChurnGuard AI")
    st.sidebar.info(
        "ChurnGuard AI is an explainable customer churn prediction and retention analytics platform. "
        "It uses a tuned XGBoost model regularized on 43 engineered features, evaluated on an untouched test set (ROC-AUC 0.8408), "
        "and explained via SHAP TreeExplainer at a locked 30% decision threshold."
    )

    # Construct Raw Input Dictionary
    raw_input = {
        "customerID": cid,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": lines,
        "InternetService": internet,
        "OnlineSecurity": sec,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    # Action Button
    st.markdown("### 🔍 Customer Churn Risk Analysis")
    predict_btn = st.button("🚀 Predict Churn Risk & Explain", type="primary", use_container_width=True)

    # Auto-run prediction on page load or button click
    if predict_btn or "auto_run" not in st.session_state:
        st.session_state["auto_run"] = True

        with st.spinner("Executing pipeline inference & SHAP tree explanation..."):
            try:
                res = pipeline.predict_single(raw_input)
            except Exception as e:
                st.error(f"Inference error: {e}")
                st.stop()

        # 1. Metric Cards Row
        render_metrics_row(res)

        # 2. Customer Summary Card & Model Expander
        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            render_customer_summary(raw_input)
        with col_right:
            render_model_info_expander()

        # 3. SHAP Explainability Plot
        render_shap_chart(res)

        # 4. Retention Recommendations
        render_retention_recommendations(res)


if __name__ == "__main__":
    main()
