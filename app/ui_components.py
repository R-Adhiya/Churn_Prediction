"""
app/ui_components.py

UI Component renderers for ChurnGuard AI Streamlit Dashboard.
Renders metric cards, Plotly SHAP feature charts, retention recommendation cards,
customer profile summaries, and model evaluation benchmarks.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any


def render_header():
    """Renders application header banner."""
    st.markdown(
        """
        <div style="text-align: center; padding-bottom: 1rem;">
            <h1 style="color: #0f172a; margin-bottom: 0.2rem; font-weight: 800;">ChurnGuard AI</h1>
            <p style="color: #475569; font-size: 1.1rem; font-weight: 500;">
                Explainable Customer Churn Prediction & Retention Analytics Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics_row(res: Dict[str, Any]):
    """Renders 5 top-level KPI metric cards for prediction output."""
    prob_pct = res["churn_probability"] * 100
    risk_level = res["risk_level"]
    prediction = res["prediction"]
    pred_text = "Likely to Churn" if prediction == 1 else "Unlikely to Churn"
    threshold_pct = int(res["official_threshold"] * 100)

    # Risk level color selector
    val_class = "metric-value-high" if risk_level == "High" else ("metric-value-med" if risk_level == "Medium" else "metric-value-low")
    badge_class = "badge-high" if risk_level == "High" else ("badge-med" if risk_level == "Medium" else "badge-low")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Customer ID</div>
                <div class="metric-value-neutral" style="font-size: 1.3rem;">{res['customer_id']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Churn Probability</div>
                <div class="{val_class}">{prob_pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Risk Level</div>
                <div style="margin-top: 0.4rem;"><span class="{badge_class}">{risk_level.upper()}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Prediction</div>
                <div class="{val_class}" style="font-size: 1.2rem; padding-top: 0.3rem;">{pred_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Locked Threshold</div>
                <div class="metric-value-neutral">{threshold_pct}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_shap_chart(res: Dict[str, Any]):
    """Renders interactive Plotly horizontal bar chart of top SHAP risk drivers and protective factors."""
    st.markdown('<div class="section-header">Why is this customer at risk? (SHAP Explainability)</div>', unsafe_allow_html=True)

    pos_factors = res.get("top_risk_factors", [])
    neg_factors = res.get("protective_factors", [])

    # Combine top 5 positive and top 5 negative drivers
    all_factors = pos_factors[:5] + neg_factors[:5]
    if not all_factors:
        st.info("No detailed SHAP factors available.")
        return

    df_shap = pd.DataFrame(all_factors).sort_values(by="shap_value", ascending=True)

    colors = ["#dc3545" if v > 0 else "#198754" for v in df_shap["shap_value"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df_shap["feature"],
            x=df_shap["shap_value"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#000000", width=0.8)),
            text=[f"{v:+.4f}" for v in df_shap["shap_value"]],
            textposition="outside",
            hoverinfo="y+x",
        )
    )

    fig.update_layout(
        title="Top Factors Influencing Churn Prediction (SHAP Values)",
        title_font=dict(size=14, family="Arial", color="#0f172a"),
        xaxis_title="SHAP Value (Contribution to Log-Odds Margin)",
        yaxis_title="Feature Name",
        height=380,
        margin=dict(l=20, r=40, t=40, b=20),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=True, zerolinecolor="#64748b", zerolinewidth=1.5),
        yaxis=dict(showgrid=False),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<p class="non-causal-note">Note: SHAP values represent statistical feature contributions to model decision margin. Positive values indicate features associated with higher predicted churn risk, while negative values indicate features associated with lower predicted churn risk (non-causal).</p>',
        unsafe_allow_html=True,
    )


def render_retention_recommendations(res: Dict[str, Any]):
    """Renders rule-based business retention suggestions."""
    st.markdown('<div class="section-header">Recommended Retention Actions</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; font-size: 0.9rem;">AI-assisted retention suggestions based on customer profile and key risk drivers:</p>', unsafe_allow_html=True)

    recs = res.get("retention_recommendations", [])
    if not recs:
        st.info("No specific retention actions required.")
        return

    for rec in recs:
        st.markdown(f'<div class="recommendation-card">🎯 <b>Action:</b> {rec}</div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="non-causal-note">Disclaimer: Retention recommendations are rule-based business suggestions designed to support, not replace, customer success workflows.</p>',
        unsafe_allow_html=True,
    )


def render_customer_summary(raw_input: Dict[str, Any]):
    """Renders a compact summary card of the input customer profile."""
    st.markdown('<div class="section-header">Customer Profile Summary</div>', unsafe_allow_html=True)

    # Compute active services count
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    active_count = sum(1 for c in service_cols if raw_input.get(c, "No") == "Yes")

    tenure = raw_input.get("tenure", 0)
    contract = raw_input.get("Contract", "Month-to-month")
    internet = raw_input.get("InternetService", "DSL")
    monthly = raw_input.get("MonthlyCharges", 0.0)
    total = raw_input.get("TotalCharges", 0.0)
    payment = raw_input.get("PaymentMethod", "Electronic check")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"• **Tenure**: `{tenure}` months ({tenure/12.0:.1f} years)")
        st.write(f"• **Contract**: `{contract}`")
        st.write(f"• **Internet Service**: `{internet}`")
        st.write(f"• **Active Services**: `{active_count} / 8` add-ons")

    with col2:
        st.write(f"• **Monthly Charges**: `${monthly:.2f}`")
        st.write(f"• **Total Charges**: `${float(total):.2f}`")
        st.write(f"• **Payment Method**: `{payment}`")
        st.write(f"• **Paperless Billing**: `{raw_input.get('PaperlessBilling', 'No')}`")


def render_model_info_expander():
    """Renders expandable section displaying benchmark test metrics and model specs."""
    with st.expander("ℹ️ Model Architecture & Benchmark Evaluation Metrics (Untouched Test Set)"):
        st.markdown(
            """
            ### Final Model Specifications
            - **Algorithm**: Tuned XGBoost Classifier (`xgboost.XGBClassifier`)
            - **Feature Count**: `43` encoded features (Phase 2 schema)
            - **Locked Classification Threshold**: **`0.30`** (established on validation data)
            - **Explainability**: `SHAP TreeExplainer`
            - **Evaluation Dataset**: Untouched 15% Test Set (`1,057` records, zero data leakage)

            ### Benchmark Evaluation Results (15% Test Set @ t = 0.30)
            | Performance Metric | Measured Value | Category |
            | :--- | :---: | :--- |
            | **Test ROC-AUC** | **`0.8408`** | Threshold-Independent |
            | **Test PR-AUC** | **`0.6639`** | Threshold-Independent |
            | **Test Accuracy** | **`0.7682`** | Threshold-Dependent ($t=0.30$) |
            | **Test Precision** | **`0.5476`** | Threshold-Dependent ($t=0.30$) |
            | **Test Recall** | **`0.7367`** | Threshold-Dependent ($t=0.30$) |
            | **Test F1-Score** | **`0.6282`** | Threshold-Dependent ($t=0.30$) |
            """
        )
