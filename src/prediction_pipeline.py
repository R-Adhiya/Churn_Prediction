"""
src/prediction_pipeline.py

Phase 8 Reusable Churn Risk Prediction Pipeline for ChurnGuard AI.
Accepts raw customer input (dictionary or DataFrame), applies pre-fitted Phase 2 transformations,
validates feature matrix integrity, executes XGBoost model inference, calculates SHAP explanations,
and outputs risk classifications and business retention recommendations.

Data Rules & Constraints:
- Preprocessor (models/preprocessor.joblib) is pre-fitted — NEVER refit during inference.
- Final Model (models/final/churnguard_xgboost_final.joblib) is pre-trained — NEVER retrain.
- Locked Classification Threshold = 0.30 (prediction = 1 if prob >= 0.30 else 0).
- Presentation Risk Levels: Low (< 0.20), Medium (0.20 <= prob < 0.50), High (>= 0.50).
- Non-causal SHAP phrasing enforced ("associated with higher predicted churn risk").
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Union, Dict, List, Any, Tuple


class ChurnPredictionPipeline:
    """Production-style reusable churn prediction and explanation pipeline."""

    def __init__(
        self,
        model_path: str = "models/final/churnguard_xgboost_final.joblib",
        preprocessor_path: str = "models/preprocessor.joblib",
        feature_names_path: str = "data/processed/feature_names.json",
        threshold: float = 0.30,
    ):
        self.threshold = threshold
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.feature_names_path = feature_names_path

        # Validate file existence
        for path_val, name in [
            (self.model_path, "Final Model Artifact"),
            (self.preprocessor_path, "Preprocessor Artifact"),
            (self.feature_names_path, "Feature Names Schema"),
        ]:
            if not os.path.exists(path_val):
                raise FileNotFoundError(f"{name} not found at {path_val}")

        # Load pre-trained artifacts
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        with open(self.feature_names_path, "r", encoding="utf-8") as f:
            self.feature_names = json.load(f)

        # Initialize SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

        # Preprocessing Column Groups (identical to Phase 2)
        self.service_cols = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        self.binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]

    def _preprocess_raw(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Applies exact Phase 2 cleaning, feature engineering, and preprocessor transformation."""
        df = raw_df.copy()

        # 1. Clean TotalCharges (handle blank space strings or missing values)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = (
                pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", "0.0"), errors="coerce")
                .fillna(0.0)
            )
        else:
            df["TotalCharges"] = 0.0

        # Ensure tenure & MonthlyCharges exist and are numeric
        df["tenure"] = pd.to_numeric(df.get("tenure", 0), errors="coerce").fillna(0)
        df["MonthlyCharges"] = pd.to_numeric(df.get("MonthlyCharges", 0.0), errors="coerce").fillna(0.0)

        # 2. Feature Engineering (Exact Phase 2 Formulas)
        df["TenureYears"] = df["tenure"] / 12.0
        df["AverageMonthlySpend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], 0.0)

        # Count active add-on services ('Yes')
        active_services_mask = pd.DataFrame(index=df.index)
        for s_col in self.service_cols:
            active_services_mask[s_col] = (df.get(s_col, "No") == "Yes")
        df["TotalServicesCount"] = active_services_mask.sum(axis=1)

        # 3. Drop customerID and Churn target if present
        X_raw = df.drop(columns=[c for c in ["customerID", "Churn"] if c in df.columns])

        # 4. Binary Encoding Mappings
        X_raw["gender"] = X_raw.get("gender", "Female").map({"Female": 0, "Male": 1}).fillna(0).astype(int)

        for b_col in self.binary_cols:
            X_raw[b_col] = X_raw.get(b_col, "No").map({"No": 0, "Yes": 1}).fillna(0).astype(int)

        # SeniorCitizen handling
        X_raw["SeniorCitizen"] = pd.to_numeric(X_raw.get("SeniorCitizen", 0), errors="coerce").fillna(0).astype(int)

        # 5. Transform via Pre-Fitted ColumnTransformer
        processed_arr = self.preprocessor.transform(X_raw)
        processed_df = pd.DataFrame(processed_arr, columns=self.feature_names)

        # 6. Feature Integrity Validation
        self.validate_feature_matrix(processed_df)

        return processed_df

    def validate_feature_matrix(self, df_proc: pd.DataFrame) -> bool:
        """Validates that transformed feature matrix matches the required schema exactly."""
        if df_proc.shape[1] != 43:
            raise ValueError(f"Feature count mismatch: Expected 43, got {df_proc.shape[1]}")

        if list(df_proc.columns) != self.feature_names:
            raise ValueError("Feature names or column order do not match feature_names.json!")

        if df_proc.isnull().sum().sum() > 0:
            raise ValueError("NaN values detected in transformed feature matrix!")

        if np.isinf(df_proc.values).sum() > 0:
            raise ValueError("Infinite values detected in transformed feature matrix!")

        return True

    def _assign_risk_level(self, probability: float) -> str:
        """Assigns presentation risk category based on predicted probability."""
        if probability < 0.20:
            return "Low"
        elif probability < 0.50:
            return "Medium"
        else:
            return "High"

    def _generate_retention_recommendations(self, raw_input: Dict[str, Any], top_risk_features: List[str]) -> List[str]:
        """Generates targeted business retention recommendations based on customer attributes and SHAP drivers."""
        recs = []

        contract = str(raw_input.get("Contract", ""))
        pm = str(raw_input.get("PaymentMethod", ""))
        security = str(raw_input.get("OnlineSecurity", ""))
        tech = str(raw_input.get("TechSupport", ""))
        internet = str(raw_input.get("InternetService", ""))
        tenure = float(raw_input.get("tenure", 0))
        monthly = float(raw_input.get("MonthlyCharges", 0.0))

        # Contract Recommendation
        if contract == "Month-to-month" or "Contract_Month-to-month" in top_risk_features:
            recs.append("Offer an annual or 2-year contract incentive (15-20% bill discount) to transition from month-to-month billing.")

        # Payment Method Recommendation
        if pm == "Electronic check" or "PaymentMethod_Electronic check" in top_risk_features:
            recs.append("Promote automated payment enrollment (Credit Card / Auto-Bank Transfer) with a recurring $5 monthly bill credit.")

        # Online Security Add-on
        if security == "No" or "OnlineSecurity_No" in top_risk_features:
            recs.append("Offer a complimentary 6-month cybersecurity & online security add-on package to increase service stickiness.")

        # Tech Support Package
        if tech == "No" or "TechSupport_No" in top_risk_features:
            recs.append("Provide a promotional priority 24/7 technical support bundle to improve service experience.")

        # Fiber Optic High-Spend Optimization
        if (internet == "Fiber optic" or "InternetService_Fiber optic" in top_risk_features) and monthly > 70.0:
            recs.append("Review high-cost fiber optic tier and offer a customized loyalty bundle discount or speed upgrade credit.")

        # Early Tenure Onboarding
        if tenure <= 6 or "tenure" in top_risk_features:
            recs.append("Trigger an early-tenure customer success check-in and dedicated onboarding review.")

        # Default recommendation if none triggered
        if not recs:
            recs.append("Conduct a routine customer satisfaction check-in and review usage plan tier.")

        return recs

    def predict_single(self, customer_input: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts churn probability, risk level, SHAP drivers, and retention recommendations for one customer."""
        customer_id = str(customer_input.get("customerID", "Sample_Customer"))
        df_raw = pd.DataFrame([customer_input])

        # Preprocess & validate
        X_proc = self._preprocess_raw(df_raw)

        # Model Inference
        prob = float(self.model.predict_proba(X_proc)[0, 1])
        prediction = int(prob >= self.threshold)
        risk_level = self._assign_risk_level(prob)
        risk_classification = "Higher predicted churn risk" if prob >= self.threshold else "Lower predicted churn risk"

        # SHAP Explanation for single row
        shap_values_row = self.explainer(X_proc)[0]
        shap_vals = shap_values_row.values
        feat_vals = X_proc.iloc[0].values

        df_shap = pd.DataFrame({
            "feature": self.feature_names,
            "feature_value": feat_vals,
            "shap_value": shap_vals,
        }).sort_values(by="shap_value", ascending=False)

        top_pos = df_shap[df_shap["shap_value"] > 0].head(5).to_dict(orient="records")
        top_neg = df_shap[df_shap["shap_value"] < 0].sort_values(by="shap_value", ascending=True).head(5).to_dict(orient="records")

        # Format SHAP features cleanly
        formatted_pos = [
            {
                "feature": r["feature"],
                "feature_value": round(float(r["feature_value"]), 4),
                "shap_value": round(float(r["shap_value"]), 4),
                "interpretation": f"'{r['feature']}' is associated with higher predicted churn risk in the model (SHAP +{r['shap_value']:.4f}).",
            }
            for r in top_pos
        ]

        formatted_neg = [
            {
                "feature": r["feature"],
                "feature_value": round(float(r["feature_value"]), 4),
                "shap_value": round(float(r["shap_value"]), 4),
                "interpretation": f"'{r['feature']}' is associated with lower predicted churn risk in the model (SHAP {r['shap_value']:.4f}).",
            }
            for r in top_neg
        ]

        # Extract top risk feature names for recommendation triggers
        top_risk_names = [r["feature"] for r in top_pos]
        recommendations = self._generate_retention_recommendations(customer_input, top_risk_names)

        # Build Human-Readable Explanation String
        explanation_str = f"Customer ID: {customer_id}\n"
        explanation_str += f"Predicted Churn Probability: {prob * 100:.1f}%\n"
        explanation_str += f"Official Decision Threshold: {self.threshold:.2f}\n"
        explanation_str += f"Prediction: Class {prediction} ({risk_classification})\n"
        explanation_str += f"Presentation Risk Category: {risk_level}\n\n"

        explanation_str += "Top factors increasing predicted churn risk:\n"
        for item in formatted_pos[:3]:
            explanation_str += f"  • {item['feature']} (val = {item['feature_value']}): SHAP contribution +{item['shap_value']:.4f}\n"

        explanation_str += "\nTop factors reducing predicted churn risk:\n"
        for item in formatted_neg[:3]:
            explanation_str += f"  • {item['feature']} (val = {item['feature_value']}): SHAP contribution {item['shap_value']:.4f}\n"

        return {
            "customer_id": customer_id,
            "churn_probability": round(prob, 4),
            "prediction": prediction,
            "official_threshold": self.threshold,
            "risk_level": risk_level,
            "risk_classification": risk_classification,
            "top_risk_factors": formatted_pos,
            "protective_factors": formatted_neg,
            "human_readable_explanation": explanation_str,
            "retention_recommendations": recommendations,
        }

    def predict_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Predicts churn probabilities and risk levels for a DataFrame of multiple customers."""
        df_raw = df_input.copy()
        customer_ids = df_raw.get("customerID", pd.Series([f"Customer_{i+1}" for i in range(len(df_raw))]))

        # Preprocess & validate
        X_proc = self._preprocess_raw(df_raw)

        # Model Inference
        probs = self.model.predict_proba(X_proc)[:, 1]
        preds = (probs >= self.threshold).astype(int)
        risk_levels = [self._assign_risk_level(p) for p in probs]

        # Calculate SHAP for batch to find strongest risk factor per customer
        shap_obj = self.explainer(X_proc)
        shap_vals = shap_obj.values

        strongest_risk_factors = []
        for i in range(len(df_raw)):
            row_shap = shap_vals[i]
            max_idx = int(np.argmax(row_shap))
            if row_shap[max_idx] > 0:
                strongest_risk_factors.append(self.feature_names[max_idx])
            else:
                strongest_risk_factors.append("None (Low Risk)")

        df_out = pd.DataFrame({
            "customerID": customer_ids.values,
            "churn_probability": np.round(probs, 4),
            "prediction": preds,
            "risk_level": risk_levels,
            "strongest_risk_factor": strongest_risk_factors,
        })

        return df_out
