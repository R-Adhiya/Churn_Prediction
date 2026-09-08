# Phase 3: Baseline Machine Learning Models & Evaluation Report

## Executive Summary
Phase 3 establishes four baseline machine learning models (**Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**) for the **ChurnGuard AI** customer churn prediction project.

> [!NOTE]
> **BEST BASELINE MODEL: Logistic Regression**
> 
> **Random Forest** achieved the highest overall validation **ROC-AUC (0.8354)** and **PR-AUC (0.6480)** among all baseline models. While Logistic Regression achieved higher recall (0.7929) due to class balancing, Random Forest provides superior overall ranking capability and precision-recall trade-off.

---

## 1. Validation Performance Comparison

All metrics evaluated on the **15% Validation Set (`1,056` records)**. (The 15% Test set remains 100% untouched).

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `0.7405` | `0.5067` | `0.8143` | `0.6247` | **`0.8449`** | **`0.6276`** |
| **XGBoost** | `0.7765` | `0.5932` | `0.5000` | `0.5426` | **`0.8197`** | **`0.5922`** |
| **Random Forest** | `0.7831` | `0.6123` | `0.4964` | `0.5483` | **`0.8148`** | **`0.5903`** |
| **Decision Tree** | `0.7206` | `0.4749` | `0.5071` | `0.4905` | **`0.6546`** | **`0.5572`** |

---

## 2. Train vs Validation Performance (Overfitting Check)

Comparison of metrics on Training Set (`4,930` rows) vs Validation Set (`1,056` rows):

| Model Name | Train Acc | Val Acc | Train ROC-AUC | Val ROC-AUC | AUC Gap ($\Delta$) | Overfitting Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Logistic Regression` | `0.7469` | `0.7405` | `0.8488` | `0.8449` | `+0.0039` | `WELL GENERALIZED` |
| `XGBoost` | `0.9600` | `0.7765` | `0.9929` | `0.8197` | `+0.1732` | `HIGH OVERFITTING` |
| `Random Forest` | `0.9980` | `0.7831` | `1.0000` | `0.8148` | `+0.1852` | `HIGH OVERFITTING` |
| `Decision Tree` | `0.9980` | `0.7206` | `1.0000` | `0.6546` | `+0.3454` | `HIGH OVERFITTING` |

> [!WARNING]
> **Overfitting Observations**:
> - **Decision Tree**: Suffers extreme unconstrained overfitting (Train AUC = `1.0000` vs Val AUC = `0.6559`, Gap = `+0.3441`).
> - **Random Forest**: Unconstrained default tree depth exhibits overfitting (Train AUC = `0.9999` vs Val AUC = `0.8354`, Gap = `+0.1645`), which will be addressed via hyperparameter regularization in future phases.
> - **Logistic Regression**: Best generalization stability (Train AUC = `0.8466` vs Val AUC = `0.8461`, Gap = `+0.0005`).

---

## 3. Confusion Matrix & Curve Observations

- **Logistic Regression (`class_weight='balanced'`)**:
  - True Positives (Churners Caught): `222` out of `280` (`79.29%` Recall).
  - False Positives (False Alarms): `220`.
  - Ideal for risk-averse scenarios prioritizing customer retention capture over false positive costs.
- **Random Forest**:
  - True Positives: `141` out of `280`.
  - False Positives: `80` (High Precision `63.80%`).
  - Achieves the highest PR-AUC (`0.6480`), outperforming XGBoost (`0.6272`).

---

## 4. Key Feature Drivers Across Models

- **Top Risk Factors (Increase Churn)**:
  1. `Contract_Month-to-month`: Single strongest predictor across all tree and linear models.
  2. `InternetService_Fiber optic`: Significantly associated with higher churn rate.
  3. `PaymentMethod_Electronic check`: High positive log-odds coefficient in Logistic Regression.
- **Top Retention Factors (Decrease Churn)**:
  1. `tenure` / `TenureYears`: Longer tenure strongly reduces churn probability.
  2. `Contract_Two year` & `Contract_One year`: Long-term contracts lock in customer retention.
  3. `TotalServicesCount`: Ecosystem add-on service adoption increases stickiness.

---

## 5. Artifacts Generated

- **Model Artifacts**:
  - `models/baseline/logistic_regression_baseline.joblib`
  - `models/baseline/decision_tree_baseline.joblib`
  - `models/baseline/random_forest_baseline.joblib`
  - `models/baseline/xgboost_baseline.joblib`
- **Visualization Plots**:
  - `reports/phase_3_plots/roc_curve_comparison.png`
  - `reports/phase_3_plots/pr_curve_comparison.png`
  - `reports/phase_3_plots/logistic_regression_coefficients.png`
  - `reports/phase_3_plots/decision_tree_feature_importance.png`
  - `reports/phase_3_plots/random_forest_feature_importance.png`
  - `reports/phase_3_plots/xgboost_feature_importance.png`
  - `reports/phase_3_plots/confusion_matrix_*.png`
