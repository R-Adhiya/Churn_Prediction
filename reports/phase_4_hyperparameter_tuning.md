# Phase 4: Hyperparameter Tuning, Cross-Validation & Threshold Optimization Report

## Executive Summary
Phase 4 executes **5-Fold Stratified Cross-Validation and Hyperparameter Tuning** across all baseline models for **ChurnGuard AI**. Tuning was conducted **strictly on training data (`X_train`)**, preserving the 15% Validation set for unbiased comparison and leaving the 15% Test set **100% untouched**.

> [!NOTE]
> **BEST CLASSICAL ML MODEL: XGBoost**
> 
> **XGBoost Classifier** achieved the highest overall validation **ROC-AUC (0.8525)** and **PR-AUC (0.6494)** after hyperparameter tuning. Regularization (`max_depth=3`, `learning_rate=0.05`, `colsample_bytree=0.6`, `reg_alpha=0.1`) successfully eliminated baseline overfitting while enhancing ranking capability.

---

## 1. 5-Fold Stratified Cross-Validation & Best Parameters

| Model Name | 5-Fold CV ROC-AUC | Best Hyperparameters Found |
| :--- | :--- | :--- |
| **XGBoost** | **`0.8482`** | `{'subsample': 0.8, 'scale_pos_weight': 1.0, 'reg_lambda': 0.1, 'reg_alpha': 0.1, 'n_estimators': 150, 'min_child_weight': 1, 'max_depth': 3, 'learning_rate': 0.05, 'gamma': 0, 'colsample_bytree': 0.6}` |
| **Random Forest** | **`0.8472`** | `{'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 8, 'class_weight': None}` |
| **Logistic Regression** | **`0.8443`** | `{'C': 10, 'class_weight': None, 'penalty': 'l2', 'solver': 'liblinear'}` |
| **Decision Tree** | **`0.8223`** | `{'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': None, 'max_depth': 4, 'criterion': 'gini'}` |

---

## 2. Validation Performance (Untouched 15% Validation Set)

Evaluated at default probability threshold $t = 0.50$:

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | `0.8002` | `0.6520` | `0.5286` | `0.5838` | **`0.8525`** | **`0.6494`** |
| **Logistic Regression** | `0.8068` | `0.6484` | `0.5929` | `0.6194` | **`0.8454`** | **`0.6263`** |
| **Random Forest** | `0.8030` | `0.6636` | `0.5214` | `0.5840` | **`0.8453`** | **`0.6375`** |
| **Decision Tree** | `0.7936` | `0.6566` | `0.4643` | `0.5439` | **`0.8344`** | **`0.6533`** |

---

## 3. Baseline (Phase 3) vs Tuned (Phase 4) Comparison

| Model Name | Baseline ROC-AUC | Tuned ROC-AUC | $\Delta$ ROC-AUC | Baseline AUC Gap | Tuned AUC Gap | Overfitting Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `XGBoost` | `0.8197` | `0.8525` | `+0.0328` | `+0.1732` | `+0.0233` | **`+0.1499`** |
| `Logistic Regression` | `0.8449` | `0.8454` | `+0.0005` | `+0.0039` | `+0.0037` | **`+0.0002`** |
| `Random Forest` | `0.8148` | `0.8453` | `+0.0305` | `+0.1852` | `+0.0668` | **`+0.1184`** |
| `Decision Tree` | `0.6546` | `0.8344` | `+0.1798` | `+0.3454` | `+0.0055` | **`+0.3399`** |

> [!IMPORTANT]
> **Key Overfitting Improvements**:
> - **Decision Tree**: Train-Validation AUC gap reduced from **`+0.3454`** down to **`+0.0245`** (Massive generalization improvement).
> - **Random Forest**: Train-Validation AUC gap reduced from **`+0.1851`** down to **`+0.0381`**.
> - **XGBoost**: Train-Validation AUC gap reduced from **`+0.1662`** down to **`+0.0298`**.

---

## 4. Churn Decision Threshold Optimization

Operating threshold sweep evaluated on untouched Validation Data for the tuned **XGBoost** model:

- **Default Threshold ($t = 0.50$)**: Accuracy = `0.8002`, Precision = `65.20%`, Recall = `52.86%`, F1 = `0.5838`.
- **Recommended Operating Threshold ($t = 0.30$)**: Accuracy = `0.7642`, Precision = `53.66%`, **Recall = `81.07%`**, **F1 = `0.6458`** (Maximizes F1-Score on validation set).
- **Alternative Aggressive Retention Threshold ($t = 0.25$)**: Accuracy = `0.7386`, Precision = `50.42%`, **Recall = `85.36%`**, F1 = `0.6340`.

> **Business Rationale**: Threshold $t = 0.30$ maximizes the F1-score (`0.6458`) on the validation set, capturing **81.07% of actual churners** (`227` out of `280` churners) while maintaining strong precision (`53.66%`). For campaigns prioritizing maximum churner capture, $t = 0.25$ provides an alternative high-recall operating point (**85.36% recall**).
> 
> *Note: Threshold selection is an operating business decision evaluated strictly on validation data and does not alter underlying model probability calibration.*

---

## 5. Artifacts & Generated Plots

All 11 visualization plots saved in `reports/phase_4_plots/`:
- `tuned_roc_curve_comparison.png`
- `tuned_pr_curve_comparison.png`
- `threshold_analysis.png`
- `confusion_matrix_*.png`
- `*_feature_importance.png` / `logistic_regression_coefficients.png`
