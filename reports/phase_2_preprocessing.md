# Phase 2: Data Preprocessing & Feature Engineering Report

## Executive Summary
Phase 2 data preprocessing and feature engineering for the **IBM Telco Customer Churn** dataset (`Telco-Customer-Churn.csv`) has been completed. All transformations (imputation, scaling, one-hot encoding) were **fitted strictly on the 70% training set** to prevent data leakage. The target class distribution (~26.54% churn) was preserved across all splits without synthetic resampling.

---

## 1. Dataset Split & Shape Summary

- **Original Dataset**: `7,043` rows $\times$ `21` columns
- **Removed Columns**: `['customerID', 'Churn']`
- **Final Encoded Features**: `43`

| Split | Customer Count | Percentage | Churn Count | Churn Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Train Set** | `4,930` | `70.0%` | `1,307` | `26.53%` |
| **Validation Set** | `1,056` | `15.0%` | `280` | `26.52%` |
| **Test Set** | `1,057` | `15.0%` | `280` | `26.58%` |

---

## 2. Preprocessing & Encoding Details

### A. Missing Value Treatment
- `TotalCharges`: Replaced `11` blank space strings (`' '`) with `0.0` for new accounts with `tenure == 0`. No records were dropped.

### B. Target & Binary Encodings
- **Target (`churn`)**: `No` $\rightarrow 0$, `Yes` $\rightarrow 1$
- **`gender`**: `Female` $\rightarrow 0$, `Male` $\rightarrow 1$
- **`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`**: `No` $\rightarrow 0$, `Yes` $\rightarrow 1$
- **`SeniorCitizen`**: Passthrough (already binary `0`/`1`).

### C. Multi-Category One-Hot Encoding
The following 10 multi-category features were encoded using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` producing 31 encoded binary indicator features:
`MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod`.

### D. Numerical Feature Scaling
`StandardScaler` was fitted on training data for the 6 numerical features:
`tenure`, `MonthlyCharges`, `TotalCharges`, `TenureYears`, `AverageMonthlySpend`, `TotalServicesCount`.

---

## 3. Feature Engineering

| Feature Name | Mathematical Formula | Business Rationale |
| :--- | :--- | :--- |
| `TenureYears` | `tenure / 12.0` | Converts tenure months to years for intuitive baseline interpretation. |
| `AverageMonthlySpend` | `TotalCharges / tenure` (0 if `tenure == 0`) | Captures historical average monthly spending rate per customer. |
| `TotalServicesCount` | Sum of `Yes` across 8 add-on service columns | Quantifies ecosystem stickiness and service portfolio adoption. |

---

## 4. Data Leakage Checklist

- [x] **`customerID` excluded from feature matrix `X`**: Yes (stored separately for record mapping).
- [x] **Original `Churn` excluded from `X`**: Yes.
- [x] **Target `churn` excluded from `X`**: Yes.
- [x] **Transformers fitted ONLY on Training Data**: Yes (Scaler, Imputer, and OneHotEncoder fitted exclusively on `X_train_raw`).
- [x] **Zero overlapping customer IDs between splits**: Verified (0 overlap across train, val, and test).
- [x] **No future information or label leakage**: Verified.

---

## 5. Generated Artifacts

- **Processed Feature Matrices**:
  - `data/processed/X_train.csv` (`4,930` $\times$ `43`)
  - `data/processed/X_val.csv` (`1,056` $\times$ `43`)
  - `data/processed/X_test.csv` (`1,057` $\times$ `43`)
- **Target Vectors**:
  - `data/processed/y_train.csv`, `y_val.csv`, `y_test.csv`
- **Feature Schema**: `data/processed/feature_names.json`
- **Fitted Preprocessing Pipeline**: `models/preprocessor.joblib`
