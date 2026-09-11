# Phase 7: Final Model Selection & Untouched Test-Set Evaluation Report

## Executive Summary
Phase 7 executes the final, unbiased evaluation of **ChurnGuard AI** on the completely untouched **15% Test Set (`1,057` records)**. Model selection was finalized in Phase 4 based strictly on **Validation Set ROC-AUC**, selecting the tuned **XGBoost Classifier**. The official decision threshold was previously established on validation data at **`0.30`** and remained strictly locked for this evaluation. **No retraining, tuning, or threshold adjustment occurred during or after test set evaluation.**

> [!IMPORTANT]
> **UNBIASED TEST EVALUATION SUMMARY**
> 
> - **Test ROC-AUC**: **`0.8408`** (Validation ROC-AUC `0.8525` | Gap = `+0.0117`)
> - **Test PR-AUC**: **`0.6639`** (Validation PR-AUC `0.6494` | Gap = `-0.0145`)
> - **Test Metrics @ t = 0.30**: Accuracy = **`0.7682`** | Precision = **`0.5476`** | Recall = **`0.7367`** | F1 = **`0.6282`**
> - **Generalization Assessment**: **The test performance is broadly consistent with validation performance, demonstrating robust generalization with a minor ROC-AUC gap of +0.0117.**

---

## 1. Final Model Architecture & Configuration

The final deployed candidate is the **Phase 4 Tuned XGBoost Classifier** (`models/final/churnguard_xgboost_final.joblib`).

```text
=================================================================
Hyperparameter                     Configured Value
=================================================================
n_estimators                       150
max_depth                          3
learning_rate                      0.05
min_child_weight                   1
subsample                          0.8
colsample_bytree                   0.6
gamma                              0
reg_alpha                          0.1
reg_lambda                         0.1
scale_pos_weight                   1.0
random_state                       42
eval_metric                        logloss
=================================================================
```

---

## 2. Validation vs. Test Set Comparison Table

Below is the side-by-side performance comparison between the 15% Validation Set (`1,056` records) and the 15% Test Set (`1,057` records) at the official operating threshold ($t = 0.30$):

| Performance Metric | Validation Set | Test Set | Difference (Val - Test) | Metric Category |
| :--- | :---: | :---: | :---: | :--- |
| **ROC-AUC** | `0.8525` | **`0.8408`** | **`+0.0117`** | Threshold-Independent |
| **PR-AUC** | `0.6494` | **`0.6639`** | **`-0.0145`** | Threshold-Independent |
| **Accuracy** | `0.7642` | **`0.7682`** | `-0.0040` | Threshold-Dependent ($t=0.30$) |
| **Precision** | `0.5366` | **`0.5476`** | `-0.0110` | Threshold-Dependent ($t=0.30$) |
| **Recall** | `0.8107` | **`0.7367`** | `+0.0740` | Threshold-Dependent ($t=0.30$) |
| **F1-Score** | `0.6458` | **`0.6282`** | `+0.0176` | Threshold-Dependent ($t=0.30$) |

---

## 3. Test Set Confusion Matrix Breakdown (@ t = 0.30)

```text
=================================================================
                      Predicted No Churn      Predicted Churn
=================================================================
Actual No Churn       TN = 605                FP = 171
Actual Churn          FN = 74                 TP = 207
=================================================================
Total Test Samples: 1,057 | Actual Churners: 281 | Actual Non-Churners: 776
```

### Technical Breakdown:
- **True Negatives (TN = 605)**: Non-churning customers correctly predicted as lower risk.
- **True Positives (TP = 207)**: High-risk churning customers correctly flagged for retention intervention (**73.67% Recall**).
- **False Positives (FP = 171)**: Non-churning customers flagged for intervention, representing manageable marketing outreach cost.
- **False Negatives (FN = 74)**: Actual churners missed by the model (**26.33% miss rate**).

---

## 4. Generalization & Risk Assessment

> [!NOTE]
> **GENERALIZATION CONCLUSION**
> 
> The ROC-AUC degradation between Validation (`0.8525`) and Test (`0.8408`) is only **`0.0117`** (`1.17%`), and Test PR-AUC (`0.6639`) actually improved over Validation PR-AUC (`0.6494`). This confirms that the model generalizes robustly to unseen customer data without overfitting.

---

## 5. Test Set Integrity & Isolation Checks

| Integrity Check | Status | Description |
| :--- | :---: | :--- |
| **Test Sample Count (1,057)** | PASS | Exact match with Phase 2 split specification. |
| **Feature Dimension (43)** | PASS | Exact match with processed 43-feature schema. |
| **No Missing / NaN Values** | PASS | Zero missing values across X_test and y_test. |
| **No Infinite Values** | PASS | Zero infinite values across feature matrices. |
| **Feature Schema Alignment** | PASS | Exact feature name and order alignment. |
| **Locked Threshold (0.30)** | PASS | Threshold maintained strictly without test set optimization. |
| **Single-Pass Evaluation** | PASS | Model evaluated exactly once on test set. |
| **No Test Data Leakage** | PASS | Zero test records used during model fitting or tuning. |

---

## 6. Generated Artifacts

- **Final Model Artifact**: `models/final/churnguard_xgboost_final.joblib`
- **Test Predictions CSV**: `data/predictions/final_test_predictions.csv`
- **JSON Results**: `reports/phase_7_final_test_results.json`
- **Markdown Report**: `reports/phase_7_final_test_evaluation.md`
- **Plots Saved (`reports/phase_7_plots/`)**:
  - `final_test_confusion_matrix.png`
  - `final_test_roc_curve.png`
  - `final_test_pr_curve.png`
  - `final_test_probability_distribution.png`
  - `final_test_calibration_curve.png`
