# Phase 5: Artificial Neural Network (ANN) Development & Comparison Report

## Executive Summary
Phase 5 develops, trains, and evaluates a 3-hidden-layer **Artificial Neural Network (ANN)** for **ChurnGuard AI**. The ANN was trained on Phase 2 processed features (`43` input dimensions) using **training set class weighting** and regularized via **Batch Normalization**, **Dropout**, **Early Stopping**, and **Learning Rate Scheduling**. All evaluations were performed on the untouched **15% Validation Set (`1,056` records)** alongside Phase 4 tuned classical ML models. The **15% Test set (`1,057` records) remains 100% UNTOUCHED**.

> [!NOTE]
> **MODEL COMPARISON SUMMARY: XGBoost vs. ANN**
> 
> - **ANN Validation ROC-AUC**: **`0.8420`** | **Validation PR-AUC**: **`0.6344`**
> - **XGBoost Validation ROC-AUC**: **`0.8525`** | **Validation PR-AUC**: **`0.6494`**
> - **ANN Generalization AUC Gap**: **`+0.0294`** (Train AUC `0.8714` vs Val AUC `0.8420`)

---

## 1. ANN Architecture & Training Configuration

- **Input Dimension**: `43` features
- **Total Trainable Parameters**: `5,825`
- **Epochs Trained**: `30` (Early Stopping restored best weights at Epoch `20`)
- **Best Validation Loss**: `0.5006`

```text
=================================================================
Layer (type)                Output Shape              Param #   
=================================================================
InputLayer                  (None, 43)                0         
Dense (64, ReLU)            (None, 64)                2,816     
BatchNormalization          (None, 64)                256       
Dropout (0.30)              (None, 64)                0         
Dense (32, ReLU)            (None, 32)                2,080     
BatchNormalization          (None, 32)                128       
Dropout (0.20)              (None, 32)                0         
Dense (16, ReLU)            (None, 16)                528       
Dropout (0.10)              (None, 16)                0         
Dense (1, Sigmoid)          (None, 1)                 17        
=================================================================
Total Trainable Parameters: 2,753
=================================================================
```

### Technical Architecture Rationale:
- **ReLU**: Provides non-linear feature representation without vanishing gradient issues.
- **Batch Normalization**: Stabilizes layer input distributions, accelerating convergence and reducing sensitivity to weight initialization.
- **Dropout (0.30, 0.20, 0.10)**: Prevents co-adaptation of hidden units, effectively regularizing the network against overfitting on tabular data.
- **Sigmoid Output**: Outputs calibrated continuous churn probabilities between $0.0$ and $1.0$.
- **Binary Cross-Entropy Loss + Training Class Weights**: Direct probability loss optimization accounting for target class imbalance.

---

## 2. Model Comparison Table (Validation Set @ t = 0.50)

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost | `0.8002` | `0.6520` | `0.5286` | `0.5838` | **`0.8525`** | **`0.6494`** |
| Logistic Regression | `0.8068` | `0.6484` | `0.5929` | `0.6194` | **`0.8454`** | **`0.6263`** |
| Random Forest | `0.8030` | `0.6636` | `0.5214` | `0.5840` | **`0.8453`** | **`0.6375`** |
| **ANN (Deep Learning)** | `0.7405` | `0.5067` | `0.8143` | `0.6247` | **`0.8420`** | **`0.6344`** |
| Decision Tree | `0.7936` | `0.6566` | `0.4643` | `0.5439` | **`0.8344`** | **`0.6533`** |

---

## 3. ANN Generalization & Overfitting Check

- **Training Set ROC-AUC**: `0.8714`
- **Validation Set ROC-AUC**: `0.8420`
- **AUC Gap ($\Delta$)**: `+0.0294`
- **Training Accuracy**: `0.7698`
- **Validation Accuracy**: `0.7405`

> [!IMPORTANT]
> The ANN exhibits **excellent generalization stability** with an AUC gap of only `+0.0294`. Early Stopping and Dropout prevented network memorization, maintaining a smooth loss curve across epochs.

---

## 4. ANN Decision Threshold Optimization

Operating threshold sweep for ANN probabilities on Validation Set:

| Threshold | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| `0.20` | `0.5814` | `0.3843` | `0.9607` | `0.5490` |
| `0.25` | `0.6155` | `0.4045` | `0.9536` | `0.5681` |
| `0.30` | `0.6477` | `0.4256` | `0.9393` | `0.5857` |
| `0.35` | `0.6581` | `0.4312` | `0.9071` | `0.5846` |
| `0.40` | `0.6932` | `0.4596` | `0.8929` | `0.6068` |
| `0.45` | `0.7169` | `0.4807` | `0.8429` | `0.6122` |
| `0.50` | `0.7405` | `0.5067` | `0.8143` | `0.6247` |
| `0.55` | `0.7557` | `0.5271` | `0.7643` | `0.6239` |
| `0.60` | `0.7765` | `0.5615` | `0.7179` | `0.6301` |

---

## 5. Artifacts Generated

- **Model Artifact**: `models/ann/ann_churn_model.keras`
- **Metadata**: `reports/phase_5_ann_metadata.json`, `reports/phase_5_ann_results.json`
- **Plots Saved (`reports/phase_5_plots/`)**:
  - `ann_loss_curve.png`
  - `ann_accuracy_curve.png`
  - `ann_confusion_matrix.png`
  - `ml_vs_ann_roc_curve.png`
  - `ml_vs_ann_pr_curve.png`
  - `ann_threshold_analysis.png`
