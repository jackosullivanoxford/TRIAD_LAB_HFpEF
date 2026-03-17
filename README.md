# TRIAD_LAB_HFpEF

**XGBoost classifier for predicting heart failure with preserved ejection fraction (HFpEF) from clinical and laboratory biomarkers.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

HFpEF accounts for roughly half of all heart failure cases and remains diagnostically challenging. This model uses 39 routinely collected clinical and laboratory features to predict HFpEF status via gradient-boosted trees (XGBoost), with automated hyperparameter tuning and class-imbalance correction.

The training pipeline performs:
- 60/20/20 train-validation-test split
- Randomized hyperparameter search (150 iterations, 3-fold CV)
- Automatic class weighting via `scale_pos_weight`
- AUC-ROC evaluation on a held-out test set

## Quick Start

```bash
# Clone
git clone https://github.com/jackosullivanoxford/TRIAD_LAB_HFpEF.git
cd TRIAD_LAB_HFpEF

# Install dependencies
pip install -r requirements.txt

# Train
python train.py
```

### Pre-trained Model

The trained model weights are available on Hugging Face:
**[jackosullivan/TRIAD_LAB_HFPEF](https://huggingface.co/jackosullivan/TRIAD_LAB_HFPEF)**

```python
import joblib
model = joblib.load("model.joblib")
y_prob = model.predict_proba(X_new)[:, 1]
```

## Data Format

Input CSV (`data.csv`) should contain 39 predictor columns and a binary target column `HFpEF`.

| Category | Features |
|---|---|
| **Demographics** | `Age_at_echo`, `sex`, `BMI`, `Age_BMI_interaction` |
| **Comorbidities** | `ICDHTN`, `AFib`, `DM` |
| **Lipids** | `totalHDL`, `LDL`, `totalchol`, `trig`, `totalHDL_ratio`, `LDL_to_total_ratio` |
| **Metabolic** | `Creatinine`, `HbA1c` |
| **Lab means** | `mean_total_bili`, `mean_MCH`, `mean_PLT`, `mean_DBP`, `mean_pulse`, `mean_MCHC`, `mean_glucose`, `mean_AST`, `mean_urea`, `mean_neutrophil`, `mean_total_protein`, `mean_eosino`, `mean_lymphocyte_count`, `mean_wbc`, `mean_hct`, `mean_SBP`, `mean_ALT`, `mean_MCV`, `mean_monocyte`, `mean_RBC`, `mean_baso`, `mean_Hb`, `mean_calcium`, `mean_ALP` |

## Hyperparameter Search Space

| Parameter | Values |
|---|---|
| `learning_rate` | 0.05, 0.06, 0.07 |
| `n_estimators` | 100, 150, 200 |
| `max_depth` | 2, 3, 4 |
| `subsample` | 0.7, 0.8, 1.0 |
| `colsample_bytree` | 0.7, 0.8, 1.0 |
| `lambda` (L2) | 0, 0.1, 1 |
| `alpha` (L1) | 0, 0.1, 1 |

## Requirements

- Python ≥ 3.8
- xgboost ≥ 1.7.0
- scikit-learn ≥ 1.0.0
- pandas ≥ 1.5.0
- joblib (included with scikit-learn)

## Project Structure

```
TRIAD_LAB_HFpEF/
├── train.py          # Training pipeline
├── README.md
├── requirements.txt  # pip dependencies
├── data.csv          # Input data (not included)
└── model.joblib      # Trained model (also on HuggingFace)
```

## Citation

If you use this code, please cite:

```
O'Sullivan JW, et al. Multimodal Machine Learning Reveals the Genomic and Proteomic Architecture of Heart Failure with Preserved Ejection Fraction. 
https://www.medrxiv.org/content/10.64898/2026.02.07.26345811v2
```

## Contact

Jack O'Sullivan — [jackos@stanford.edu](mailto:jackos@stanford.edu) Stanford
