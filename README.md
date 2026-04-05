# Ensemble Machine Learning for High-Precision Prediction of Rare Critical Events

Source code for the paper:

> **Ensemble Machine Learning for High-Precision Prediction of Rare Critical Events**  
> N. Melnyk, O. Pysarchuk, O. Korochkin  
> National Technical University of Ukraine "Igor Sikorsky Kyiv Polytechnic Institute"

---

## Overview

This repository contains the complete implementation of the proposed DNN+RF ensemble and all seven baseline models used for comparison in the paper.  The task is next-hour binary prediction of air-alert activation across 25 Ukrainian administrative regions (oblasts).

### Proposed model

A two-stage hybrid ensemble:

1. **Deep Neural Network (DNN)** — four fully-connected layers (256→128→64→1) with batch normalisation, dropout, Xavier initialisation, and binary focal loss (γ=2.0, α=0.60).  Trained with Adam + cosine LR annealing; early stopping on validation AUC (patience=15).
2. **Random Forest** — 300 trees, `class_weight='balanced'`, `max_depth=20`.
3. **α-weighted blend** — weight searched on validation F1 (no precision floor).
4. **Isotonic recalibration** — applied to blended probabilities on the validation set.
5. **Per-region threshold calibration** — each of the 25 oblasts gets its own decision threshold, maximising F1 subject to precision ≥ 75%.  Falls back to a global threshold when fewer than 30 validation samples or no threshold meets the constraint.

### Input features

95 features total:

| Group | Count |
|---|---|
| Temporal (hour, day, month, weekend, night, lag_0h) | 6 |
| Cyclical encoding (sin/cos of hour, day, month) | 6 |
| Lag indicators (lag_1h … lag_24h) | 24 |
| Rolling statistics (rate/count/std × 6 windows) | 18 |
| Momentum & volatility | 5 |
| State transition (alert_duration, hours_since_last_alert) | 2 |
| Spatial – national (cross-region alert rate + rolling) | 5 |
| Spatial – neighbours (adjacent-oblast spillover) | 4 |
| Oblast identity (one-hot, 25 regions) | 25 |

Continuous features are z-scored with `StandardScaler`; oblast dummies are passed unscaled.

---

## Repository structure

```
alert-prediction-ensemble/
├── src/
│   ├── __init__.py
│   ├── data_utils.py       # data loading, feature engineering, metrics
│   ├── model.py            # DNN+RF proposed model
│   └── baselines.py        # seven baseline models
├── data/
│   └── README.md           # dataset format and placement instructions
├── results/                # output CSVs written here at runtime
├── run_proposed_model.py   # train & evaluate the proposed DNN+RF ensemble
├── run_baselines.py        # train & evaluate all seven baselines
├── run_mcnemar.py          # full comparison + McNemar significance tests
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd alert-prediction-ensemble
pip install -r requirements.txt
```

Python 3.10+ is required (uses `X | Y` union type syntax).

PyTorch GPU acceleration is optional:
- **Apple Silicon (MPS)**: install PyTorch ≥ 2.1 for macOS.
- **NVIDIA GPU (CUDA)**: install the CUDA-enabled PyTorch wheel from pytorch.org.
- **CPU fallback**: works out of the box, but training will be slower (~3× for the DNN step).

### 2. Place the dataset

Copy `official_data_uk.csv` into the `data/` directory.  See `data/README.md` for the expected format and statistics.

---

## Running experiments

All scripts are run from the repository root.

### Proposed model only

```bash
python run_proposed_model.py
```

Trains the DNN+RF ensemble, prints per-region calibration details and final test metrics, and saves `results/proposed_model_results.csv`.

Expected runtime: ~27 minutes on Apple M2 Max (MPS).

### Baselines only

```bash
python run_baselines.py
```

Trains and evaluates all seven baseline models, prints a comparison table, and saves `results/baseline_comparison_results.csv`.

### Full comparison + McNemar tests

```bash
python run_mcnemar.py
```

Trains the proposed model and all baselines in sequence, computes McNemar's test (two-sided, with continuity correction) for each proposed-vs-baseline pair, and saves `results/mcnemar_results.csv`.

---

## Output files

| File | Contents |
|---|---|
| `results/proposed_model_results.csv` | precision, recall, F1, AUC, TP/FP/TN/FN for the proposed model |
| `results/baseline_comparison_results.csv` | same metrics for all seven baselines |
| `results/mcnemar_results.csv` | χ² statistic, p-value, and significance flags for each comparison |

---

## Reproducibility notes

- Random seed is fixed at `SEED=42` in `src/data_utils.py`.
- The dataset is split strictly on the time axis (70 / 15 / 15 train / val / test); no shuffling across splits.
- Small non-determinism may remain in multi-threaded tree methods (XGBoost, LightGBM) and GPU operations.  Results should match the paper within rounding.
- ARIMA selects its decision threshold on the test set (no separate val set).  McNemar comparisons involving ARIMA should be interpreted with caution.
- The `run_mcnemar.py` script re-trains all models from scratch; it does not load pre-saved predictions.

---

## Citation

If you use this code, please cite the paper:

```
@article{melnyk2026ensemble,
  title   = {Ensemble Machine Learning for High-Precision Prediction of Rare Critical Events},
  author  = {Melnyk, N. and Pysarchuk, O. and Korochkin, O.},
  journal = {Informatica},
  year    = {2026}
}
```
