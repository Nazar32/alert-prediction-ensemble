# A Per-Region Precision-Constrained Calibration Framework for Rare Critical Event Prediction in Spatially Heterogeneous Environments

Source code for the paper:

> **A Per-Region Precision-Constrained Calibration Framework for Rare Critical Event Prediction in Spatially Heterogeneous Environments**
> N. Melnyk, O. Pysarchuk, O. Korochkin
> National Technical University of Ukraine "Igor Sikorsky Kyiv Polytechnic Institute"
> *IEEE Access* (resubmission 2026)

---

## Overview

This repository contains the implementation of a model-agnostic **per-region precision-constrained calibration framework** for rare critical event prediction. The framework's core contribution is per-region threshold calibration: for each spatial subpopulation, a separate decision threshold is calibrated on the validation set to maximize F1 subject to a hard precision floor (≥75%), with a principled global fallback for sparse regions.

The framework is validated on next-hour air-alert prediction across 25 Ukrainian oblasts (March 2022 – October 2025). It is evaluated on four model families (DNN+RF ensemble, LightGBM, XGBoost, LSTM) to demonstrate that calibration scope is the dominant design decision, contributing +3–4 pp precision independently of model architecture.

### Reference (DNN+RF) instantiation

The DNN+RF ensemble is retained as the reference instantiation throughout the paper because it supports the richest ablation analysis (focal-loss vs. binary cross-entropy training, neural-network vs. tree-based representation, ensemble-weighting attribution). Practitioners requiring the highest precision in production with simpler training may prefer LightGBM or XGBoost combined with per-region calibration.

The DNN+RF pipeline:

1. **Deep Neural Network (DNN)** — four fully-connected layers (256→128→64→1) with batch normalisation, dropout, Xavier initialisation, and binary focal loss (γ=2.0, α=0.60). Trained with Adam + cosine LR annealing; early stopping on validation AUC (patience=15).
2. **Random Forest** — 300 trees, `class_weight='balanced'`, `max_depth=20`.
3. **α-weighted blend** — weight searched on validation F1 (no precision floor at this stage).
4. **Isotonic recalibration** — applied to blended probabilities on the validation set.
5. **Per-region threshold calibration** — each of the 25 oblasts gets its own decision threshold, maximising F1 subject to precision ≥ 75%. Falls back to a global threshold when fewer than 30 positive validation samples or no threshold meets the constraint.

### Cross-model evaluation

Strong gradient-boosted and recurrent baselines (LightGBM, XGBoost, LSTM) are evaluated under the **identical** per-region calibration protocol. Results on the held-out test set (n = 117,725):

| Model | Calibration scope | Precision | Recall | F1 | AUC |
|-------|------------------|-----------|--------|----|-----|
| LightGBM | Global (≥75% floor) | 75.35% | 76.59% | 75.96% | 93.21% |
| LightGBM | **Per-region (≥75%)** | **78.66%** | 69.38% | 73.73% | 93.21% |
| XGBoost  | Global (≥75% floor) | 74.98% | 76.48% | 75.72% | 93.19% |
| XGBoost  | **Per-region (≥75%)** | **78.83%** | 66.61% | 72.21% | 93.19% |
| LSTM     | Global (≥75% floor) | 75.49% | 76.32% | 75.90% | 92.83% |
| LSTM     | **Per-region (≥75%)** | **78.66%** | 66.80% | 72.25% | 92.83% |
| DNN+RF (reference) | **Per-region (≥75%)** | **77.95%** | 67.99% | 72.63% | 92.96% |

### Input features

95 features total (numeric features z-scored with `StandardScaler`; oblast dummies passed unscaled):

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

A complete listing of every feature is in [`FEATURES.md`](FEATURES.md).

---

## Repository structure

```
alert-prediction-ensemble/
├── src/                              # primary pipeline scripts (referenced by paper)
│   ├── proposed_model.py             # data loading, feature engineering, base classifiers, evaluation
│   ├── proposed_model_dnn_rf_fixed.py  # final DNN+RF ensemble + per-region calibration
│   ├── baseline_comparison.py        # all seven baselines (Persistence, ARIMA, LR, RF, XGBoost,
│   │                                 #   LightGBM, LSTM) under identical evaluation protocol
│   ├── baseline_per_region_calibration.py  # baselines retrofit with per-region calibration
│   ├── run_lgb_per_region.py         # LightGBM + per-region (Table 4)
│   ├── run_xgb_per_region.py         # XGBoost  + per-region (Table 4)
│   ├── run_lstm_per_region.py        # LSTM     + per-region (Table 4)
│   ├── mcnemar_dnn_rf.py             # McNemar's χ² tests vs. DNN+RF (Table 4)
│   ├── extract_per_region_detailed.py  # per-region TP/FP/TN/FN, thresholds, onset/continuation,
│   │                                 #   operating-point sweep (Tables 2, 5, 6)
│   └── fix_onset_continuation.py     # onset vs. continuation reanalysis (Table 5)
├── deprecated/                       # legacy single-instantiation scripts and runners
│   │                                 #   (kept for reference; not used by the current pipeline)
│   ├── model.py, baselines.py, data_utils.py     # legacy library code
│   ├── proposed_model_dnn_rf.py                  # superseded by src/proposed_model_dnn_rf_fixed.py
│   ├── run_comparison.py                         # legacy combined runner
│   └── run_proposed_model.py, run_baselines.py, run_mcnemar.py  # legacy root runners
├── data/                             # input data (real official_data_uk.csv is gitignored)
│   ├── README.md
│   └── (place official_data_uk.csv here, or use synthetic data — see below)
├── results/                          # output CSVs (gitignored)
├── generate_synthetic_data.py        # synthetic dataset generator for end-to-end pipeline tests
├── FEATURES.md                       # complete 95-feature catalogue
├── METHODS.md                        # extended methodology details
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/Nazar32/alert-prediction-ensemble.git
cd alert-prediction-ensemble
pip install -r requirements.txt
```

Python 3.10+ is required.

PyTorch GPU acceleration is optional:
- **Apple Silicon (MPS)**: install PyTorch ≥ 2.1 for macOS.
- **NVIDIA GPU (CUDA)**: install the CUDA-enabled PyTorch wheel from pytorch.org.
- **CPU fallback**: works out of the box, but training will be slower (~3× for the DNN step).

### 2. Provide a dataset

The real dataset (`official_data_uk.csv`) is restricted under a data-use agreement (see paper §VII, Data and Code Availability). To execute the full pipeline end-to-end without the real data, generate a synthetic dataset with the same schema:

```bash
python generate_synthetic_data.py --out data/synthetic_data_uk.csv --seed 42
```

This produces a multi-month event log with realistic temporal autocorrelation across 25 oblasts. Note that all metric values obtained from synthetic data will differ from the figures reported in the paper — the generator is intended for pipeline verification, not result reproduction.

To use the synthetic file with the experiment scripts, either:
- Symlink: `ln -s synthetic_data_uk.csv data/official_data_uk.csv`, or
- Edit the `DATA_PATH` constant at the top of each script.

---

## Running experiments

All scripts are run from the repository root.

### Reference DNN+RF instantiation (Table 4 row, Tables 2, 5, 6)

```bash
python src/extract_per_region_detailed.py
```

Trains the DNN+RF ensemble, calibrates per-region thresholds, writes all per-region metrics, onset/continuation split, and operating-point sweep to `results/`. Expected runtime: ~27 minutes on Apple M2 Max (MPS).

### Cross-model per-region calibration (Table 4 ablation rows)

```bash
python src/run_lgb_per_region.py    # LightGBM + per-region
python src/run_xgb_per_region.py    # XGBoost  + per-region
python src/run_lstm_per_region.py   # LSTM     + per-region
```

Each runs an independent baseline with the identical per-region calibration protocol and writes its own CSV under `results/`.

### Globally calibrated baseline comparison (Table 4 main rows)

```bash
python src/baseline_comparison.py
```

Trains and evaluates Persistence, ARIMA, Logistic Regression, Random Forest, XGBoost, LightGBM, and LSTM under their nominal global calibration. Saves `results/baseline_comparison_results.csv`.

### McNemar significance (Table 4 χ² column)

```bash
python src/mcnemar_dnn_rf.py
```

Computes McNemar's two-sided χ² (continuity-corrected) for each globally-calibrated baseline against the DNN+RF instantiation. Saves `results/mcnemar_dnn_rf_results.csv`.

---

## Output files

| File | Contents | Paper reference |
|---|---|---|
| `results/per_region_detailed.csv` | precision, recall, F1, TP/FP/TN/FN, calibrated threshold for all 25 oblasts | Table 2 |
| `results/proposed_dnn_rf_fixed_results.csv` | aggregate DNN+RF metrics | Table 4 last row |
| `results/baseline_comparison_results.csv` | aggregate metrics for the seven globally-calibrated baselines | Table 4 main rows |
| `results/lgb_per_region_results.csv` | LightGBM + per-region calibration | Table 4 ablation |
| `results/xgb_per_region_results.csv` | XGBoost + per-region | Table 4 ablation |
| `results/lstm_per_region_results.csv` | LSTM + per-region | Table 4 ablation |
| `results/mcnemar_dnn_rf_results.csv` | McNemar χ² and p-values | Table 4 χ² column |
| `results/onset_continuation_fixed.csv` | onset vs. continuation breakdown | Table 5 |
| `results/operating_points.csv` | precision/recall at six precision floors | Table 6 |

---

## Reproducibility notes

- Random seed is fixed at `SEED=42` in `src/proposed_model.py`.
- The dataset is split strictly on the time axis (70 / 15 / 15 train / val / test); no shuffling across splits. Boundary dates are printed at runtime by `extract_per_region_detailed.py` and reported in the paper's Reproducibility Details appendix.
- Small non-determinism may remain in multi-threaded tree methods (XGBoost, LightGBM) and GPU/MPS operations. Results should match the paper within ±0.02 pp precision.
- ARIMA selects its decision threshold on the test set (no separate val set is feasible for per-region AR models). McNemar comparisons involving ARIMA should be interpreted accordingly.
- The 25 per-region thresholds, exact split dates, package versions, and other reproducibility details are in the paper's Appendix A.

---

## Citation

If you use this code, please cite the paper:

```
@article{melnyk2026perregion,
  title   = {A Per-Region Precision-Constrained Calibration Framework for
             Rare Critical Event Prediction in Spatially Heterogeneous Environments},
  author  = {Melnyk, N. and Pysarchuk, O. and Korochkin, O.},
  journal = {IEEE Access},
  year    = {2026}
}
```
