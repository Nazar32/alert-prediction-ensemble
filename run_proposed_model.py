#!/usr/bin/env python3
"""
run_proposed_model.py
=====================
Train and evaluate the proposed DNN+RF ensemble on the Ukrainian air-alert
dataset.  Reproduces the results in Table 4 of the paper.

Usage
-----
    python run_proposed_model.py

Output
------
Prints per-region threshold calibration info and final test metrics.
Saves results to results/proposed_model_results.csv.

Expected runtime: ~27 minutes on Apple M2 Max (MPS).
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from src.data_utils import (
    load_data, engineer_features, temporal_split, prepare_arrays,
    compute_metrics, DATA_PATH, RESULTS_DIR, SEED,
)
from src.model import run_proposed_dnn_rf, TARGET_PRECISION

np.random.seed(SEED)
torch.manual_seed(SEED)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}")
    print(f"Precision target: {TARGET_PRECISION:.0%}\n")

    print("Loading data...")
    raw = load_data(DATA_PATH)
    print(f"  {len(raw):,} rows | {raw['oblast'].nunique()} oblasts | "
          f"alert rate {raw['alert_occurred'].mean():.4f}\n")

    print("Engineering features (may take ~2 min)...")
    feat = engineer_features(raw)
    train_f, val_f, test_f = temporal_split(feat)
    print(f"  Train {len(train_f):,} | Val {len(val_f):,} | Test {len(test_f):,}\n")

    X_train, y_train, scaler = prepare_arrays(train_f, fit_scaler=True)
    X_val,   y_val,   _      = prepare_arrays(val_f,   scaler=scaler)
    X_test,  y_test,  _      = prepare_arrays(test_f,  scaler=scaler)

    val_clean  = val_f.dropna(subset=["target"]).reset_index(drop=True)
    test_clean = test_f.dropna(subset=["target"]).reset_index(drop=True)

    print(f"Feature matrix: {X_train.shape[1]} features\n")
    print("Training DNN+RF Proposed Ensemble...")

    y_true, y_pred, y_prob, test_df_out, region_thresholds = run_proposed_dnn_rf(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        val_clean, test_clean,
        device=device,
    )

    metrics = compute_metrics(y_true, y_pred, y_prob)

    print("\n" + "=" * 55)
    print("DNN+RF PROPOSED MODEL — TEST RESULTS")
    print("=" * 55)
    for k, v in metrics.items():
        if k in ("TP", "FP", "TN", "FN"):
            print(f"  {k:<14} {v:>10,}")
        else:
            print(f"  {k:<14} {v:>10.4f}")
    print(f"\n  precision target   {TARGET_PRECISION:>10.0%}")

    # Per-region precision summary
    from sklearn.metrics import precision_score
    oblasts = test_df_out["oblast"].unique()
    region_precisions = []
    for o in oblasts:
        mask = (test_df_out["oblast"] == o).values
        yp_r = y_pred[mask]
        yt_r = y_true[mask]
        if yp_r.sum() > 0:
            region_precisions.append(precision_score(yt_r, yp_r, zero_division=0))
    if region_precisions:
        n_meeting = sum(p >= TARGET_PRECISION for p in region_precisions)
        print(f"\n  regions meeting ≥{TARGET_PRECISION:.0%} precision: "
              f"{n_meeting}/{len(region_precisions)}")
        print(f"  mean region precision: {np.mean(region_precisions):.4f}")

    out_path = os.path.join(RESULTS_DIR, "proposed_model_results.csv")
    pd.DataFrame([{"model": "Proposed_DNN_RF", **metrics}]).to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
