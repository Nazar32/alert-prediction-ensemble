#!/usr/bin/env python3
"""
run_baselines.py
================
Run all seven baseline models on the Ukrainian air-alert dataset and print a
comparison table.  Reproduces the baseline columns in Table 4 of the paper.

Baselines
---------
  1. Persistence
  2. ARIMA (AR-24 per region)
  3. Logistic Regression
  4. Random Forest
  5. XGBoost
  6. LightGBM
  7. LSTM

Usage
-----
    python run_baselines.py

Output
------
Prints a summary table to stdout and saves
results/baseline_comparison_results.csv.

Note: ARIMA fits per region on the training set, but selects its decision
threshold by scanning F1 on the test set (no separate val set).  McNemar
results vs ARIMA should therefore be interpreted with caution.
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch

# Prevent OpenMP conflicts on Apple Silicon / multi-library setups
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from src.data_utils import (
    load_data, engineer_features, temporal_split, prepare_arrays,
    compute_metrics, DATA_PATH, RESULTS_DIR, SEED,
)
from src.baselines import (
    run_persistence, run_arima, run_sklearn_model, run_lstm,
)

np.random.seed(SEED)
torch.manual_seed(SEED)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}\n")

    print("Loading data...")
    raw = load_data(DATA_PATH)
    print(f"  {len(raw):,} rows | {raw['oblast'].nunique()} oblasts | "
          f"alert rate {raw['alert_occurred'].mean():.4f}\n")

    print("Engineering features (~2 min)...")
    feat = engineer_features(raw)
    train_f, val_f, test_f = temporal_split(feat)
    print(f"  Train {len(train_f):,} | Val {len(val_f):,} | Test {len(test_f):,}\n")

    X_train, y_train, scaler = prepare_arrays(train_f, fit_scaler=True)
    X_val,   y_val,   _      = prepare_arrays(val_f,   scaler=scaler)
    X_test,  y_test,  _      = prepare_arrays(test_f,  scaler=scaler)

    test_clean = test_f.dropna(subset=["target"]).reset_index(drop=True)
    print(f"Feature matrix: {X_train.shape[1]} features\n")

    results = {}

    # 1. Persistence
    print("[1/7] Persistence...")
    yt, yp, yprob = run_persistence(test_clean)
    results["Persistence"] = {**compute_metrics(yt, yp, yprob)}
    print(f"  Precision={results['Persistence']['precision']:.4f}  "
          f"Recall={results['Persistence']['recall']:.4f}")

    # 2. ARIMA (AR-24 per region, ~5 min)
    print("\n[2/7] ARIMA (AR-24 per region, ~5 min)...")
    yt, yp, yprob = run_arima(train_f, test_clean)
    results["ARIMA"] = {**compute_metrics(yt, yp, yprob)}
    print(f"  Precision={results['ARIMA']['precision']:.4f}  "
          f"Recall={results['ARIMA']['recall']:.4f}")

    # 3. Logistic Regression
    print("\n[3/7] Logistic Regression...")
    lr_clf = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=SEED
    )
    yt, yp, yprob = run_sklearn_model(
        lr_clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results["Logistic Regression"] = {**compute_metrics(yt, yp, yprob)}
    print(f"  Precision={results['Logistic Regression']['precision']:.4f}  "
          f"Recall={results['Logistic Regression']['recall']:.4f}")

    # 4. Random Forest
    print("\n[4/7] Random Forest...")
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    yt, yp, yprob = run_sklearn_model(
        rf_clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results["Random Forest"] = {**compute_metrics(yt, yp, yprob)}
    print(f"  Precision={results['Random Forest']['precision']:.4f}  "
          f"Recall={results['Random Forest']['recall']:.4f}")

    # 5. XGBoost
    print("\n[5/7] XGBoost...")
    scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        scale_pos_weight=scale_pos, eval_metric="logloss",
        tree_method="hist", device="cpu",
        random_state=SEED, n_jobs=1, verbosity=0,
    )
    yt, yp, yprob = run_sklearn_model(
        xgb_clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results["XGBoost"] = {**compute_metrics(yt, yp, yprob)}
    print(f"  Precision={results['XGBoost']['precision']:.4f}  "
          f"Recall={results['XGBoost']['recall']:.4f}")

    # 6. LightGBM
    print("\n[6/7] LightGBM...")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        is_unbalance=True, random_state=SEED, n_jobs=1, num_threads=1, verbose=-1,
    )
    yt, yp, yprob = run_sklearn_model(
        lgb_clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results["LightGBM"] = {**compute_metrics(yt, yp, yprob)}
    print(f"  Precision={results['LightGBM']['precision']:.4f}  "
          f"Recall={results['LightGBM']['recall']:.4f}")

    # 7. Standalone LSTM
    print("\n[7/7] Standalone LSTM...")
    yt, yp, yprob = run_lstm(
        X_train, y_train, X_val, y_val, X_test, y_test,
        epochs=30, device=device,
    )
    results["LSTM"] = {**compute_metrics(yt, yp, yprob)}
    print(f"  Precision={results['LSTM']['precision']:.4f}  "
          f"Recall={results['LSTM']['recall']:.4f}")

    # Summary
    metric_keys = ["precision", "recall", "f1", "accuracy", "roc_auc",
                   "TP", "FP", "TN", "FN"]
    rows = [{"model": name, **{k: r.get(k, float("nan")) for k in metric_keys}}
            for name, r in results.items()]
    summary = pd.DataFrame(rows).set_index("model")

    print("\n" + "=" * 72)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 72)
    print(summary[["precision", "recall", "f1", "accuracy", "roc_auc"]]
          .round(4).to_string())

    out_path = os.path.join(RESULTS_DIR, "baseline_comparison_results.csv")
    summary.to_csv(out_path)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
