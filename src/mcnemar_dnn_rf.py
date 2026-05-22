#!/usr/bin/env python3
"""
mcnemar_dnn_rf.py
=================
Compute McNemar's test comparing the DNN+RF precision-calibrated ensemble
(the model described in the paper) against all paper baselines:
  Persistence, ARIMA (AR-24), Logistic Regression, Random Forest,
  XGBoost, LightGBM, LSTM.

The existing mcnemar_test_results.csv was computed against a different model
variant (FT-Transformer+RF+CatBoost, global threshold).  This script
produces correct McNemar statistics for the DNN+RF model at TARGET_PRECISION=0.75.

Output:  results/mcnemar_dnn_rf_results.csv

Run from the project root:
    python experiments/mcnemar_dnn_rf.py
"""

import os, sys, warnings
# Must be set before any OpenMP-linked library loads (fixes XGBoost segfault on Apple Silicon)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["LIGHTGBM_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch

# ── Shared utilities (data loading, feature engineering, metrics) ─────────────
from proposed_model import (
    load_data, engineer_features, temporal_split, prepare_arrays,
    compute_metrics, mcnemar_test,
    SEED, RESULTS_DIR,
)

# ── DNN+RF proposed model (precision-calibrated, TARGET_PRECISION=0.75) ───────
from proposed_model_dnn_rf_fixed import (
    run_proposed_dnn_rf,
    calibrate_threshold as calibrate_threshold_precision,
    TARGET_PRECISION,
)

# ── Baselines ─────────────────────────────────────────────────────────────────
from baseline_comparison import (
    run_persistence,
    run_arima,
    run_sklearn_model,
    run_lstm,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = "data/official_data_uk.csv"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}")
    print(f"Proposed model precision target: {TARGET_PRECISION:.0%}\n")

    # ── Load & engineer ───────────────────────────────────────────────────────
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

    val_f_clean  = val_f.dropna(subset=["target"]).reset_index(drop=True)
    test_f_clean = test_f.dropna(subset=["target"]).reset_index(drop=True)

    print(f"Feature matrix: {X_train.shape[1]} features\n")

    results = {}

    # ── Proposed DNN+RF (precision-calibrated) ────────────────────────────────
    print("=" * 60)
    print(f"[1/8] Proposed DNN+RF (precision-calibrated, ≥{TARGET_PRECISION:.0%})...")
    print("=" * 60)
    y_true, y_pred_prop, y_prob, test_df_out, _ = run_proposed_dnn_rf(
        X_train, y_train, X_val, y_val, X_test, y_test,
        val_f_clean, test_f_clean, device=device,
    )
    m = compute_metrics(y_true, y_pred_prop, y_prob)
    results["Proposed"] = dict(y_true=y_true, y_pred=y_pred_prop, **m)
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}\n")

    # ── Baselines ─────────────────────────────────────────────────────────────
    # Persistence
    print("[2/8] Persistence...")
    yt, yp, yprob = run_persistence(test_f_clean)
    m = compute_metrics(yt, yp, yprob)
    results["Persistence"] = dict(y_true=yt, y_pred=yp, **m)
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}")

    # ARIMA
    print("\n[3/8] ARIMA (AR-24 per region, ~5 min)...")
    yt, yp, yprob = run_arima(train_f, test_f_clean)
    m = compute_metrics(yt, yp, yprob)
    results["ARIMA"] = dict(y_true=yt, y_pred=yp, **m)
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}")

    # Logistic Regression
    print("\n[4/8] Logistic Regression...")
    lr_clf = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=SEED
    )
    yt, yp, yprob = run_sklearn_model(
        lr_clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results["Logistic Regression"] = dict(y_true=yt, y_pred=yp,
                                           **compute_metrics(yt, yp, yprob))
    print(f"  Precision={results['Logistic Regression']['precision']:.4f}  "
          f"Recall={results['Logistic Regression']['recall']:.4f}")

    # Random Forest
    print("\n[5/8] Random Forest...")
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    yt, yp, yprob = run_sklearn_model(
        rf_clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results["Random Forest"] = dict(y_true=yt, y_pred=yp,
                                     **compute_metrics(yt, yp, yprob))
    print(f"  Precision={results['Random Forest']['precision']:.4f}  "
          f"Recall={results['Random Forest']['recall']:.4f}")

    # XGBoost
    print("\n[6/8] XGBoost...")
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
    results["XGBoost"] = dict(y_true=yt, y_pred=yp,
                               **compute_metrics(yt, yp, yprob))
    print(f"  Precision={results['XGBoost']['precision']:.4f}  "
          f"Recall={results['XGBoost']['recall']:.4f}")

    # LightGBM
    print("\n[7/8] LightGBM...")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        is_unbalance=True, random_state=SEED, n_jobs=1, num_threads=1, verbose=-1,
    )
    yt, yp, yprob = run_sklearn_model(
        lgb_clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results["LightGBM"] = dict(y_true=yt, y_pred=yp,
                                **compute_metrics(yt, yp, yprob))
    print(f"  Precision={results['LightGBM']['precision']:.4f}  "
          f"Recall={results['LightGBM']['recall']:.4f}")

    # LSTM
    print("\n[8/8] Standalone LSTM...")
    yt, yp, yprob = run_lstm(
        X_train, y_train, X_val, y_val, X_test, y_test,
        epochs=30, device=device,
    )
    results["LSTM"] = dict(y_true=yt, y_pred=yp,
                            **compute_metrics(yt, yp, yprob))
    print(f"  Precision={results['LSTM']['precision']:.4f}  "
          f"Recall={results['LSTM']['recall']:.4f}")

    # ── McNemar's tests ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("McNemar's test: Proposed DNN+RF (precision-calibrated) vs baselines")
    print("=" * 65)

    prop_true = results["Proposed"]["y_true"]
    prop_pred = results["Proposed"]["y_pred"]

    mcnemar_rows = []
    baselines = ["Persistence", "ARIMA", "Logistic Regression",
                 "Random Forest", "XGBoost", "LightGBM", "LSTM"]

    for name in baselines:
        r = results[name]
        n = min(len(prop_true), len(r["y_pred"]))
        stat, pval = mcnemar_test(prop_true[:n], prop_pred[:n], r["y_pred"][:n])
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"  vs {name:<22}  chi2={stat:>9.4f}  p={pval:.6f}  {sig}")
        mcnemar_rows.append({
            "baseline":          name,
            "chi2_statistic":    round(stat, 4),
            "p_value":           round(pval, 6),
            "significant_p005":  pval < 0.05,
            "significant_p001":  pval < 0.001,
        })

    mcnemar_df = pd.DataFrame(mcnemar_rows).set_index("baseline")
    out_path = os.path.join(RESULTS_DIR, "mcnemar_dnn_rf_results.csv")
    mcnemar_df.to_csv(out_path)
    print(f"\nSaved → {out_path}")

    # ── Baseline metric summary ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Model':<25} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 72)
    for name in ["Proposed"] + baselines:
        m = results[name]
        marker = " <<<" if name == "Proposed" else ""
        print(f"{name:<25} {m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['f1']:>7.4f} {m['roc_auc']:>7.4f}{marker}")
    print("=" * 72)


if __name__ == "__main__":
    main()
