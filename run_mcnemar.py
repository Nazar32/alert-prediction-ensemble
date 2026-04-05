#!/usr/bin/env python3
"""
run_mcnemar.py
==============
Run the proposed DNN+RF model and all seven baselines, then compute McNemar's
test comparing the proposed model against each baseline.  Reproduces Table 5
(statistical significance) and the full comparison in Table 4 of the paper.

Usage
-----
    python run_mcnemar.py

Output
------
Prints the full metric table and McNemar results to stdout.
Saves results/mcnemar_results.csv.

Note
----
ARIMA selects its threshold on the test set; its McNemar statistic should
be interpreted with caution.
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch

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
    compute_metrics, mcnemar_test, DATA_PATH, RESULTS_DIR, SEED,
)
from src.model import run_proposed_dnn_rf, TARGET_PRECISION
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

    val_clean  = val_f.dropna(subset=["target"]).reset_index(drop=True)
    test_clean = test_f.dropna(subset=["target"]).reset_index(drop=True)

    print(f"Feature matrix: {X_train.shape[1]} features\n")

    results = {}

    # ── Proposed DNN+RF ───────────────────────────────────────────────────────
    print("=" * 60)
    print(f"[1/8] Proposed DNN+RF (precision-calibrated, ≥{TARGET_PRECISION:.0%})...")
    print("=" * 60)
    y_true, y_pred_prop, y_prob, _, _ = run_proposed_dnn_rf(
        X_train, y_train, X_val, y_val, X_test, y_test,
        val_clean, test_clean, device=device,
    )
    m = compute_metrics(y_true, y_pred_prop, y_prob)
    results["Proposed"] = {"y_true": y_true, "y_pred": y_pred_prop, **m}
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}\n")

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("[2/8] Persistence...")
    yt, yp, yprob = run_persistence(test_clean)
    m = compute_metrics(yt, yp, yprob)
    results["Persistence"] = {"y_true": yt, "y_pred": yp, **m}
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}")

    print("\n[3/8] ARIMA (AR-24 per region, ~5 min)...")
    yt, yp, yprob = run_arima(train_f, test_clean)
    m = compute_metrics(yt, yp, yprob)
    results["ARIMA"] = {"y_true": yt, "y_pred": yp, **m}
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}")

    print("\n[4/8] Logistic Regression...")
    lr_clf = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=SEED
    )
    yt, yp, yprob = run_sklearn_model(
        lr_clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results["Logistic Regression"] = {
        "y_true": yt, "y_pred": yp, **compute_metrics(yt, yp, yprob)
    }
    print(f"  Precision={results['Logistic Regression']['precision']:.4f}  "
          f"Recall={results['Logistic Regression']['recall']:.4f}")

    print("\n[5/8] Random Forest...")
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    yt, yp, yprob = run_sklearn_model(
        rf_clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results["Random Forest"] = {
        "y_true": yt, "y_pred": yp, **compute_metrics(yt, yp, yprob)
    }
    print(f"  Precision={results['Random Forest']['precision']:.4f}  "
          f"Recall={results['Random Forest']['recall']:.4f}")

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
    results["XGBoost"] = {
        "y_true": yt, "y_pred": yp, **compute_metrics(yt, yp, yprob)
    }
    print(f"  Precision={results['XGBoost']['precision']:.4f}  "
          f"Recall={results['XGBoost']['recall']:.4f}")

    print("\n[7/8] LightGBM...")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        is_unbalance=True, random_state=SEED, n_jobs=1, num_threads=1, verbose=-1,
    )
    yt, yp, yprob = run_sklearn_model(
        lgb_clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results["LightGBM"] = {
        "y_true": yt, "y_pred": yp, **compute_metrics(yt, yp, yprob)
    }
    print(f"  Precision={results['LightGBM']['precision']:.4f}  "
          f"Recall={results['LightGBM']['recall']:.4f}")

    print("\n[8/8] Standalone LSTM...")
    yt, yp, yprob = run_lstm(
        X_train, y_train, X_val, y_val, X_test, y_test,
        epochs=30, device=device,
    )
    results["LSTM"] = {
        "y_true": yt, "y_pred": yp, **compute_metrics(yt, yp, yprob)
    }
    print(f"  Precision={results['LSTM']['precision']:.4f}  "
          f"Recall={results['LSTM']['recall']:.4f}")

    # ── McNemar's tests ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("McNemar's test: Proposed DNN+RF vs each baseline")
    print("=" * 65)

    prop_true = results["Proposed"]["y_true"]
    prop_pred = results["Proposed"]["y_pred"]

    baselines = ["Persistence", "ARIMA", "Logistic Regression",
                 "Random Forest", "XGBoost", "LightGBM", "LSTM"]
    mcnemar_rows = []
    for name in baselines:
        r = results[name]
        n = min(len(prop_true), len(r["y_pred"]))
        stat, pval = mcnemar_test(prop_true[:n], prop_pred[:n], r["y_pred"][:n])
        sig = ("***" if pval < 0.001 else
               "**"  if pval < 0.01  else
               "*"   if pval < 0.05  else "ns")
        print(f"  vs {name:<22}  chi2={stat:>9.4f}  p={pval:.6f}  {sig}")
        mcnemar_rows.append({
            "baseline":         name,
            "chi2_statistic":   round(stat, 4),
            "p_value":          round(pval, 6),
            "significant_p005": pval < 0.05,
            "significant_p001": pval < 0.001,
        })

    out_path = os.path.join(RESULTS_DIR, "mcnemar_results.csv")
    pd.DataFrame(mcnemar_rows).set_index("baseline").to_csv(out_path)
    print(f"\nMcNemar results saved → {out_path}")

    # ── Full metric summary ───────────────────────────────────────────────────
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
