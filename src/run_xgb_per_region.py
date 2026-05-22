#!/usr/bin/env python3
"""Run XGBoost with per-region calibration. Save results to disk."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, f1_score
import xgboost as xgb

from proposed_model import (
    load_data, engineer_features, temporal_split, prepare_arrays,
    compute_metrics, SEED,
)

DATA_PATH = "data/official_data_uk.csv"
TARGET_PRECISION = 0.75
np.random.seed(SEED)

def calibrate_threshold_prec(y_val, y_prob_val, target_prec=TARGET_PRECISION):
    best_t, best_f1 = None, -1.0
    for t in np.linspace(0.01, 0.99, 199):
        y_pred = (y_prob_val >= t).astype(int)
        if y_pred.sum() < 5:
            continue
        if target_prec > 0:
            p = precision_score(y_val, y_pred, zero_division=0)
            if p < target_prec:
                continue
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

raw = load_data(DATA_PATH)
feat = engineer_features(raw)
train_f, val_f, test_f = temporal_split(feat)
X_train, y_train, scaler = prepare_arrays(train_f, fit_scaler=True)
X_val, y_val, _          = prepare_arrays(val_f,   scaler=scaler)
X_test, y_test, _        = prepare_arrays(test_f,  scaler=scaler)
val_clean  = val_f.dropna(subset=["target"]).reset_index(drop=True)
test_clean = test_f.dropna(subset=["target"]).reset_index(drop=True)

print("Training XGBoost...")
scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
clf = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
    scale_pos_weight=scale_pos, eval_metric="logloss",
    random_state=SEED, n_jobs=1, verbosity=0)
clf.fit(X_train, y_train)
p_val  = clf.predict_proba(X_val)[:, 1]
p_test = clf.predict_proba(X_test)[:, 1]

global_t = calibrate_threshold_prec(y_val, p_val)
if global_t is None:
    global_t = calibrate_threshold_prec(y_val, p_val, target_prec=0.0)

val_clean["_prob"]  = p_val
test_clean["_prob"] = p_test
thresholds = {"_global": global_t}
for oblast in test_clean["oblast"].unique():
    vm = val_clean["oblast"] == oblast
    if vm.sum() < 30:
        thresholds[oblast] = global_t
        continue
    t = calibrate_threshold_prec(val_clean.loc[vm, "target"].astype(int).values,
                                  val_clean.loc[vm, "_prob"].values)
    thresholds[oblast] = t if t is not None else global_t

y_pred = np.zeros(len(test_clean), dtype=int)
for oblast in test_clean["oblast"].unique():
    mask = (test_clean["oblast"] == oblast).values
    t = thresholds.get(oblast, global_t)
    y_pred[mask] = (test_clean.loc[mask, "_prob"].values >= t).astype(int)

y_true = test_clean["target"].astype(int).values
m = compute_metrics(y_true, y_pred, p_test)
print(f"XGBoost+per-region: Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  "
      f"F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")
pd.DataFrame([{"model": "XGBoost+per-region", **m}]).to_csv(
    "results/xgb_per_region_results.csv", index=False)
print("Saved results/xgb_per_region_results.csv")
