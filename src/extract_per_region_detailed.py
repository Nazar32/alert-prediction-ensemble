#!/usr/bin/env python3
"""
Extract per-region detailed metrics from the proposed DNN+RF model.
Saves a full 25-region table (TP/FP/TN/FN, per-region threshold, alert rate, etc.)
and onset vs. continuation split metrics.
Run from the project root: python experiments/extract_per_region_detailed.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from proposed_model import (
    load_data, engineer_features, temporal_split, prepare_arrays, SEED,
)
from proposed_model_dnn_rf_fixed import run_proposed_dnn_rf

DATA_PATH = "data/official_data_uk.csv"
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)

device = ("mps" if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Loading data...")
raw = load_data(DATA_PATH)

print("Engineering features (may take ~2 min)...")
feat = engineer_features(raw)

print("Splitting...")
train_f, val_f, test_f = temporal_split(feat)

# Get split boundary dates for Reproducibility Appendix
train_f_clean = train_f.dropna(subset=["target"])
val_f_clean   = val_f.dropna(subset=["target"])
test_f_clean  = test_f.dropna(subset=["target"])

train_hours = train_f["hour"].sort_values()
val_hours   = val_f["hour"].sort_values()
test_hours  = test_f["hour"].sort_values()

print(f"\nSplit dates:")
print(f"  Train:      {train_hours.min()} → {train_hours.max()}  ({len(train_f_clean):,} samples)")
print(f"  Validation: {val_hours.min()} → {val_hours.max()}  ({len(val_f_clean):,} samples)")
print(f"  Test:       {test_hours.min()} → {test_hours.max()}  ({len(test_f_clean):,} samples)")

X_train, y_train, scaler = prepare_arrays(train_f, fit_scaler=True)
X_val,   y_val,   _      = prepare_arrays(val_f,   scaler=scaler)
X_test,  y_test,  _      = prepare_arrays(test_f,  scaler=scaler)

print("\nTraining proposed DNN+RF model...")
y_true, y_pred, y_prob, test_df, thresholds = run_proposed_dnn_rf(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    val_f_clean, test_f_clean,
    device=device,
)

# ─────────────────────────────────────────────────────────────────
# 1. Per-region detailed table
# ─────────────────────────────────────────────────────────────────
test_df["y_pred"] = y_pred
test_df["y_true"] = y_true

# Compute alert rate from full dataset per region (not just test)
alert_rates = raw.groupby("oblast")["alert_occurred"].mean().rename("alert_rate_overall")

rows = []
for oblast in sorted(test_df["oblast"].unique()):
    mask = test_df["oblast"] == oblast
    yt = test_df.loc[mask, "y_true"].values
    yp = test_df.loc[mask, "y_pred"].values
    prob = test_df.loc[mask, "ens_prob"].values
    thr  = thresholds.get(oblast, thresholds["_global"])

    TP = int(((yt == 1) & (yp == 1)).sum())
    FP = int(((yt == 0) & (yp == 1)).sum())
    TN = int(((yt == 0) & (yp == 0)).sum())
    FN = int(((yt == 1) & (yp == 0)).sum())
    n  = int(mask.sum())
    alert_rate = float(yt.mean())

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    rows.append({
        "oblast": oblast,
        "alert_rate": alert_rate,
        "threshold": round(thr, 3),
        "n_test": n,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    })

per_region_df = pd.DataFrame(rows)
per_region_df = per_region_df.sort_values("alert_rate")
per_region_df.to_csv("results/per_region_detailed.csv", index=False)
print("\nSaved results/per_region_detailed.csv")
print(per_region_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────
# 2. Onset vs continuation split
# ─────────────────────────────────────────────────────────────────
# Onset:        lag_0h == 0 and y_true == 1  (new alert starting)
# Continuation: lag_0h == 1 and y_true == 1  (alert already active)
# Non-event:    y_true == 0

lag0 = test_df["lag_0h"].values  # current alert state at time t

onset_mask = (lag0 == 0) & (y_true == 1)
cont_mask  = (lag0 == 1) & (y_true == 1)
non_mask   = y_true == 0

def split_metrics(yt, yp, label):
    TP = int(((yt == 1) & (yp == 1)).sum())
    FP = int(((yt == 0) & (yp == 1)).sum())
    TN = int(((yt == 0) & (yp == 0)).sum())
    FN = int(((yt == 1) & (yp == 0)).sum())
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    print(f"\n{label} (n={len(yt):,}): Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  TP={TP}  FP={FP}  TN={TN}  FN={FN}")
    return {"split": label, "n": len(yt), "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

print("\n" + "="*60)
print("ONSET vs CONTINUATION SPLIT")
print("="*60)

onset_yt  = y_true[onset_mask]
onset_yp  = y_pred[onset_mask]
cont_yt   = y_true[cont_mask]
cont_yp   = y_pred[cont_mask]

r_onset = split_metrics(onset_yt, onset_yp, "Onset (lag_0h=0 & true=1)")
r_cont  = split_metrics(cont_yt,  cont_yp,  "Continuation (lag_0h=1 & true=1)")

# Persistence baseline for comparison
persist_pred = lag0.astype(int)  # predict: whatever current state is
r_persist_onset = split_metrics(
    y_true[onset_mask], persist_pred[onset_mask], "Persistence - Onset"
)
r_persist_cont  = split_metrics(
    y_true[cont_mask],  persist_pred[cont_mask],  "Persistence - Continuation"
)

split_df = pd.DataFrame([r_onset, r_cont, r_persist_onset, r_persist_cont])
split_df.to_csv("results/onset_continuation_results.csv", index=False)
print("\nSaved results/onset_continuation_results.csv")

# ─────────────────────────────────────────────────────────────────
# 3. Operating point table (precision floors → resulting recall on VAL set)
# ─────────────────────────────────────────────────────────────────
from proposed_model_dnn_rf_fixed import calibrate_threshold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from proposed_model import compute_metrics

# We already have val probabilities saved in thresholds; rebuild from run
# Just use test probabilities at different global thresholds to show trade-off
print("\n" + "="*60)
print("OPERATING POINT TABLE (TEST SET, global threshold sweep)")
print("="*60)
op_rows = []
for floor in [0.0, 0.65, 0.70, 0.75, 0.80, 0.85]:
    t = calibrate_threshold(y_true, y_prob, target_prec=floor)
    if t is None:
        t = calibrate_threshold(y_true, y_prob, target_prec=0.0)
    if t is None:
        continue
    yp = (y_prob >= t).astype(int)
    prec = precision_score(y_true, yp, zero_division=0)
    rec  = recall_score(y_true, yp, zero_division=0)
    f1   = f1_score(y_true, yp, zero_division=0)
    tp   = int(((y_true==1)&(yp==1)).sum())
    fp   = int(((y_true==0)&(yp==1)).sum())
    fn   = int(((y_true==1)&(yp==0)).sum())
    print(f"  Floor={floor:.0%}  t={t:.3f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  TP={tp}  FP={fp}  FN={fn}")
    op_rows.append({"precision_floor": floor, "threshold": round(t,3),
                    "precision": round(prec,4), "recall": round(rec,4),
                    "f1": round(f1,4), "TP": tp, "FP": fp, "FN": fn})

pd.DataFrame(op_rows).to_csv("results/operating_points.csv", index=False)
print("Saved results/operating_points.csv")
print("\nDONE.")
