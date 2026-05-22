#!/usr/bin/env python3
"""
Correct onset vs continuation analysis.
Onset subset:        all cases where lag_0h == 0 (currently no alert)
Continuation subset: all cases where lag_0h == 1 (alert currently active)
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
import torch; torch.manual_seed(SEED)

device = ("mps" if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

raw = load_data(DATA_PATH)
feat = engineer_features(raw)
train_f, val_f, test_f = temporal_split(feat)
val_clean  = val_f.dropna(subset=["target"]).reset_index(drop=True)
test_clean = test_f.dropna(subset=["target"]).reset_index(drop=True)

# Print split dates
print(f"Train: {train_f['hour'].min()} → {train_f['hour'].max()}")
print(f"Val:   {val_f['hour'].min()} → {val_f['hour'].max()}")
print(f"Test:  {test_f['hour'].min()} → {test_f['hour'].max()}")
print(f"Train samples: {len(val_clean.index)} val / {len(test_clean.index)} test")

X_train, y_train, scaler = prepare_arrays(train_f, fit_scaler=True)
X_val,   y_val,   _      = prepare_arrays(val_f,   scaler=scaler)
X_test,  y_test,  _      = prepare_arrays(test_f,  scaler=scaler)

print("\nTraining proposed DNN+RF model (re-run for correct analysis)...")
y_true, y_pred, y_prob, test_df, thresholds = run_proposed_dnn_rf(
    X_train, y_train, X_val, y_val, X_test, y_test,
    val_clean, test_clean, device=device,
)

lag0 = test_df["lag_0h"].values

# Correct subsets: all rows, split by current alert state
onset_mask = (lag0 == 0)      # no alert currently → predicting new onset or no alert
cont_mask  = (lag0 == 1)      # alert currently active → predicting continuation or end
persist_pred = lag0.astype(int)

def report(name, yt, yp):
    TP = int(((yt==1)&(yp==1)).sum())
    FP = int(((yt==0)&(yp==1)).sum())
    TN = int(((yt==0)&(yp==0)).sum())
    FN = int(((yt==1)&(yp==0)).sum())
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    pos_rate = yt.mean()
    print(f"{name}: n={len(yt):,}  pos_rate={pos_rate:.3f}  "
          f"Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  TP={TP}  FP={FP}  TN={TN}  FN={FN}")
    return dict(model_subset=name, n=len(yt), pos_rate=round(pos_rate,4),
                TP=TP, FP=FP, TN=TN, FN=FN,
                precision=round(prec,4), recall=round(rec,4), f1=round(f1,4))

print("\n=== ONSET SUBSET (lag_0h=0, predicts onset or no alert) ===")
r1 = report("Proposed - Onset subset",    y_true[onset_mask], y_pred[onset_mask])
r2 = report("Persistence - Onset subset", y_true[onset_mask], persist_pred[onset_mask])

print("\n=== CONTINUATION SUBSET (lag_0h=1, predicts continuation or end) ===")
r3 = report("Proposed - Continuation subset",    y_true[cont_mask], y_pred[cont_mask])
r4 = report("Persistence - Continuation subset", y_true[cont_mask], persist_pred[cont_mask])

pd.DataFrame([r1,r2,r3,r4]).to_csv("results/onset_continuation_fixed.csv", index=False)
print("\nSaved results/onset_continuation_fixed.csv")

# Also print split dates clearly
print("\n=== SPLIT DATES ===")
print(f"Training:   {train_f['hour'].min().strftime('%Y-%m-%d')} to {train_f['hour'].max().strftime('%Y-%m-%d')}")
print(f"Validation: {val_f['hour'].min().strftime('%Y-%m-%d')} to {val_f['hour'].max().strftime('%Y-%m-%d')}")
print(f"Test:       {test_f['hour'].min().strftime('%Y-%m-%d')} to {test_f['hour'].max().strftime('%Y-%m-%d')}")
print(f"Global threshold: {thresholds['_global']:.4f}")
print("\nPer-region thresholds:")
for k, v in sorted(thresholds.items()):
    if k != '_global':
        print(f"  {k}: {v:.4f}")
