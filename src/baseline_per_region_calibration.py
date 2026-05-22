#!/usr/bin/env python3
"""
baseline_per_region_calibration.py
===================================
Answers reviewer question: "Does per-region precision-constrained calibration
alone explain the proposed model's precision advantage, or does the DNN+RF
architecture contribute independently?"

Runs LightGBM, XGBoost, and LSTM with the *same* per-region threshold
calibration (precision >= 75% floor, F1-maximising, per oblast) as the
proposed DNN+RF model.

Compares:
  A. Baseline (global F1-max calibration, no precision floor)  -- from paper
  B. Baseline + per-region precision calibration               -- this script
  C. Proposed DNN+RF + per-region precision calibration        -- from paper

If B ≈ C: calibration scope explains the gap; architecture does not matter.
If C >> B: DNN+RF architecture provides genuine additional benefit.

Run from project root:
    python experiments/baseline_per_region_calibration.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
import xgboost as xgb
import lightgbm as lgb

# Data utilities from proposed_model (TARGET_PRECISION=0.0 there, but we
# define our own calibrate_threshold below with the 0.75 floor).
from proposed_model import (
    load_data, engineer_features, temporal_split, prepare_arrays,
    compute_metrics,
    SEED, RESULTS_DIR,
)

DATA_PATH = "data/official_data_uk.csv"

warnings.filterwarnings("ignore")

TARGET_PRECISION = 0.75
MIN_VAL_SAMPLES  = 30      # per-region minimum for region-specific threshold
MIN_PREDICTIONS  = 5       # degenerate-threshold guard

np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Precision-oriented threshold calibration  (same as proposed_model_dnn_rf_fixed)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_threshold_prec(y_val, y_prob_val,
                              target_prec=TARGET_PRECISION,
                              min_predictions=MIN_PREDICTIONS):
    """
    Scan 199 thresholds in [0.01, 0.99].
    Return argmax F1 subject to precision >= target_prec.
    Returns None if no threshold satisfies the constraint.
    """
    best_t, best_f1 = None, -1.0
    for t in np.linspace(0.01, 0.99, 199):
        y_pred = (y_prob_val >= t).astype(int)
        if y_pred.sum() < min_predictions:
            continue
        if target_prec > 0:
            p = precision_score(y_val, y_pred, zero_division=0)
            if p < target_prec:
                continue
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


# ─────────────────────────────────────────────────────────────────────────────
# Per-region calibration wrapper
# ─────────────────────────────────────────────────────────────────────────────

def apply_per_region_calibration(y_prob_val, y_val, val_df,
                                  y_prob_test, test_df):
    """
    Given validation and test probabilities + DataFrames with 'oblast' column,
    calibrate one precision-floor threshold per region on val, apply to test.

    Returns y_true, y_pred, thresholds dict.
    """
    val_df  = val_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)
    val_df["_prob"]  = y_prob_val
    test_df["_prob"] = y_prob_test

    # Global fallback
    global_t = calibrate_threshold_prec(y_val, y_prob_val)
    if global_t is None:
        global_t = calibrate_threshold_prec(y_val, y_prob_val, target_prec=0.0)

    thresholds = {"_global": global_t}
    for oblast in test_df["oblast"].unique():
        val_mask = val_df["oblast"] == oblast
        if val_mask.sum() < MIN_VAL_SAMPLES:
            thresholds[oblast] = global_t
            continue
        y_val_r = val_df.loc[val_mask, "target"].astype(int).values
        p_val_r = val_df.loc[val_mask, "_prob"].values
        t = calibrate_threshold_prec(y_val_r, p_val_r)
        thresholds[oblast] = t if t is not None else global_t

    y_pred = np.zeros(len(test_df), dtype=int)
    for oblast in test_df["oblast"].unique():
        mask = (test_df["oblast"] == oblast).values
        t = thresholds.get(oblast, global_t)
        y_pred[mask] = (test_df.loc[mask, "_prob"].values >= t).astype(int)

    y_true = test_df["target"].astype(int).values
    return y_true, y_pred, thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn model runner (train + get probabilities)
# ─────────────────────────────────────────────────────────────────────────────

def train_sklearn_probs(clf, X_train, y_train, X_val, X_test):
    clf.fit(X_train, y_train)
    p_val  = clf.predict_proba(X_val)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]
    return p_val, p_test


# ─────────────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────────────

class StandaloneLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_lstm_probs(X_train, y_train, X_val, X_test,
                     epochs=30, batch_size=512, device="cpu"):
    Xt = torch.tensor(X_train[:, None, :], dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(X_val[:, None, :],  dtype=torch.float32).to(device)
    Xe = torch.tensor(X_test[:, None, :], dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    model     = StandaloneLSTM(X_train.shape[1]).to(device)
    pos_w     = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
                              dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val, patience, best_state = np.inf, 0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(Xv),
                torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device),
            ).item()
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= 7:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_val  = torch.sigmoid(model(Xv)).cpu().numpy().ravel()
        p_test = torch.sigmoid(model(Xe)).cpu().numpy().ravel()
    return p_val, p_test


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}")
    print(f"TARGET_PRECISION = {TARGET_PRECISION:.0%}")
    print(f"Experiment: baseline models + per-region precision-floor calibration\n")

    print("Loading data...")
    raw = load_data(DATA_PATH)
    print(f"  {len(raw):,} rows | {raw['oblast'].nunique()} oblasts | "
          f"alert rate {raw['alert_occurred'].mean():.4f}\n")

    print("Engineering features...")
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

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print("[1/3] LightGBM + per-region calibration...")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        is_unbalance=True, random_state=SEED, n_jobs=1, verbose=-1,
    )
    p_lgb_val, p_lgb_test = train_sklearn_probs(
        lgb_clf, X_train, y_train, X_val, X_test
    )
    y_true, y_pred, thr = apply_per_region_calibration(
        p_lgb_val, y_val, val_clean,
        p_lgb_test, test_clean,
    )
    m = compute_metrics(y_true, y_pred, p_lgb_test)
    results["LightGBM + per-region calib"] = m
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")
    n_region = sum(1 for k, v in thr.items()
                   if k != "_global" and v != thr["_global"])
    print(f"  Region-specific thresholds: {n_region}/25  "
          f"(global fallback: {25 - n_region})")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\n[2/3] XGBoost + per-region calibration...")
    scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        scale_pos_weight=scale_pos, eval_metric="logloss",
        random_state=SEED, n_jobs=1, verbosity=0,
    )
    p_xgb_val, p_xgb_test = train_sklearn_probs(
        xgb_clf, X_train, y_train, X_val, X_test
    )
    y_true, y_pred, thr = apply_per_region_calibration(
        p_xgb_val, y_val, val_clean,
        p_xgb_test, test_clean,
    )
    m = compute_metrics(y_true, y_pred, p_xgb_test)
    results["XGBoost + per-region calib"] = m
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")
    n_region = sum(1 for k, v in thr.items()
                   if k != "_global" and v != thr["_global"])
    print(f"  Region-specific thresholds: {n_region}/25  "
          f"(global fallback: {25 - n_region})")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    print("\n[3/3] LSTM + per-region calibration...")
    p_lstm_val, p_lstm_test = train_lstm_probs(
        X_train, y_train, X_val, X_test,
        epochs=30, device=device,
    )
    y_true, y_pred, thr = apply_per_region_calibration(
        p_lstm_val, y_val, val_clean,
        p_lstm_test, test_clean,
    )
    m = compute_metrics(y_true, y_pred, p_lstm_test)
    results["LSTM + per-region calib"] = m
    print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")
    n_region = sum(1 for k, v in thr.items()
                   if k != "_global" and v != thr["_global"])
    print(f"  Region-specific thresholds: {n_region}/25  "
          f"(global fallback: {25 - n_region})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY  (baselines from paper vs. + per-region calibration)")
    print("=" * 72)

    # Reference results from the paper (global F1-max calibration, no floor)
    paper_baselines = {
        "LightGBM (paper, global F1-max)": {
            "precision": 0.7535, "recall": 0.7659, "f1": 0.7596, "roc_auc": 0.9321
        },
        "XGBoost (paper, global F1-max)": {
            "precision": 0.7498, "recall": 0.7648, "f1": 0.7572, "roc_auc": 0.9319
        },
        "LSTM (paper, global F1-max)": {
            "precision": 0.7549, "recall": 0.7632, "f1": 0.7590, "roc_auc": 0.9283
        },
        "Proposed DNN+RF (paper, per-region)": {
            "precision": 0.7795, "recall": 0.6799, "f1": 0.7263, "roc_auc": 0.9296
        },
    }

    all_rows = []
    for name, m in paper_baselines.items():
        all_rows.append({
            "model": name,
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "roc_auc":   m["roc_auc"],
        })
    for name, m in results.items():
        all_rows.append({
            "model":     name,
            "precision": round(m["precision"], 4),
            "recall":    round(m["recall"],    4),
            "f1":        round(m["f1"],        4),
            "roc_auc":   round(m["roc_auc"],   4),
        })

    df = pd.DataFrame(all_rows).set_index("model")
    print(df[["precision", "recall", "f1", "roc_auc"]].to_string())

    out_path = os.path.join(RESULTS_DIR, "baseline_per_region_calib_results.csv")
    df.to_csv(out_path)
    print(f"\nResults saved → {out_path}")

    # ── Interpretation ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    proposed_prec = 0.7795
    for name, m in results.items():
        delta = proposed_prec - m["precision"]
        if delta > 0.005:
            verdict = f"Proposed DNN+RF still +{delta:.2%} higher → architecture contributes"
        elif delta < -0.005:
            verdict = f"Baseline exceeds proposed by {-delta:.2%} → calibration scope dominates"
        else:
            verdict = "Comparable precision → calibration scope explains most of the gap"
        print(f"  {name}: precision={m['precision']:.4f}  | {verdict}")


if __name__ == "__main__":
    main()
