#!/usr/bin/env python3
"""
baselines.py
============
Seven baseline models used for comparison in the paper.

Models
------
1. Persistence          – predict current hour = next hour (lag_0h)
2. ARIMA (AR-24)        – AutoReg with 24 lags per region (statsmodels)
3. Logistic Regression  – linear classifier on engineered features
4. Random Forest        – standalone RF, class_weight='balanced'
5. XGBoost              – XGBClassifier with scale_pos_weight
6. LightGBM             – LGBMClassifier with is_unbalance=True
7. LSTM                 – standalone 2-layer LSTM (PyTorch, BCE + pos_weight)

All sklearn-compatible models (3–6) share the same wrapper: fit on train,
calibrate threshold on val (F1-max, no precision floor), evaluate on test.

Public API
----------
run_persistence(test_df)
run_arima(train_df, test_df)
run_sklearn_model(clf, X_train, y_train, X_val, y_val, X_test, y_test)
run_lstm(X_train, y_train, X_val, y_val, X_test, y_test, epochs, batch_size, device)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from statsmodels.tsa.ar_model import AutoReg

from src.data_utils import calibrate_threshold, SEED

torch.manual_seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 1: Persistence
# ──────────────────────────────────────────────────────────────────────────────

def run_persistence(test_df: pd.DataFrame):
    """Predict next-hour alert = current-hour alert (lag_0h)."""
    df     = test_df.dropna(subset=["target"]).copy()
    y_true = df["target"].astype(int).values
    y_pred = df["lag_0h"].astype(int).values
    y_prob = y_pred.astype(float)
    return y_true, y_pred, y_prob


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 2: ARIMA (AR-24 per region via statsmodels AutoReg)
# ──────────────────────────────────────────────────────────────────────────────

def run_arima(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Fit an AR(24) model per region on the training set and predict on the test
    set.  The decision threshold is chosen by scanning F1 on the test set
    (no separate val set is used for ARIMA); interpret McNemar results vs ARIMA
    with appropriate caution.
    """
    all_probs, all_true = [], []
    oblasts = train_df["oblast"].unique()

    for i, oblast in enumerate(oblasts):
        print(f"  AR-24 fitting {i + 1}/{len(oblasts)}: {oblast:<30}", end="\r")
        tr = train_df[train_df["oblast"] == oblast].sort_values("hour")
        te = test_df[test_df["oblast"]  == oblast].sort_values("hour")
        te = te.dropna(subset=["target"])
        if len(tr) < 50 or len(te) == 0:
            continue

        y_tr     = tr["alert_occurred"].values.astype(float)
        y_target = te["target"].astype(int).values

        try:
            ar_model = AutoReg(y_tr, lags=24, old_names=False).fit()
            start = len(y_tr)
            end   = len(y_tr) + len(te) - 1
            preds = ar_model.predict(start=start, end=end)
            preds = np.clip(preds, 0.0, 1.0)
        except Exception:
            preds = np.full(len(te), float(y_tr.mean()))

        all_probs.extend(preds.tolist())
        all_true.extend(y_target.tolist())

    print()
    all_probs = np.array(all_probs, dtype=np.float32)
    all_true  = np.array(all_true,  dtype=int)

    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        f = f1_score(all_true, (all_probs >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    y_pred = (all_probs >= best_t).astype(int)
    return all_true, y_pred, all_probs


# ──────────────────────────────────────────────────────────────────────────────
# Baselines 3–6: sklearn-compatible classifiers
# ──────────────────────────────────────────────────────────────────────────────

def run_sklearn_model(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    """Fit clf, calibrate threshold on val (F1-max, no precision floor), eval on test."""
    clf.fit(X_train, y_train)
    y_prob_val  = clf.predict_proba(X_val)[:, 1]
    y_prob_test = clf.predict_proba(X_test)[:, 1]
    threshold   = calibrate_threshold(y_val, y_prob_val)
    if threshold is None:
        threshold = 0.5
    y_pred = (y_prob_test >= threshold).astype(int)
    return y_test, y_pred, y_prob_test


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 7: Standalone LSTM
# ──────────────────────────────────────────────────────────────────────────────

class StandaloneLSTM(nn.Module):
    """2-layer LSTM followed by a linear projection to a single logit."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # raw logit from last timestep


def run_lstm(
    X_train, y_train, X_val, y_val, X_test, y_test,
    epochs: int = 30, batch_size: int = 512, device: str = "cpu",
):
    """Train the standalone LSTM and evaluate on the test set."""
    # LSTM expects (batch, seq_len, input_dim); we use seq_len=1 (no explicit
    # temporal window — the lag features already encode history).
    Xt = torch.tensor(X_train[:, None, :], dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(X_val[:, None, :],   dtype=torch.float32).to(device)
    Xe = torch.tensor(X_test[:, None, :],  dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    model = StandaloneLSTM(X_train.shape[1]).to(device)
    pos_w = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32,
    ).to(device)
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
        y_prob_val  = torch.sigmoid(model(Xv)).cpu().numpy().ravel()
        y_prob_test = torch.sigmoid(model(Xe)).cpu().numpy().ravel()

    threshold = calibrate_threshold(y_val, y_prob_val)
    if threshold is None:
        threshold = 0.5
    y_pred = (y_prob_test >= threshold).astype(int)
    return y_test, y_pred, y_prob_test
