#!/usr/bin/env python3
"""
model.py
========
DNN + Random Forest ensemble — the proposed model from the paper.

Architecture
------------
DeepNN:  input → 256 → BN → Dropout(0.2)
                    → 128 → BN → Dropout(0.2)
                    →  64 →      Dropout(0.1)
                    →   1 → Sigmoid
         Xavier initialisation on all linear layers.

Training
--------
- Loss:      Binary focal loss  (γ=2.0, α=0.60)
- Optimiser: Adam (lr=1e-3, weight_decay=1e-4) + CosineAnnealingLR
- Patience:  15 epochs on validation AUC (early stopping)

RF:     300 trees, max_depth=20, class_weight='balanced'
Blend:  α_NN searched on validation F1 (no precision floor)
Calibration:
  1. Isotonic regression on the blended val probabilities
  2. Per-region threshold search (precision floor ≥ TARGET_PRECISION=0.75)
     Falls back to a global threshold when fewer than 30 val samples or
     when no threshold meets the precision constraint.

Public API
----------
run_proposed_dnn_rf(X_train, y_train, X_val, y_val, X_test, y_test,
                    val_df, test_df, device, epochs, batch_size)
    → y_true, y_pred, y_prob, test_df_with_probs, thresholds
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression

from src.data_utils import calibrate_threshold, SEED

TARGET_PRECISION = 0.75   # minimum precision enforced during threshold calibration

torch.manual_seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Architecture
# ──────────────────────────────────────────────────────────────────────────────

class DeepNN(nn.Module):
    """Four fully-connected layers with batch normalisation and dropout."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Focal loss  (Lin et al., 2017)
# ──────────────────────────────────────────────────────────────────────────────

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.60,
) -> torch.Tensor:
    """
    Binary focal loss.
    alpha – weight on the positive class (higher → more focus on positives)
    gamma – focusing exponent (down-weights easy negatives)
    """
    p       = logits.squeeze(1)
    t       = targets.squeeze(1)
    bce     = nn.functional.binary_cross_entropy(p, t, reduction="none")
    pt      = torch.where(t == 1, p, 1 - p)
    alpha_t = torch.where(t == 1,
                          torch.full_like(pt, alpha),
                          torch.full_like(pt, 1 - alpha))
    return (alpha_t * (1 - pt) ** gamma * bce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _batched_predict(
    model: nn.Module,
    X: np.ndarray,
    device: str,
    batch_size: int = 4096,
) -> np.ndarray:
    """Run inference in mini-batches to avoid OOM on large test sets."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i: i + batch_size], dtype=torch.float32).to(device)
            preds.append(model(xb).cpu().numpy().squeeze(axis=1))
    return np.concatenate(preds)


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def run_proposed_dnn_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    val_df:  pd.DataFrame,
    test_df: pd.DataFrame,
    device:  str = "cpu",
    epochs:  int = 60,
    batch_size: int = 2048,
):
    """
    Train the DNN+RF ensemble with precision-oriented per-region calibration.

    Parameters
    ----------
    X_train, y_train   training arrays (already z-scored)
    X_val, y_val       validation arrays
    X_test, y_test     test arrays
    val_df, test_df    feature DataFrames (must have 'oblast' and 'target' columns)
    device             'cpu', 'cuda', or 'mps'
    epochs             maximum DNN training epochs (early stopping may end sooner)
    batch_size         mini-batch size for DNN training

    Returns
    -------
    y_true      ground-truth labels (test set, NaN target rows dropped)
    y_pred      binary predictions after per-region threshold calibration
    y_prob      ensemble probabilities aligned with y_true (post isotonic recalibration)
    test_df     test DataFrame with 'ens_prob' column appended
    thresholds  dict mapping oblast name → calibrated threshold;
                key '_global' holds the global fallback threshold
    """
    # ── Step 1: Train DNN with focal loss ────────────────────────────────────
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    dnn = DeepNN(input_dim=X_train.shape[1]).to(device)
    optimizer   = optim.Adam(dnn.parameters(), lr=1e-3, weight_decay=1e-4)
    total_steps = epochs * len(loader)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val_auc, patience, best_state = -1.0, 0, None
    for epoch in range(epochs):
        dnn.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            focal_loss(dnn(xb), yb).backward()
            nn.utils.clip_grad_norm_(dnn.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        p_val_ep = _batched_predict(dnn, X_val, device)
        val_auc  = roc_auc_score(y_val, p_val_ep)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in dnn.state_dict().items()}
            patience     = 0
        else:
            patience += 1
            if patience >= 15:
                print(f"    Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"    DNN epoch {epoch + 1}/{epochs}  val_AUC={val_auc:.5f}")

    print(f"    DNN done — best val_AUC={best_val_auc:.5f}")
    dnn.load_state_dict(best_state)
    p_nn_val  = _batched_predict(dnn, X_val,  device)
    p_nn_test = _batched_predict(dnn, X_test, device)

    # ── Step 2: Train Random Forest ──────────────────────────────────────────
    print("    Training Random Forest (300 trees)...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    p_rf_val  = rf.predict_proba(X_val)[:, 1]
    p_rf_test = rf.predict_proba(X_test)[:, 1]

    # ── Step 3: Search optimal blend weight on validation F1 ─────────────────
    print("    Searching blend weight α_NN ...")
    best_alpha, best_val_f1 = 0.5, -1.0
    for alpha in np.arange(0.05, 1.0, 0.05):
        p_blend = alpha * p_nn_val + (1 - alpha) * p_rf_val
        t = calibrate_threshold(y_val, p_blend, target_prec=0.0)
        if t is None:
            continue
        f1 = f1_score(y_val, (p_blend >= t).astype(int), zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_alpha  = alpha

    print(f"    Best α_NN={best_alpha:.2f}  α_RF={1 - best_alpha:.2f}  "
          f"val_F1(no floor)={best_val_f1:.4f}")

    p_ens_val  = best_alpha * p_nn_val  + (1 - best_alpha) * p_rf_val
    p_ens_test = best_alpha * p_nn_test + (1 - best_alpha) * p_rf_test

    # ── Step 4: Isotonic recalibration ───────────────────────────────────────
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_ens_val, y_val)
    p_ens_val  = iso.transform(p_ens_val)
    p_ens_test = iso.transform(p_ens_test)

    # ── Step 5: Per-region precision-oriented threshold calibration ───────────
    val_df_clean  = val_df.dropna(subset=["target"]).copy().reset_index(drop=True)
    test_df_clean = test_df.dropna(subset=["target"]).copy().reset_index(drop=True)
    val_df_clean["ens_prob"]  = p_ens_val
    test_df_clean["ens_prob"] = p_ens_test

    global_threshold = calibrate_threshold(
        y_val, p_ens_val, target_prec=TARGET_PRECISION
    )
    if global_threshold is None:
        global_threshold = calibrate_threshold(y_val, p_ens_val, target_prec=0.0)
    print(f"    Global threshold = {global_threshold:.4f}  "
          f"(precision floor = {TARGET_PRECISION:.0%})")

    thresholds = {"_global": global_threshold}
    for oblast in test_df_clean["oblast"].unique():
        val_mask = val_df_clean["oblast"] == oblast
        if val_mask.sum() < 30:
            thresholds[oblast] = global_threshold
            continue
        y_val_r = val_df_clean.loc[val_mask, "target"].astype(int).values
        p_val_r = val_df_clean.loc[val_mask, "ens_prob"].values
        t = calibrate_threshold(y_val_r, p_val_r, target_prec=TARGET_PRECISION)
        thresholds[oblast] = t if t is not None else global_threshold

    n_fallback = sum(
        1 for k, v in thresholds.items()
        if k != "_global" and v == global_threshold
    )
    print(f"    Per-region thresholds: "
          f"{len(thresholds) - 1 - n_fallback} region-specific, "
          f"{n_fallback} fell back to global")

    # ── Step 6: Apply per-region thresholds ──────────────────────────────────
    y_pred = np.zeros(len(test_df_clean), dtype=int)
    for oblast in test_df_clean["oblast"].unique():
        mask = (test_df_clean["oblast"] == oblast).values
        t    = thresholds.get(oblast, global_threshold)
        y_pred[mask] = (test_df_clean.loc[mask, "ens_prob"].values >= t).astype(int)

    y_true = test_df_clean["target"].astype(int).values
    y_prob = test_df_clean["ens_prob"].values
    return y_true, y_pred, y_prob, test_df_clean, thresholds
