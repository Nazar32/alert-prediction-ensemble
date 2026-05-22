#!/usr/bin/env python3
# ===== DEPRECATED =====
# This is the original DNN+RF reconstruction prior to the per-region
# calibration fix. Superseded by src/proposed_model_dnn_rf_fixed.py.
"""
proposed_model_dnn_rf.py
========================
Reconstruction of the original hybrid ensemble as described in the paper:

  Deep Neural Network (4 fully-connected layers, BatchNorm, Dropout, focal loss)
  + Random Forest (300 trees, balanced class weights)

Key design choices matching the paper:
  - DNN: Linear(→256) → BN → Dropout(0.2) → Linear(→128) → BN → Dropout(0.2)
         → Linear(→64) → Dropout(0.1) → Linear(→1) → Sigmoid
  - Focal loss (γ=2.0, α_pos=0.60) for class-imbalance awareness
  - Adam optimiser with cosine LR schedule and early stopping on val AUC
  - 2-way ensemble blend weight α searched on validation F1 (no precision floor)
  - Isotonic probability recalibration to correct focal-loss compression
  - Per-region threshold calibration: finds threshold maximising F1 subject to
    precision ≥ TARGET_PRECISION (75%); falls back to global threshold for
    regions with insufficient validation samples

This is distinct from src/proposed_model.py which uses an
FT-Transformer + RF + CatBoost architecture with TARGET_PRECISION=0.0.

Run from the project root:
    python deprecated/proposed_model_dnn_rf.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.isotonic import IsotonicRegression

# Reuse shared data-loading / feature-engineering / evaluation utilities
from proposed_model import (
    load_data, engineer_features, temporal_split, prepare_arrays,
    compute_metrics, mcnemar_test,
    SEED, RESULTS_DIR,
)

DATA_PATH = "data/official_data_uk.csv"

warnings.filterwarnings("ignore")

TARGET_PRECISION = 0.75   # precision floor for per-region threshold calibration

np.random.seed(SEED)
torch.manual_seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Architecture: 4-layer DNN with BatchNorm and Dropout
# Paper: "four fully connected layers with batch normalisation and dropout"
# ──────────────────────────────────────────────────────────────────────────────

class DeepNN(nn.Module):
    """
    256 → BN → Dropout(0.2) →
    128 → BN → Dropout(0.2) →
     64 →      Dropout(0.1) →
      1 → Sigmoid
    """

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
    alpha  – weight on the positive class (higher → focus more on positives)
    gamma  – focusing exponent (down-weights easy negatives)
    """
    p      = logits.squeeze(1)
    t      = targets.squeeze(1)
    bce    = nn.functional.binary_cross_entropy(p, t, reduction="none")
    pt     = torch.where(t == 1, p, 1 - p)
    alpha_t = torch.where(t == 1,
                          torch.full_like(pt, alpha),
                          torch.full_like(pt, 1 - alpha))
    return (alpha_t * (1 - pt) ** gamma * bce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Precision-oriented threshold calibration
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_threshold(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
    target_prec: float = TARGET_PRECISION,
    min_predictions: int = 5,
) -> float | None:
    """
    Search 199 candidate thresholds in [0.01, 0.99].
    Return the one that maximises F1 subject to precision >= target_prec.
    Returns None when no threshold satisfies the constraint (callers should
    fall back to a global threshold).
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


# ──────────────────────────────────────────────────────────────────────────────
# Batched inference (avoids OOM on large test sets)
# ──────────────────────────────────────────────────────────────────────────────

def _batched_predict(
    model: nn.Module,
    X: np.ndarray,
    device: str,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
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
    Train DNN + RF ensemble with precision-oriented per-region calibration.

    Returns
    -------
    y_true      ground truth labels (test set)
    y_pred      binary predictions after per-region threshold calibration
    p_ens_test  ensemble probabilities on test set (post isotonic calibration)
    test_df     test DataFrame enriched with 'ens_prob' column
    thresholds  dict: oblast name → calibrated threshold (key '_global' = fallback)
    """
    # ── DNN with focal loss ──────────────────────────────────────────────────
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    dnn = DeepNN(input_dim=X_train.shape[1]).to(device)
    optimizer  = optim.Adam(dnn.parameters(), lr=1e-3, weight_decay=1e-4)
    total_steps = epochs * len(loader)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

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

    # ── Random Forest ────────────────────────────────────────────────────────
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

    # ── 2-way blend weight: search on validation F1 (no precision floor) ────
    print("    Searching 2-way blend weight α_NN ...")
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

    # ── Isotonic recalibration ───────────────────────────────────────────────
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_ens_val, y_val)
    p_ens_val  = iso.transform(p_ens_val)
    p_ens_test = iso.transform(p_ens_test)

    # ── Per-region precision-oriented threshold calibration ──────────────────
    # Attach ensemble probabilities to the clean (non-NaN target) val/test frames
    val_df_clean  = val_df.dropna(subset=["target"]).copy().reset_index(drop=True)
    test_df_clean = test_df.dropna(subset=["target"]).copy().reset_index(drop=True)
    val_df_clean["ens_prob"]  = p_ens_val
    test_df_clean["ens_prob"] = p_ens_test

    # Global fallback threshold (with precision floor, then without if needed)
    global_threshold = calibrate_threshold(y_val, p_ens_val, target_prec=TARGET_PRECISION)
    if global_threshold is None:
        global_threshold = calibrate_threshold(y_val, p_ens_val, target_prec=0.0)
    print(f"    Global threshold = {global_threshold:.4f}  "
          f"(precision floor = {TARGET_PRECISION:.0%})")

    thresholds = {"_global": global_threshold}
    for oblast in test_df_clean["oblast"].unique():
        val_mask = val_df_clean["oblast"] == oblast
        if val_mask.sum() < 30:          # too few validation samples → use global
            thresholds[oblast] = global_threshold
            continue
        y_val_r = val_df_clean.loc[val_mask, "target"].astype(int).values
        p_val_r = val_df_clean.loc[val_mask, "ens_prob"].values
        t = calibrate_threshold(y_val_r, p_val_r, target_prec=TARGET_PRECISION)
        thresholds[oblast] = t if t is not None else global_threshold

    n_global_fallback = sum(1 for k, v in thresholds.items()
                            if k != "_global" and v == global_threshold)
    print(f"    Per-region thresholds calibrated  "
          f"({len(thresholds) - 1 - n_global_fallback} region-specific, "
          f"{n_global_fallback} fell back to global)")

    # Vectorised threshold application
    y_pred = np.zeros(len(test_df_clean), dtype=int)
    for oblast in test_df_clean["oblast"].unique():
        mask = (test_df_clean["oblast"] == oblast).values
        t    = thresholds.get(oblast, global_threshold)
        y_pred[mask] = (test_df_clean.loc[mask, "ens_prob"].values >= t).astype(int)

    y_true = test_df_clean["target"].astype(int).values
    return y_true, y_pred, p_ens_test, test_df_clean, thresholds


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}")
    print(f"TARGET_PRECISION = {TARGET_PRECISION:.0%}\n")

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

    val_f_clean  = val_f.dropna(subset=["target"]).reset_index(drop=True)
    test_f_clean = test_f.dropna(subset=["target"]).reset_index(drop=True)

    print(f"Feature matrix: {X_train.shape[1]} features\n")
    print("Training DNN+RF Proposed Ensemble...")

    y_true, y_pred, y_prob, test_df_out, region_thresholds = run_proposed_dnn_rf(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        val_f_clean, test_f_clean,
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
    oblasts = test_df_out["oblast"].unique()
    region_precisions = []
    for o in oblasts:
        mask = (test_df_out["oblast"] == o).values
        yp   = y_pred[mask]
        yt   = y_true[mask]
        if yp.sum() > 0:
            region_precisions.append(precision_score(yt, yp, zero_division=0))
    if region_precisions:
        print(f"\n  regions meeting ≥{TARGET_PRECISION:.0%} precision: "
              f"{sum(p >= TARGET_PRECISION for p in region_precisions)}"
              f"/{len(region_precisions)}")
        print(f"  mean region precision: {np.mean(region_precisions):.4f}")

    # Save
    out_path = os.path.join(RESULTS_DIR, "proposed_dnn_rf_results.csv")
    pd.DataFrame([{"model": "Proposed_DNN_RF", **metrics}]).to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
