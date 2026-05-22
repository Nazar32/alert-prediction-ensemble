#!/usr/bin/env python3
"""
proposed_model.py
=================
Standalone implementation of the proposed hybrid ensemble:

  FT-Transformer (Feature-Tokenizer Transformer, 3-layer, focal loss)
  + Random Forest (300 trees, balanced class weights)

The FT-Transformer tokenises each feature into a learned embedding and
processes all tokens with multi-head self-attention, enabling complex
feature-interaction learning.  Ensembled with RF via α-weighted blend.

Key design choices:
  - Base temporal/lag + cross-region + neighbour spillover + oblast one-hots;
    StandardScaler on numeric columns only (one-hots unscaled)
  - FT-Transformer: d_token=128, 8 heads, 3 layers, GELU, cosine LR
  - Focal loss (α=0.60, γ=2.0) for class-imbalance awareness
  - Isotonic probability recalibration on blended scores
  - Per-region F1-maximising threshold calibration (global fallback)

This module also exports the shared data-loading, feature-engineering, and
evaluation utilities used by baseline_comparison.py.

Run from the project root:
    python experiments/proposed_model.py
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
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2 as chi2_dist
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostClassifier
import math

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

TARGET_PRECISION = 0.0   # 0.0 = no precision floor; calibrate to maximise F1
DATA_PATH   = "data/official_data_uk.csv"
RESULTS_DIR = "results"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Load and expand alert events into hourly active/inactive indicators.

    Each row in the CSV is an alert *event* with a start and end timestamp.
    Correct preprocessing marks every hour that overlaps with an active alert
    as alert_occurred=1, not just the start hour.  Counting only the start
    hour (floor(started_at)) causes severe under-counting because many alerts
    span multiple hours, collapsing multi-hour autocorrelation and making the
    task much harder than it actually is.
    """
    df = pd.read_csv(path, parse_dates=["started_at", "finished_at"])
    oblast_df = df[df["level"] == "oblast"].copy()

    rows = []
    for _, row in oblast_df.iterrows():
        start = row["started_at"].floor("h")
        end   = row["finished_at"].floor("h")
        for h in pd.date_range(start, end, freq="h"):
            rows.append({"oblast": row["oblast"], "hour": h})

    expanded = pd.DataFrame(rows)
    # Deduplicate: multiple overlapping alerts in the same region-hour → still 1
    hourly = (
        expanded.groupby(["oblast", "hour"])
        .size()
        .reset_index(name="n")
        .assign(alert_occurred=1)[["oblast", "hour", "alert_occurred"]]
    )

    all_oblasts = oblast_df["oblast"].unique()
    full_range  = pd.date_range(hourly["hour"].min(), hourly["hour"].max(), freq="h")
    idx = pd.MultiIndex.from_product(
        [all_oblasts, full_range], names=["oblast", "hour"]
    )
    full_df = pd.DataFrame(index=idx).reset_index()
    full_df = full_df.merge(
        hourly[["oblast", "hour", "alert_occurred"]],
        on=["oblast", "hour"],
        how="left",
    )
    full_df["alert_occurred"] = full_df["alert_occurred"].fillna(0).astype(int)
    full_df = full_df.sort_values(["oblast", "hour"]).reset_index(drop=True)
    return full_df


# ──────────────────────────────────────────────────────────────────────────────
# Neighbour graph (shared land borders, administrative geography — simplified)
# ──────────────────────────────────────────────────────────────────────────────

OBLAST_NEIGHBORS = {
    "Івано-Франківська область": [
        "Львівська область", "Тернопільська область", "Закарпатська область",
        "Чернівецька область",
    ],
    "Волинська область": ["Рівненська область", "Львівська область"],
    "Вінницька область": [
        "Житомирська область", "Київська область", "Черкаська область",
        "Хмельницька область", "Тернопільська область", "Кіровоградська область",
        "Одеська область",
    ],
    "Дніпропетровська область": [
        "Запорізька область", "Донецька область", "Харківська область",
        "Полтавська область", "Кіровоградська область", "Миколаївська область",
    ],
    "Донецька область": [
        "Луганська область", "Дніпропетровська область", "Запорізька область",
    ],
    "Житомирська область": [
        "Волинська область", "Рівненська область", "Київська область",
        "Вінницька область", "Чернігівська область",
    ],
    "Закарпатська область": [
        "Львівська область", "Івано-Франківська область",
    ],
    "Запорізька область": [
        "Дніпропетровська область", "Донецька область", "Херсонська область",
    ],
    "Київська область": [
        "м. Київ", "Житомирська область", "Вінницька область", "Черкаська область",
        "Полтавська область", "Чернігівська область",
    ],
    "Кіровоградська область": [
        "Вінницька область", "Черкаська область", "Полтавська область",
        "Дніпропетровська область", "Миколаївська область", "Одеська область",
    ],
    "Луганська область": ["Донецька область", "Харківська область"],
    "Львівська область": [
        "Волинська область", "Рівненська область", "Тернопільська область",
        "Івано-Франківська область", "Закарпатська область",
    ],
    "Миколаївська область": [
        "Одеська область", "Кіровоградська область", "Дніпропетровська область",
        "Херсонська область",
    ],
    "Одеська область": [
        "Миколаївська область", "Кіровоградська область", "Вінницька область",
    ],
    "Полтавська область": [
        "Київська область", "Черкаська область", "Кіровоградська область",
        "Харківська область", "Сумська область",
    ],
    "Рівненська область": [
        "Волинська область", "Житомирська область", "Тернопільська область",
        "Хмельницька область",
    ],
    "Сумська область": ["Чернігівська область", "Полтавська область", "Харківська область"],
    "Тернопільська область": [
        "Рівненська область", "Львівська область", "Івано-Франківська область",
        "Хмельницька область", "Вінницька область",
    ],
    "Харківська область": [
        "Сумська область", "Полтавська область", "Дніпропетровська область",
        "Донецька область", "Луганська область",
    ],
    "Херсонська область": [
        "Миколаївська область", "Дніпропетровська область", "Запорізька область",
    ],
    "Хмельницька область": [
        "Рівненська область", "Тернопільська область", "Вінницька область",
        "Чернівецька область",
    ],
    "Черкаська область": [
        "Київська область", "Полтавська область", "Кіровоградська область",
        "Вінницька область",
    ],
    "Чернівецька область": [
        "Івано-Франківська область", "Тернопільська область", "Хмельницька область",
    ],
    "Чернігівська область": [
        "Київська область", "Сумська область", "Житомирська область",
    ],
    "м. Київ": ["Київська область"],
}


def _add_neighbor_spillover_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (hour, oblast), sum alert_occurred over *adjacent* oblasts only
    (same clock hour).  Adds lag and short rolling views of that signal.
    """
    wide = (
        df.pivot_table(index="hour", columns="oblast", values="alert_occurred",
                       aggfunc="max")
        .fillna(0)
    )
    cols = list(wide.columns)
    neighbor_sum = pd.DataFrame(0.0, index=wide.index, columns=cols, dtype=np.float64)
    neighbor_deg = pd.DataFrame(1.0, index=wide.index, columns=cols, dtype=np.float64)
    for o in cols:
        nbrs = [n for n in OBLAST_NEIGHBORS.get(o, []) if n in wide.columns]
        if nbrs:
            neighbor_sum[o] = wide[nbrs].sum(axis=1).values
            neighbor_deg[o] = float(len(nbrs))
    stacked_sum = neighbor_sum.stack().reset_index()
    stacked_sum.columns = ["hour", "oblast", "neighbor_alerts_now"]
    stacked_deg = neighbor_deg.stack().reset_index()
    stacked_deg.columns = ["hour", "oblast", "neighbor_graph_degree"]
    spill = stacked_sum.merge(stacked_deg, on=["hour", "oblast"])
    spill["neighbor_rate_now"] = (
        spill["neighbor_alerts_now"] / spill["neighbor_graph_degree"].clip(lower=1.0)
    )
    df = df.merge(
        spill[["hour", "oblast", "neighbor_alerts_now", "neighbor_rate_now"]],
        on=["hour", "oblast"],
        how="left",
    )
    df["neighbor_alerts_now"] = df["neighbor_alerts_now"].fillna(0).astype(np.float32)
    df["neighbor_rate_now"] = df["neighbor_rate_now"].fillna(0).astype(np.float32)

    df = df.sort_values(["oblast", "hour"]).reset_index(drop=True)
    df["neighbor_alerts_lag1h"] = (
        df.groupby("oblast")["neighbor_alerts_now"].shift(1).fillna(0).astype(np.float32)
    )
    df["neighbor_alerts_roll3h"] = (
        df.groupby("oblast")["neighbor_alerts_now"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
        .fillna(0)
        .astype(np.float32)
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering  (identical feature set shared with all baseline models)
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Temporal ---
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["month"]       = df["hour"].dt.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"]    = df["hour_of_day"].isin(
        list(range(0, 6)) + list(range(22, 24))
    ).astype(int)

    # --- Cyclical encoding ---
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["day_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Current alert status (lag_0h) ---
    # Valid for 1-hour-ahead prediction: at time t we know the current alert
    # status and use it to predict t+1.  This is the single most informative
    # feature due to the multi-hour duration of alerts.
    df["lag_0h"] = df["alert_occurred"].astype(float)

    # --- Lag features (24 h lookback) ---
    for i in range(1, 25):
        df[f"lag_{i}h"] = (
            df.groupby("oblast")["alert_occurred"].shift(i).fillna(0)
        )

    # --- Rolling statistics over multiple windows (including current hour) ---
    # No shift(1): at prediction time t all data up to and including t is known.
    for w in [3, 6, 12, 24, 48, 72]:
        df[f"rate_{w}h"] = (
            df.groupby("oblast")["alert_occurred"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
            .fillna(0)
        )
        df[f"count_{w}h"] = (
            df.groupby("oblast")["alert_occurred"]
            .transform(lambda x: x.rolling(w, min_periods=1).sum())
            .fillna(0)
        )
        df[f"std_{w}h"] = (
            df.groupby("oblast")["alert_occurred"]
            .transform(lambda x: x.rolling(w, min_periods=1).std())
            .fillna(0)
        )

    # --- Momentum features ---
    df["momentum_3_12"]  = df["rate_3h"]  - df["rate_12h"]
    df["momentum_6_24"]  = df["rate_6h"]  - df["rate_24h"]
    df["momentum_12_72"] = df["rate_12h"] - df["rate_72h"]

    # --- Volatility / relative rate ---
    global_rate = df.groupby("oblast")["alert_occurred"].transform("mean")
    df["volatility_24h"] = (
        df.groupby("oblast")["alert_occurred"]
        .transform(lambda x: x.rolling(24, min_periods=1).std())
        .fillna(0)
    )
    df["rel_rate_24h"] = df["rate_24h"] / (global_rate + 1e-6)

    # --- Alert state-transition features ---
    # These capture alert momentum far more directly than binary lag indicators.
    # alert_duration:        consecutive hours in current alert (0 when no alert)
    # hours_since_last_alert: hours elapsed since last alert ended (0 when in alert)
    # Both use the "run-length within group" trick: group by change-points, then
    # count position within each constant run.
    def _run_length(s):
        """Position (1-indexed) of each element within its constant run."""
        run_id = (s != s.shift()).cumsum()
        return s.groupby(run_id).cumcount() + 1

    df["alert_duration"] = (
        df.groupby("oblast")["alert_occurred"]
        .transform(lambda s: s * _run_length(s))
        .fillna(0)
    )
    inv = df["alert_occurred"].map({1: 0, 0: 1}).fillna(0)
    df["_inv_alert"] = inv
    df["hours_since_last_alert"] = (
        df.groupby("oblast")["_inv_alert"]
        .transform(lambda s: s * _run_length(s))
        .fillna(0)
    )
    df.drop(columns=["_inv_alert"], inplace=True)

    # --- Cross-region spatial features ---
    # At each hour, count how many OTHER regions currently have an alert.
    # This captures spatial propagation: if many regions are under alert,
    # this region is more likely to be next.  CatBoost/LightGBM cannot
    # learn this from per-region features alone.
    alerts_per_hour = df.groupby("hour")["alert_occurred"].transform("sum")
    df["other_regions_alert"] = alerts_per_hour - df["alert_occurred"]
    n_regions = df["oblast"].nunique()
    df["national_alert_rate"] = df["other_regions_alert"] / max(n_regions - 1, 1)

    # Rolling national alert momentum (was the country-wide alert level rising?)
    df["national_rate_3h"] = (
        df.groupby("oblast")["national_alert_rate"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
        .fillna(0)
    )
    df["national_rate_12h"] = (
        df.groupby("oblast")["national_alert_rate"]
        .transform(lambda x: x.rolling(12, min_periods=1).mean())
        .fillna(0)
    )
    df["national_momentum"] = df["national_rate_3h"] - df["national_rate_12h"]

    # --- Adjacency-weighted neighbour spillover (same hour, bordering oblasts) ---
    df = _add_neighbor_spillover_features(df)

    # --- Oblast identity (full frame before split → stable dummy columns) ---
    oblast_dummies = pd.get_dummies(df["oblast"], prefix="oblast", dtype=np.float32)
    df = pd.concat([df, oblast_dummies], axis=1)

    # --- Target: next-hour alert ---
    df["target"] = df.groupby("oblast")["alert_occurred"].shift(-1)

    return df


BASE_FEATURE_COLS = (
    ["lag_0h"]
    + ["hour_of_day", "day_of_week", "month", "is_weekend", "is_night",
       "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"]
    + [f"lag_{i}h" for i in range(1, 25)]
    + [f"rate_{w}h"  for w in [3, 6, 12, 24, 48, 72]]
    + [f"count_{w}h" for w in [3, 6, 12, 24, 48, 72]]
    + [f"std_{w}h"   for w in [3, 6, 12, 24, 48, 72]]
    + ["momentum_3_12", "momentum_6_24", "momentum_12_72",
       "volatility_24h", "rel_rate_24h"]
    + ["alert_duration", "hours_since_last_alert"]
    + ["other_regions_alert", "national_alert_rate",
       "national_rate_3h", "national_rate_12h", "national_momentum"]
)

NEIGHBOR_FEATURE_COLS = (
    "neighbor_alerts_now",
    "neighbor_rate_now",
    "neighbor_alerts_lag1h",
    "neighbor_alerts_roll3h",
)

# Continuous / count features — StandardScaler; oblast one-hots are passed through.
NUMERIC_FEATURE_COLS = tuple(BASE_FEATURE_COLS) + NEIGHBOR_FEATURE_COLS


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric (scaled) columns + sorted oblast_* dummies present in ``df``."""
    oblast_cols = sorted(c for c in df.columns if c.startswith("oblast_"))
    return list(NUMERIC_FEATURE_COLS) + oblast_cols


# ──────────────────────────────────────────────────────────────────────────────
# Temporal split and array preparation
# ──────────────────────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame):
    """Strict 70/15/15 split on time axis — no shuffling, no data leakage."""
    hours   = df["hour"].sort_values().unique()
    n       = len(hours)
    train_end = hours[int(n * 0.70)]
    val_end   = hours[int(n * 0.85)]
    train = df[df["hour"] <  train_end]
    val   = df[(df["hour"] >= train_end) & (df["hour"] < val_end)]
    test  = df[df["hour"] >= val_end]
    return train, val, test


def prepare_arrays(df_feat, scaler=None, fit_scaler=False):
    df_clean = df_feat.dropna(subset=["target"])
    cols = get_feature_columns(df_feat)
    n_num = len(NUMERIC_FEATURE_COLS)
    num_cols = list(NUMERIC_FEATURE_COLS)
    oblast_cols = cols[n_num:]
    X_num = df_clean[num_cols].values.astype(np.float32)
    y = df_clean["target"].astype(int).values
    if fit_scaler:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num).astype(np.float32)
    elif scaler is not None:
        X_num = scaler.transform(X_num).astype(np.float32)
    if oblast_cols:
        X_obl = df_clean[oblast_cols].values.astype(np.float32)
        X = np.concatenate([X_num, X_obl], axis=1)
    else:
        X = X_num
    return X, y, scaler


# ──────────────────────────────────────────────────────────────────────────────
# Shared evaluation utilities
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_threshold(y_val, y_prob_val, target_prec=TARGET_PRECISION,
                        min_predictions=5):
    """
    Return the threshold in [0.01, 0.99] that maximises F1-score.
    When target_prec > 0, the threshold must also satisfy precision >= target_prec.

    Uses a 199-point grid for fine resolution.
    Returns None if no threshold satisfies the constraint (only possible when
    target_prec > 0); callers should fall back to a global threshold.
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
    return best_t  # None only when target_prec > 0 and no threshold qualifies


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    m = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "accuracy":  accuracy_score(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            m["roc_auc"] = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    m["TP"] = int(cm[1, 1])
    m["FP"] = int(cm[0, 1])
    m["TN"] = int(cm[0, 0])
    m["FN"] = int(cm[1, 0])
    return m


def mcnemar_test(y_true, y_pred_proposed, y_pred_baseline):
    """McNemar's test (two-sided) with continuity correction."""
    b = int(np.sum((y_pred_proposed == y_true) & (y_pred_baseline != y_true)))
    c = int(np.sum((y_pred_proposed != y_true) & (y_pred_baseline == y_true)))
    if (b + c) == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p    = chi2_dist.sf(stat, df=1)
    return float(stat), float(p)


# ──────────────────────────────────────────────────────────────────────────────
# Proposed model: FT-Transformer + RF hybrid ensemble
# ──────────────────────────────────────────────────────────────────────────────

class NumericalTokenizer(nn.Module):
    """Project each scalar feature into a d_token-dimensional embedding."""
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (B, n_features)  →  (B, n_features, d_token)
        return x.unsqueeze(-1) * self.weight[None] + self.bias[None]


class FTTransformer(nn.Module):
    """
    Feature-Tokenizer Transformer for tabular classification.

    Each numerical feature is projected to a d_token-dimensional embedding via
    a learned linear map.  A learnable [CLS] token is prepended to the
    sequence, and a standard Transformer encoder processes the tokens.
    The [CLS] output is passed through a classification head.

    Reference: Gorishniy et al., "Revisiting Deep Learning Models for
    Tabular Data", NeurIPS 2021.
    """
    def __init__(self, n_features, d_token=192, n_heads=8, n_layers=3,
                 d_ffn=512, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.tokenizer = NumericalTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln   = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        tokens = self.tokenizer(x)                                       # (B, F, d)
        cls    = self.cls_token.expand(x.size(0), -1, -1)                # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)                         # (B, F+1, d)
        tokens = self.transformer(tokens)                                # (B, F+1, d)
        cls_out = self.ln(tokens[:, 0])                                  # (B, d)
        return self.head(cls_out)                                        # (B, 1)


def _batched_predict(model, X_np, device, batch_size=4096):
    """Run model.forward on X_np in chunks to avoid OOM on large sets."""
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32).to(device)
            parts.append(model(xb).cpu().numpy().ravel())
    return np.concatenate(parts)


def focal_loss(y_pred, y_true, alpha=0.60, gamma=2.0, label_smoothing=0.05):
    if label_smoothing > 0:
        y_true = y_true * (1 - label_smoothing) + 0.5 * label_smoothing
    y_pred = y_pred.clamp(1e-7, 1.0 - 1e-7)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    a_t = y_true * alpha  + (1 - y_true) * (1 - alpha)
    return -(a_t * (1 - p_t) ** gamma * torch.log(p_t)).mean()


def run_proposed(
    X_train, y_train, X_val, y_val, X_test, y_test,
    df_feat_val, df_feat_test,
    epochs=80, batch_size=1024, device="cpu",
):
    """
    FT-Transformer + RF hybrid ensemble.

    The FT-Transformer (Feature-Tokenizer Transformer) processes each feature
    through a learned embedding and multi-head self-attention, enabling complex
    feature-interaction learning that standard MLPs cannot capture.  It is
    ensembled with a Random Forest via validation-optimised blending weight α.

    Returns:
        y_true     – ground truth labels (test set)
        y_pred     – binary predictions after per-region threshold calibration
        p_ens_test – ensemble probabilities on test set
        test_df    – test DataFrame enriched with 'ens_prob' column
        thresholds – dict mapping oblast name → calibrated threshold
    """
    # ── FT-Transformer with focal loss ───────────────────────────────────────
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    ft_model = FTTransformer(
        n_features=X_train.shape[1], d_token=128, n_heads=8, n_layers=3,
        d_ffn=256, dropout=0.1, attn_dropout=0.1,
    ).to(device)

    optimizer = optim.AdamW(ft_model.parameters(), lr=3e-4, weight_decay=1e-4)
    total_steps = epochs * len(loader)
    warmup_steps = min(len(loader) * 5, total_steps // 4)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_auc, patience, best_state = -1.0, 0, None
    global_step = 0
    for epoch in range(epochs):
        ft_model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            focal_loss(ft_model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

        # Early-stop on val AUC (more stable than focal loss for threshold search)
        p_val_batch = _batched_predict(ft_model, X_val, device)
        val_auc = roc_auc_score(y_val, p_val_batch)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in ft_model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break

        if (epoch + 1) % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"    FT-T epoch {epoch+1}/{epochs}  val_AUC={val_auc:.5f}  lr={lr_now:.6f}")

    print(f"    FT-Transformer training done at epoch {epoch+1}, best_val_AUC={best_val_auc:.5f}")
    ft_model.load_state_dict(best_state)
    p_ft_val  = _batched_predict(ft_model, X_val,  device)
    p_ft_test = _batched_predict(ft_model, X_test, device)

    # Report standalone FT-T quality
    t_ft = calibrate_threshold(y_val, p_ft_val)
    if t_ft is not None:
        f1_ft = f1_score(y_val, (p_ft_val >= t_ft).astype(int), zero_division=0)
        auc_ft = roc_auc_score(y_val, p_ft_val)
        print(f"    FT-T standalone:  val_F1={f1_ft:.4f}  val_AUC={auc_ft:.4f}")

    # ── Random Forest ────────────────────────────────────────────────────────
    print("    Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    p_rf_val  = rf.predict_proba(X_val)[:, 1]
    p_rf_test = rf.predict_proba(X_test)[:, 1]

    # ── CatBoost ─────────────────────────────────────────────────────────────
    print("    Training CatBoost...")
    cat = CatBoostClassifier(
        iterations=300, depth=8, learning_rate=0.05,
        auto_class_weights="Balanced", random_seed=SEED, verbose=0,
    )
    cat.fit(X_train, y_train)
    p_cat_val  = cat.predict_proba(X_val)[:, 1]
    p_cat_test = cat.predict_proba(X_test)[:, 1]

    # ── Optimise 3-way ensemble weights ──────────────────────────────────────
    print("    Optimising 3-way weights (FT-T, RF, CatBoost)...")
    best_w, best_val_f1 = (0.4, 0.3, 0.3), -1.0
    for w_ft in np.arange(0.1, 0.8, 0.1):
        for w_rf in np.arange(0.0, 0.6, 0.1):
            w_cat = round(1.0 - w_ft - w_rf, 2)
            if w_cat < 0.05 or w_cat > 0.8:
                continue
            p_blend = w_ft * p_ft_val + w_rf * p_rf_val + w_cat * p_cat_val
            t = calibrate_threshold(y_val, p_blend)
            if t is None:
                continue
            f1 = f1_score(y_val, (p_blend >= t).astype(int), zero_division=0)
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_w = (w_ft, w_rf, w_cat)

    w_ft, w_rf, w_cat = best_w
    print(f"    Best → FT-T={w_ft:.2f}  RF={w_rf:.2f}  CAT={w_cat:.2f}  "
          f"val_F1={best_val_f1:.4f}")

    p_ens_val  = w_ft * p_ft_val  + w_rf * p_rf_val  + w_cat * p_cat_val
    p_ens_test = w_ft * p_ft_test + w_rf * p_rf_test + w_cat * p_cat_test

    # ── Isotonic calibration ─────────────────────────────────────────────────
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_ens_val, y_val)
    p_ens_val  = iso.transform(p_ens_val)
    p_ens_test = iso.transform(p_ens_test)

    # ── Global F1-maximising threshold on validation set ────────────────────
    # Single global threshold (same approach as all baselines) to ensure a
    # fair comparison.  Per-region calibration fragmented the threshold search
    # onto small subsets and hurt overall F1 when no precision floor is imposed.
    test_df = df_feat_test.dropna(subset=["target"]).copy().reset_index(drop=True)
    test_df["ens_prob"] = p_ens_test

    global_threshold = calibrate_threshold(y_val, p_ens_val)
    if global_threshold is None:
        global_threshold = 0.5

    print(f"    Global threshold = {global_threshold:.4f}")

    y_pred = (p_ens_test >= global_threshold).astype(int)
    y_true = test_df["target"].astype(int).values
    thresholds = {"_global": global_threshold}

    return y_true, y_pred, p_ens_test, test_df, thresholds


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────────────

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

    print("Engineering features (this may take ~2 minutes)...")
    feat = engineer_features(raw)
    train_f, val_f, test_f = temporal_split(feat)
    print(f"  Train {len(train_f):,} | Val {len(val_f):,} | Test {len(test_f):,}\n")

    X_train, y_train, scaler = prepare_arrays(train_f, fit_scaler=True)
    X_val,   y_val,   _      = prepare_arrays(val_f,   scaler=scaler)
    X_test,  y_test,  _      = prepare_arrays(test_f,  scaler=scaler)

    val_f_clean  = val_f.dropna(subset=["target"]).reset_index(drop=True)
    test_f_clean = test_f.dropna(subset=["target"]).reset_index(drop=True)

    print(f"Feature matrix: {X_train.shape[1]} features\n")
    print("Training Proposed Ensemble (FT-Transformer + RF)...")

    y_true, y_pred, y_prob, test_df_w_probs, region_thresholds = run_proposed(
        X_train, y_train, X_val, y_val, X_test, y_test,
        val_f_clean, test_f_clean, device=device,
    )

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print("\n" + "=" * 50)
    print("PROPOSED MODEL RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        if k in ("TP", "FP", "TN", "FN"):
            print(f"  {k:<12} {v:>8,}")
        else:
            print(f"  {k:<12} {v:>8.4f}")

    # Per-region breakdown
    print("\nPer-region results:")
    per_region_rows = []
    for oblast in test_df_w_probs["oblast"].unique():
        sub  = test_df_w_probs[test_df_w_probs["oblast"] == oblast]
        yt_r = sub["target"].astype(int).values
        t    = region_thresholds.get(oblast, 0.5)
        yp_r = (sub["ens_prob"].values >= t).astype(int)
        m    = compute_metrics(yt_r, yp_r)
        per_region_rows.append({
            "oblast":    oblast,
            "threshold": round(t, 3),
            "samples":   len(sub),
            **{k: round(v, 4) for k, v in m.items()},
        })
        print(f"  {oblast:<35} τ={t:.2f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

    per_region_df = pd.DataFrame(per_region_rows)
    per_region_df.to_csv(f"{RESULTS_DIR}/per_region_results.csv", index=False)
    print(f"\nPer-region results saved to {RESULTS_DIR}/per_region_results.csv")


if __name__ == "__main__":
    main()
