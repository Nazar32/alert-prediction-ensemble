#!/usr/bin/env python3
"""
data_utils.py
=============
Shared data-loading, feature-engineering, and evaluation utilities used by
all models in this repository.

Public API
----------
load_data(path)             Load alert events → hourly active/inactive grid
engineer_features(df)       Compute 95 input features (70 numeric + 25 oblast dummies)
temporal_split(df)          Strict 70 / 15 / 15 train / val / test split
prepare_arrays(df, ...)     Build NumPy arrays, fit/apply StandardScaler
calibrate_threshold(...)    F1-max threshold search with optional precision floor
compute_metrics(...)        Precision / recall / F1 / accuracy / AUC / confusion matrix
mcnemar_test(...)           McNemar's test (two-sided) with continuity correction
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2 as chi2_dist

warnings.filterwarnings("ignore")

SEED        = 42
DATA_PATH   = "data/official_data_uk.csv"
RESULTS_DIR = "results"

np.random.seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Load and expand alert events into hourly active/inactive indicators.

    The source CSV contains one row per alert *event* (start/end timestamps).
    Each hour that overlaps with an active alert is marked alert_occurred=1.
    Counting only the start hour would severely under-count alert activity
    because most alerts span multiple hours.

    Returns a DataFrame with columns [oblast, hour, alert_occurred] covering
    every (oblast, hour) pair in the dataset date range, with alert_occurred
    already deduplicated (overlapping alerts in the same region-hour → 1).
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
# Oblast adjacency graph  (shared land borders, administrative geography)
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
    "Сумська область": [
        "Чернігівська область", "Полтавська область", "Харківська область",
    ],
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
    Add four neighbour-spillover features for each (hour, oblast):
      neighbor_alerts_now    – count of adjacent oblasts with active alert (same hour)
      neighbor_rate_now      – neighbor_alerts_now / adjacency degree
      neighbor_alerts_lag1h  – neighbor_alerts_now shifted 1 hour back
      neighbor_alerts_roll3h – 3-hour rolling mean of neighbor_alerts_now
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
    df["neighbor_rate_now"]   = df["neighbor_rate_now"].fillna(0).astype(np.float32)

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
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full 95-feature input matrix and the next-hour prediction target.

    Feature groups
    --------------
    Temporal (6)          hour_of_day, day_of_week, month, is_weekend, is_night, lag_0h
    Cyclical (6)          hour_sin/cos, day_sin/cos, month_sin/cos
    Lag (24)              lag_1h … lag_24h
    Rolling stats (18)    rate/count/std over windows {3,6,12,24,48,72} h
    Momentum & volatility (5)
    State transition (2)  alert_duration, hours_since_last_alert
    Spatial – national (5)
    Spatial – neighbours (4)
    Oblast identity (25)  one-hot encoding of 25 administrative regions

    Total: 70 continuous/binary features + 25 oblast dummies = 95 input dimensions
    Target: alert_occurred at t+1 (next-hour prediction)
    """
    df = df.copy()

    # Temporal
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["month"]       = df["hour"].dt.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"]    = df["hour_of_day"].isin(
        list(range(0, 6)) + list(range(22, 24))
    ).astype(int)

    # Cyclical encoding
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["day_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Current alert status (most informative single feature)
    df["lag_0h"] = df["alert_occurred"].astype(float)

    # Lag features (24-hour lookback)
    for i in range(1, 25):
        df[f"lag_{i}h"] = (
            df.groupby("oblast")["alert_occurred"].shift(i).fillna(0)
        )

    # Rolling statistics over multiple windows
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

    # Momentum
    df["momentum_3_12"]  = df["rate_3h"]  - df["rate_12h"]
    df["momentum_6_24"]  = df["rate_6h"]  - df["rate_24h"]
    df["momentum_12_72"] = df["rate_12h"] - df["rate_72h"]

    # Volatility / relative rate
    global_rate = df.groupby("oblast")["alert_occurred"].transform("mean")
    df["volatility_24h"] = (
        df.groupby("oblast")["alert_occurred"]
        .transform(lambda x: x.rolling(24, min_periods=1).std())
        .fillna(0)
    )
    df["rel_rate_24h"] = df["rate_24h"] / (global_rate + 1e-6)

    # Alert state-transition features
    def _run_length(s):
        """1-indexed position of each element within its constant run."""
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

    # Cross-region spatial features
    alerts_per_hour = df.groupby("hour")["alert_occurred"].transform("sum")
    df["other_regions_alert"] = alerts_per_hour - df["alert_occurred"]
    n_regions = df["oblast"].nunique()
    df["national_alert_rate"] = df["other_regions_alert"] / max(n_regions - 1, 1)
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

    # Neighbour spillover features
    df = _add_neighbor_spillover_features(df)

    # Oblast identity dummies
    oblast_dummies = pd.get_dummies(df["oblast"], prefix="oblast", dtype=np.float32)
    df = pd.concat([df, oblast_dummies], axis=1)

    # Target: next-hour alert
    df["target"] = df.groupby("oblast")["alert_occurred"].shift(-1)

    return df


# Feature column definitions
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

# Continuous features that are StandardScaler-normalised; oblast dummies are not.
NUMERIC_FEATURE_COLS = tuple(BASE_FEATURE_COLS) + NEIGHBOR_FEATURE_COLS


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return ordered list: numeric features + sorted oblast_* dummies present in df."""
    oblast_cols = sorted(c for c in df.columns if c.startswith("oblast_"))
    return list(NUMERIC_FEATURE_COLS) + oblast_cols


# ──────────────────────────────────────────────────────────────────────────────
# Temporal split and array preparation
# ──────────────────────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame):
    """
    Strict 70 / 15 / 15 train / val / test split on the time axis.
    No shuffling, no leakage between splits.
    """
    hours     = df["hour"].sort_values().unique()
    n         = len(hours)
    train_end = hours[int(n * 0.70)]
    val_end   = hours[int(n * 0.85)]
    train = df[df["hour"] <  train_end]
    val   = df[(df["hour"] >= train_end) & (df["hour"] < val_end)]
    test  = df[df["hour"] >= val_end]
    return train, val, test


def prepare_arrays(df_feat, scaler=None, fit_scaler=False):
    """
    Convert a feature DataFrame to NumPy arrays (X, y).

    Numeric features are z-scored with StandardScaler; oblast one-hot dummies
    are passed through unscaled.  Pass fit_scaler=True for the training set,
    then reuse the returned scaler for val/test.
    """
    df_clean  = df_feat.dropna(subset=["target"])
    cols      = get_feature_columns(df_feat)
    n_num     = len(NUMERIC_FEATURE_COLS)
    num_cols  = list(NUMERIC_FEATURE_COLS)
    oblast_cols = cols[n_num:]

    X_num = df_clean[num_cols].values.astype(np.float32)
    y     = df_clean["target"].astype(int).values

    if fit_scaler:
        scaler = StandardScaler()
        X_num  = scaler.fit_transform(X_num).astype(np.float32)
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

def calibrate_threshold(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
    target_prec: float = 0.0,
    min_predictions: int = 5,
) -> float | None:
    """
    Search 199 thresholds in [0.01, 0.99] and return the one that maximises F1.

    When target_prec > 0, only thresholds satisfying precision >= target_prec
    are considered.  Returns None if no threshold passes the constraint (callers
    should fall back to a global threshold calibrated without the floor).
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


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Return precision, recall, F1, accuracy, ROC-AUC, and raw confusion-matrix counts."""
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
