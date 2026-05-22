#!/usr/bin/env python3
"""
baseline_comparison.py
======================
Full comparison of the proposed ensemble against eight baseline models
(Section 6.2 of the paper).

Models evaluated:
  1. Persistence          – predict same as current hour (lag_0h)
  2. ARIMA (AR-24)        – AutoReg with 24 lags per region (statsmodels)
  3. Logistic Regression  – linear baseline with engineered features
  4. Random Forest        – standalone RF, class_weight='balanced'
  5. XGBoost              – XGBClassifier, scale_pos_weight
  6. LightGBM             – LGBMClassifier, is_unbalance=True
  7. CatBoost             – CatBoostClassifier, auto_class_weights='Balanced'
  8. LSTM                 – standalone 2-layer LSTM (PyTorch, BCE+pos_weight)
  9. Proposed Ensemble    – FT-Transformer + RF + CatBoost; 3-way weights tuned
                            on val F1; isotonic calibration; single global
                            threshold (F1-max on val, same protocol as baselines).

All models share the same engineered features (base + spatial + neighbour
spillover + oblast one-hots; exact width is printed at runtime) and the same
temporal 70/15/15 split.  ``prepare_arrays`` z-scores numeric columns only;
oblast one-hots stay in {0,1}.  Test metrics use only
rows with a defined next-hour target (25 oblasts × last test hour dropped).

Note: ARIMA fits per region on train only, but chooses its decision threshold
by scanning F1 on the test set (optimistic vs other models); McNemar vs ARIMA
should be interpreted cautiously.

Outputs written to results/:
  baseline_comparison_results.csv  – precision/recall/F1/AUC for all models
  mcnemar_test_results.csv         – McNemar chi2/p-value vs proposed ensemble
  per_region_results.csv           – proposed model breakdown per oblast

Run from the project root:
    python experiments/baseline_comparison.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from statsmodels.tsa.ar_model import AutoReg
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from proposed_model import (
    load_data, engineer_features, temporal_split, prepare_arrays,
    calibrate_threshold, compute_metrics, mcnemar_test, run_proposed,
    SEED, TARGET_PRECISION, DATA_PATH, RESULTS_DIR,
)

warnings.filterwarnings("ignore")

np.random.seed(SEED)
torch.manual_seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 1: Persistence
# ──────────────────────────────────────────────────────────────────────────────

def run_persistence(df_feat_test):
    # True persistence: predict next hour = current hour (lag_0h).
    # Using lag_1h (1 hour ago) would create a 2-step gap and underestimate
    # precision, since alerts are correlated over consecutive hours.
    df     = df_feat_test.dropna(subset=["target"]).copy()
    y_true = df["target"].astype(int).values
    y_pred = df["lag_0h"].astype(int).values
    y_prob = y_pred.astype(float)
    return y_true, y_pred, y_prob


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 2: ARIMA (AR-24 per region via statsmodels AutoReg)
# ──────────────────────────────────────────────────────────────────────────────

def run_arima(df_feat_train, df_feat_test):
    all_probs, all_true = [], []
    oblasts = df_feat_train["oblast"].unique()

    for i, oblast in enumerate(oblasts):
        print(f"  AR-24 fitting {i+1}/{len(oblasts)}: {oblast:<30}", end="\r")
        tr = df_feat_train[df_feat_train["oblast"] == oblast].sort_values("hour")
        te = df_feat_test[df_feat_test["oblast"] == oblast].sort_values("hour")
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

    # Threshold by best F1 (no separate val set for ARIMA; use test distribution)
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        f = f1_score(all_true, (all_probs >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    y_pred = (all_probs >= best_t).astype(int)
    return all_true, y_pred, all_probs


# ──────────────────────────────────────────────────────────────────────────────
# Baselines 3–7: sklearn-compatible classifiers
# ──────────────────────────────────────────────────────────────────────────────

def run_sklearn_model(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    """Fit, calibrate threshold on val set, evaluate on test set."""
    clf.fit(X_train, y_train)
    y_prob_val  = clf.predict_proba(X_val)[:, 1]
    y_prob_test = clf.predict_proba(X_test)[:, 1]
    threshold   = calibrate_threshold(y_val, y_prob_val)
    if threshold is None:
        threshold = 0.5
    y_pred = (y_prob_test >= threshold).astype(int)
    return y_test, y_pred, y_prob_test


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 8: Standalone LSTM  (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

class StandaloneLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # raw logit


def run_lstm(X_train, y_train, X_val, y_val, X_test, y_test,
             epochs=30, batch_size=512, device="cpu"):
    Xt = torch.tensor(X_train[:, None, :], dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(X_val[:, None, :],  dtype=torch.float32).to(device)
    Xe = torch.tensor(X_test[:, None, :], dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    model  = StandaloneLSTM(X_train.shape[1]).to(device)
    pos_w  = torch.tensor(
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


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}\n")

    # ── Load & engineer ───────────────────────────────────────────────────────
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

    results = {}

    # ── 1. Persistence ────────────────────────────────────────────────────────
    print("[1/9] Persistence...")
    yt, yp, yprob = run_persistence(test_f_clean)
    results["Persistence"] = dict(y_true=yt, y_pred=yp, y_prob=yprob,
                                  **compute_metrics(yt, yp, yprob))
    print(f"  Precision={results['Persistence']['precision']:.4f}  "
          f"Recall={results['Persistence']['recall']:.4f}")

    # ── 2. ARIMA (AR-24 per region) ───────────────────────────────────────────
    print("\n[2/9] ARIMA (AR-24 per region)...")
    yt, yp, yprob = run_arima(train_f, test_f_clean)
    results["ARIMA"] = dict(y_true=yt, y_pred=yp, y_prob=yprob,
                             **compute_metrics(yt, yp, yprob))
    print(f"  Precision={results['ARIMA']['precision']:.4f}  "
          f"Recall={results['ARIMA']['recall']:.4f}")

    # ── 3. Logistic Regression ────────────────────────────────────────────────
    print("\n[3/9] Logistic Regression...")
    lr_clf = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=SEED
    )
    yt, yp, yprob = run_sklearn_model(
        lr_clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results["Logistic Regression"] = dict(y_true=yt, y_pred=yp, y_prob=yprob,
                                           **compute_metrics(yt, yp, yprob))
    print(f"  Precision={results['Logistic Regression']['precision']:.4f}  "
          f"Recall={results['Logistic Regression']['recall']:.4f}")

    # ── 4. Random Forest (standalone) ─────────────────────────────────────────
    print("\n[4/9] Random Forest (standalone)...")
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    yt, yp, yprob = run_sklearn_model(
        rf_clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results["Random Forest"] = dict(y_true=yt, y_pred=yp, y_prob=yprob,
                                     **compute_metrics(yt, yp, yprob))
    print(f"  Precision={results['Random Forest']['precision']:.4f}  "
          f"Recall={results['Random Forest']['recall']:.4f}")

    # ── 5. XGBoost ────────────────────────────────────────────────────────────
    print("\n[5/9] XGBoost...")
    scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    xgb_clf   = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        scale_pos_weight=scale_pos, eval_metric="logloss",
        random_state=SEED, n_jobs=-1, verbosity=0,
    )
    yt, yp, yprob = run_sklearn_model(
        xgb_clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results["XGBoost"] = dict(y_true=yt, y_pred=yp, y_prob=yprob,
                               **compute_metrics(yt, yp, yprob))
    print(f"  Precision={results['XGBoost']['precision']:.4f}  "
          f"Recall={results['XGBoost']['recall']:.4f}")

    # ── 6. LightGBM ───────────────────────────────────────────────────────────
    print("\n[6/9] LightGBM...")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        is_unbalance=True, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    yt, yp, yprob = run_sklearn_model(
        lgb_clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results["LightGBM"] = dict(y_true=yt, y_pred=yp, y_prob=yprob,
                                **compute_metrics(yt, yp, yprob))
    print(f"  Precision={results['LightGBM']['precision']:.4f}  "
          f"Recall={results['LightGBM']['recall']:.4f}")

    # ── 7. CatBoost ───────────────────────────────────────────────────────────
    print("\n[7/9] CatBoost...")
    cat_clf = CatBoostClassifier(
        iterations=300, depth=8, learning_rate=0.05,
        auto_class_weights="Balanced", random_seed=SEED, verbose=0,
    )
    yt, yp, yprob = run_sklearn_model(
        cat_clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results["CatBoost"] = dict(y_true=yt, y_pred=yp, y_prob=yprob,
                                **compute_metrics(yt, yp, yprob))
    print(f"  Precision={results['CatBoost']['precision']:.4f}  "
          f"Recall={results['CatBoost']['recall']:.4f}")

    # ── 8. Standalone LSTM ────────────────────────────────────────────────────
    print("\n[8/9] Standalone LSTM...")
    yt, yp, yprob = run_lstm(
        X_train, y_train, X_val, y_val, X_test, y_test,
        epochs=30, device=device,
    )
    results["LSTM"] = dict(y_true=yt, y_pred=yp, y_prob=yprob,
                            **compute_metrics(yt, yp, yprob))
    print(f"  Precision={results['LSTM']['precision']:.4f}  "
          f"Recall={results['LSTM']['recall']:.4f}")

    # ── 9. Proposed ensemble ──────────────────────────────────────────────────
    print("\n[9/9] Proposed ensemble (FT-Transformer + RF + CatBoost, global threshold)...")
    yt, yp, yprob, test_df_w_probs, region_thresholds = run_proposed(
        X_train, y_train, X_val, y_val, X_test, y_test,
        val_f_clean, test_f_clean, device=device,
    )
    results["Proposed"] = dict(y_true=yt, y_pred=yp, y_prob=yprob,
                                **compute_metrics(yt, yp, yprob))
    print(f"  Precision={results['Proposed']['precision']:.4f}  "
          f"Recall={results['Proposed']['recall']:.4f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    metric_keys = ["precision", "recall", "f1", "accuracy", "roc_auc",
                   "TP", "FP", "TN", "FN"]
    rows = [{"model": name, **{k: r.get(k, float("nan")) for k in metric_keys}}
            for name, r in results.items()]
    summary = pd.DataFrame(rows).set_index("model")

    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print(summary[["precision", "recall", "f1", "accuracy", "roc_auc"]]
          .round(4).to_string())

    summary.to_csv(f"{RESULTS_DIR}/baseline_comparison_results.csv")

    # ── McNemar's tests (Proposed vs each baseline) ───────────────────────────
    prop_true = results["Proposed"]["y_true"]
    prop_pred = results["Proposed"]["y_pred"]

    mcnemar_rows = []
    for name, r in results.items():
        if name == "Proposed":
            continue
        if len(r["y_true"]) != len(prop_true):
            raise ValueError(
                f"McNemar requires aligned test rows: Proposed n={len(prop_true)}, "
                f"{name} n={len(r['y_true'])}"
            )
        n = len(prop_true)
        stat, pval = mcnemar_test(prop_true[:n], prop_pred[:n], r["y_pred"][:n])
        mcnemar_rows.append({
            "baseline":         name,
            "chi2_statistic":   round(stat, 4),
            "p_value":          round(pval, 6),
            "significant_p005": pval < 0.05,
            "significant_p001": pval < 0.001,
        })

    mcnemar_df = pd.DataFrame(mcnemar_rows).set_index("baseline")
    print("\nMcNemar's test (Proposed vs each baseline):")
    print(mcnemar_df.to_string())
    mcnemar_df.to_csv(f"{RESULTS_DIR}/mcnemar_test_results.csv")

    # ── Per-region results for proposed model ─────────────────────────────────
    # run_proposed stores only {"_global": τ}; apply that threshold everywhere.
    global_t = region_thresholds.get("_global", 0.5)
    per_region_rows = []
    for oblast in test_df_w_probs["oblast"].unique():
        sub  = test_df_w_probs[test_df_w_probs["oblast"] == oblast]
        yt_r = sub["target"].astype(int).values
        yp_r = (sub["ens_prob"].values >= global_t).astype(int)
        m    = compute_metrics(yt_r, yp_r)
        per_region_rows.append({
            "oblast":    oblast,
            "threshold": round(global_t, 3),
            "samples":   len(sub),
            **{k: round(v, 4) for k, v in m.items()},
        })

    per_region_df = pd.DataFrame(per_region_rows)
    per_region_df.to_csv(f"{RESULTS_DIR}/per_region_results.csv", index=False)
    print(f"\nPer-region results saved.")

    print(f"\nAll results written to {RESULTS_DIR}/")
    print("  baseline_comparison_results.csv")
    print("  mcnemar_test_results.csv")
    print("  per_region_results.csv")


if __name__ == "__main__":
    main()
