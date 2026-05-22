#!/usr/bin/env python3
# ===== DEPRECATED =====
# Legacy combined runner using the FT-Transformer + GRU + CatBoost stack from
# src/proposed_model.py with TARGET_PRECISION=0. The current pipeline uses
# src/proposed_model_dnn_rf_fixed.py and src/run_*_per_region.py.
"""Quick runner: proposed ensemble vs key baselines on MPS GPU."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from proposed_model import *
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}", flush=True)

print("Loading data...", flush=True)
raw = load_data(DATA_PATH)
feat = engineer_features(raw)
train_f, val_f, test_f = temporal_split(feat)

X_train, y_train, scaler = prepare_arrays(train_f, fit_scaler=True)
X_val,   y_val,   _      = prepare_arrays(val_f, scaler=scaler)
X_test,  y_test,  _      = prepare_arrays(test_f, scaler=scaler)

train_clean = train_f.dropna(subset=["target"]).reset_index(drop=True)
val_clean  = val_f.dropna(subset=["target"]).reset_index(drop=True)
test_clean = test_f.dropna(subset=["target"]).reset_index(drop=True)
print(f"Features: {X_train.shape[1]}  Train: {len(y_train):,}  "
      f"Val: {len(y_val):,}  Test: {len(y_test):,}\n", flush=True)

# ── Proposed ensemble ────────────────────────────────────────────────────
print("=" * 60, flush=True)
print("[1/5] Proposed Ensemble (FT-Hybrid + GRU + CatBoost, OOF stack)", flush=True)
print("=" * 60, flush=True)
# Lighter OOF so a full run finishes in ~1h on MPS (raise for paper-quality)
yt, yp, yprob, test_df, thresholds = run_proposed(
    X_train, y_train, X_val, y_val, X_test, y_test,
    val_clean, test_clean, device=device,
    df_feat_train=train_clean,
    stack_n_splits=2,
    oof_ft_epochs=8,
    oof_gru_epochs=8,
    oof_cat_iterations=120,
    epochs=24,
)
proposed_m = compute_metrics(yt, yp, yprob)
print(flush=True)

# ── Baselines ────────────────────────────────────────────────────────────
scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
baselines = [
    ("Random Forest",       RandomForestClassifier(n_estimators=200, max_depth=20,
        class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ("CatBoost",            CatBoostClassifier(iterations=300, depth=8,
        learning_rate=0.05, auto_class_weights="Balanced", random_seed=SEED, verbose=0)),
    ("Logistic Regression", LogisticRegression(class_weight="balanced",
        max_iter=1000, random_state=SEED)),
]

all_results = {"Proposed": dict(m=proposed_m, y_true=yt, y_pred=yp)}

for i, (name, clf) in enumerate(baselines, 2):
    total = len(baselines) + 1
    print(f"[{i}/{total}] {name}...", end=" ", flush=True)
    clf.fit(X_train, y_train)
    pv = clf.predict_proba(X_val)[:, 1]
    pt = clf.predict_proba(X_test)[:, 1]
    t = calibrate_threshold(y_val, pv)
    if t is None:
        t = 0.5
    yp_b = (pt >= t).astype(int)
    m = compute_metrics(y_test, yp_b, pt)
    all_results[name] = dict(m=m, y_true=y_test, y_pred=yp_b)
    print(f"done", flush=True)

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"{'Model':<25} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7}")
print("-" * 72)
for name in ["Proposed"] + [n for n, _ in baselines]:
    m = all_results[name]["m"]
    marker = " <<<" if name == "Proposed" else ""
    print(f"{name:<25} {m['precision']:>7.4f} {m['recall']:>7.4f} "
          f"{m['f1']:>7.4f} {m['roc_auc']:>7.4f}{marker}")
print("=" * 72)

# ── Conditional: lag_0h == 0 (hard onset cases) ─────────────────────────
lag0_mask = test_clean["lag_0h"].values == 0
print(f"\nCONDITIONAL: lag_0h == 0  ({lag0_mask.sum()} of {len(y_test)} rows)")
print("-" * 72)
print(f"{'Model':<25} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7}")
print("-" * 72)
for name in ["Proposed"] + [n for n, _ in baselines]:
    r = all_results[name]
    yt_c = r["y_true"][lag0_mask]
    yp_c = r["y_pred"][lag0_mask]
    mc = compute_metrics(yt_c, yp_c)
    marker = " <<<" if name == "Proposed" else ""
    auc_str = f"{mc.get('roc_auc', float('nan')):>7.4f}"
    print(f"{name:<25} {mc['precision']:>7.4f} {mc['recall']:>7.4f} "
          f"{mc['f1']:>7.4f} {auc_str}{marker}")
print("=" * 72)

# ── McNemar's tests ──────────────────────────────────────────────────────
print("\nMcNemar's test (Proposed vs each baseline):")
prop_pred = all_results["Proposed"]["y_pred"]
prop_true = all_results["Proposed"]["y_true"]
for name in [n for n, _ in baselines]:
    bp = all_results[name]["y_pred"]
    stat, pval = mcnemar_test(prop_true, prop_pred, bp)
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    print(f"  vs {name:<22} chi2={stat:>8.2f}  p={pval:.6f}  {sig}")

print("\nDone!", flush=True)
