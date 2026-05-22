#!/usr/bin/env python3
"""Run LSTM with per-region calibration. Save results to disk."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from proposed_model import (
    load_data, engineer_features, temporal_split, prepare_arrays,
    compute_metrics, SEED,
)

DATA_PATH = "data/official_data_uk.csv"
TARGET_PRECISION = 0.75
np.random.seed(SEED)
torch.manual_seed(SEED)

def calibrate_threshold_prec(y_val, y_prob_val, target_prec=TARGET_PRECISION):
    best_t, best_f1 = None, -1.0
    for t in np.linspace(0.01, 0.99, 199):
        y_pred = (y_prob_val >= t).astype(int)
        if y_pred.sum() < 5:
            continue
        if target_prec > 0:
            p = precision_score(y_val, y_pred, zero_division=0)
            if p < target_prec:
                continue
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

class StandaloneLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = ("mps" if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

raw = load_data(DATA_PATH)
feat = engineer_features(raw)
train_f, val_f, test_f = temporal_split(feat)
X_train, y_train, scaler = prepare_arrays(train_f, fit_scaler=True)
X_val, y_val, _          = prepare_arrays(val_f,   scaler=scaler)
X_test, y_test, _        = prepare_arrays(test_f,  scaler=scaler)
val_clean  = val_f.dropna(subset=["target"]).reset_index(drop=True)
test_clean = test_f.dropna(subset=["target"]).reset_index(drop=True)

print("Training LSTM...")
Xt = torch.tensor(X_train[:, None, :], dtype=torch.float32)
yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
Xv = torch.tensor(X_val[:, None, :],  dtype=torch.float32).to(device)
Xe = torch.tensor(X_test[:, None, :], dtype=torch.float32).to(device)
loader = DataLoader(TensorDataset(Xt, yt), batch_size=512, shuffle=True)

model     = StandaloneLSTM(X_train.shape[1]).to(device)
pos_w     = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
                          dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val, patience, best_state = np.inf, 0, None
for epoch in range(30):
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
            print(f"  Early stop epoch {epoch+1}")
            break
    if (epoch + 1) % 5 == 0:
        print(f"  epoch {epoch+1}/30  val_loss={val_loss:.5f}")

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    p_val  = torch.sigmoid(model(Xv)).cpu().numpy().ravel()
    p_test = torch.sigmoid(model(Xe)).cpu().numpy().ravel()

global_t = calibrate_threshold_prec(y_val, p_val)
if global_t is None:
    global_t = calibrate_threshold_prec(y_val, p_val, target_prec=0.0)

val_clean["_prob"]  = p_val
test_clean["_prob"] = p_test
thresholds = {"_global": global_t}
for oblast in test_clean["oblast"].unique():
    vm = val_clean["oblast"] == oblast
    if vm.sum() < 30:
        thresholds[oblast] = global_t
        continue
    t = calibrate_threshold_prec(val_clean.loc[vm, "target"].astype(int).values,
                                  val_clean.loc[vm, "_prob"].values)
    thresholds[oblast] = t if t is not None else global_t

y_pred = np.zeros(len(test_clean), dtype=int)
for oblast in test_clean["oblast"].unique():
    mask = (test_clean["oblast"] == oblast).values
    t = thresholds.get(oblast, global_t)
    y_pred[mask] = (test_clean.loc[mask, "_prob"].values >= t).astype(int)

y_true = test_clean["target"].astype(int).values
m = compute_metrics(y_true, y_pred, p_test)
print(f"LSTM+per-region: Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  "
      f"F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")
pd.DataFrame([{"model": "LSTM+per-region", **m}]).to_csv(
    "results/lstm_per_region_results.csv", index=False)
print("Saved results/lstm_per_region_results.csv")
