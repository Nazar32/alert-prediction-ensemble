# Feature List

Complete list of all 95 input features used by the DNN+RF ensemble.
All features are computed in `src/data_utils.py` → `engineer_features()`.

Continuous/binary features (70) are z-scored with `StandardScaler` before
being passed to the DNN and RF. Oblast identity dummies (25) are passed
through unscaled.

---

## Temporal (6 features)

| Feature | Description |
|---|---|
| `lag_0h` | Current alert status at time *t* (1 = alert active, 0 = no alert) |
| `hour_of_day` | Hour of day (0–23) |
| `day_of_week` | Day of week (0 = Monday, 6 = Sunday) |
| `month` | Month of year (1–12) |
| `is_weekend` | Binary: 1 if Saturday or Sunday |
| `is_night` | Binary: 1 if hour ∈ {0–5, 22–23} |

---

## Cyclical Encoding (6 features)

| Feature | Description |
|---|---|
| `hour_sin`, `hour_cos` | Sine/cosine encoding of `hour_of_day` over 24-hour cycle |
| `day_sin`, `day_cos` | Sine/cosine encoding of `day_of_week` over 7-day cycle |
| `month_sin`, `month_cos` | Sine/cosine encoding of `month` over 12-month cycle |

---

## Lag Features (24 features)

| Feature | Description |
|---|---|
| `lag_1h` … `lag_24h` | Binary alert status at *t*−1 through *t*−24 hours |

---

## Rolling Statistics (18 features)

Computed over windows *w* ∈ {3, 6, 12, 24, 48, 72} hours — 6 features per statistic type.

| Feature | Description |
|---|---|
| `rate_{w}h` | Mean alert rate over rolling window *w* |
| `count_{w}h` | Alert count over rolling window *w* |
| `std_{w}h` | Alert standard deviation over rolling window *w* |

---

## Momentum & Volatility (5 features)

| Feature | Description |
|---|---|
| `momentum_3_12` | `rate_3h` − `rate_12h`: short-term vs. medium-term trend |
| `momentum_6_24` | `rate_6h` − `rate_24h`: medium-term vs. daily trend |
| `momentum_12_72` | `rate_12h` − `rate_72h`: daily vs. three-day trend |
| `volatility_24h` | Rolling 24-hour standard deviation of alert occurrences |
| `rel_rate_24h` | `rate_24h` divided by the region's long-term mean alert rate |

---

## State Transition (2 features)

| Feature | Description |
|---|---|
| `alert_duration` | Consecutive hours in the current alert run (0 if no alert active) |
| `hours_since_last_alert` | Hours elapsed since the last alert ended (0 if alert currently active) |

---

## Spatial — National (5 features)

| Feature | Description |
|---|---|
| `other_regions_alert` | Count of other regions with an active alert at time *t* |
| `national_alert_rate` | `other_regions_alert` / (*R*−1), where *R* = 25 regions |
| `national_rate_3h` | Rolling 3-hour mean of `national_alert_rate` |
| `national_rate_12h` | Rolling 12-hour mean of `national_alert_rate` |
| `national_momentum` | `national_rate_3h` − `national_rate_12h` |

---

## Spatial — Neighbours (4 features)

Neighbours are defined by the `OBLAST_NEIGHBORS` adjacency graph in `src/data_utils.py`,
based on shared land borders between Ukrainian oblasts.

| Feature | Description |
|---|---|
| `neighbor_alerts_now` | Count of geographically adjacent regions with an active alert at time *t* |
| `neighbor_rate_now` | `neighbor_alerts_now` divided by the region's adjacency degree |
| `neighbor_alerts_lag1h` | Neighbour alert count lagged by 1 hour |
| `neighbor_alerts_roll3h` | Rolling 3-hour mean of neighbour alert counts |

---

## Oblast Identity Dummies (25 features)

| Feature | Description |
|---|---|
| `oblast_{name}` | One-hot encoding of the administrative region — one binary dummy per oblast |

The 25 oblasts are: Вінницька, Волинська, Дніпропетровська, Донецька, Житомирська,
Закарпатська, Запорізька, Івано-Франківська, Київська, Кіровоградська, Луганська,
Львівська, Миколаївська, Одеська, Полтавська, Рівненська, Сумська, Тернопільська,
Харківська, Херсонська, Хмельницька, Черкаська, Чернівецька, Чернігівська, м. Київ.

---

## Summary

| Group | Count |
|---|---|
| Temporal | 6 |
| Cyclical encoding | 6 |
| Lag (1–24 h) | 24 |
| Rolling statistics | 18 |
| Momentum & volatility | 5 |
| State transition | 2 |
| Spatial – national | 5 |
| Spatial – neighbours | 4 |
| **Subtotal (continuous/binary)** | **70** |
| Oblast identity dummies | 25 |
| **Total input dimensions** | **95** |
