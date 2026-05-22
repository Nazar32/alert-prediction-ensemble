"""
generate_synthetic_data.py — Synthetic Ukrainian air-alert dataset generator.

Produces a CSV with the same schema and column types as
`alert_hts/data/official_data_uk.csv`, so the full training/evaluation
pipeline (proposed_model.py, baseline_per_region_calibration.py, etc.)
can be executed end-to-end **without** the original (restricted) dataset.

Output schema (matches official_data_uk.csv):
    oblast, raion, hromada, level, started_at, finished_at, source

The generator does NOT attempt to reproduce real military-operational
patterns. It produces a plausibly-structured synthetic stream that
preserves three aspects relevant to the modelling pipeline:

  1. 25 oblast partitions with **heterogeneous base rates** so that
     per-region calibration has something non-trivial to calibrate.
  2. **Temporal autocorrelation** (alerts cluster, persistence baseline
     is informative) modelled via two-state Markov chains per oblast.
  3. Realistic event durations (mean ~2 h, exponential-ish tail).

Usage:
    python generate_synthetic_data.py \\
        --start 2022-03-15 \\
        --end   2025-10-13 \\
        --out   data/synthetic_data_uk.csv \\
        --seed  42

To run the full pipeline on synthetic data, override DATA_PATH in
`experiments/extract_per_region_detailed.py` (or set the same env var)
to point at the generated CSV.

Notes
-----
* All metric values produced from this synthetic dataset will differ
  from the figures reported in the paper. The script is intended for
  *pipeline verification*, not for reproducing the empirical results.
* SEED=42 is the same seed used in the published experiments; this
  makes the synthetic stream deterministic and shareable.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import math
import random
import sys
from pathlib import Path

# Ordered list of 25 Ukrainian oblast names (Cyrillic, matching the
# real dataset's `oblast` column). Each entry pairs the name with an
# approximate **steady-state alert probability** used to drive the
# Markov chain. Values are illustrative — sufficient to exercise the
# per-region calibration path on a wide range of base rates.
OBLAST_BASE_RATES = [
    ("Вінницька область",         0.09),
    ("Волинська область",         0.05),
    ("Дніпропетровська область",  0.22),
    ("Донецька область",          0.85),
    ("Житомирська область",       0.13),
    ("Закарпатська область",      0.03),
    ("Запорізька область",        0.50),
    ("Івано-Франківська область", 0.04),
    ("Київська область",          0.16),
    ("Кіровоградська область",    0.19),
    ("Луганська область",         0.00),  # occupied — no alerts recorded
    ("Львівська область",         0.04),
    ("Миколаївська область",      0.21),
    ("Одеська область",           0.17),
    ("Полтавська область",        0.26),
    ("Рівненська область",        0.05),
    ("Сумська область",           0.53),
    ("Тернопільська область",     0.04),
    ("Харківська область",        0.46),
    ("Херсонська область",        0.19),
    ("Хмельницька область",       0.04),
    ("Черкаська область",         0.10),
    ("Чернівецька область",       0.04),
    ("Чернігівська область",      0.29),
    ("м. Київ",                   0.14),
]

# Mean alert duration in hours (matching the article's ~2.22 h figure
# derived from real data); we sample a duration from an exponential
# distribution truncated to [0.25 h, 12 h].
MEAN_DURATION_H = 2.22
MIN_DURATION_H  = 0.25
MAX_DURATION_H  = 12.0


def _sample_duration(rng: random.Random) -> float:
    """Truncated exponential alert duration in hours."""
    while True:
        d = rng.expovariate(1.0 / MEAN_DURATION_H)
        if MIN_DURATION_H <= d <= MAX_DURATION_H:
            return d


def _markov_alert_stream(rng: random.Random, n_hours: int,
                          base_rate: float) -> list[bool]:
    """Two-state Markov chain producing hourly y_t in {0, 1} with
    steady-state probability `base_rate` and persistence p≈0.85
    when active (so mean alert run length is ~6 h on the hourly grid,
    matching the heuristic 2-3 h continuous-time duration).
    """
    if base_rate <= 0:
        return [False] * n_hours
    p_stay_active = 0.85
    # Solve for p_start so steady-state = base_rate.
    p_start = base_rate * (1.0 - p_stay_active) / (1.0 - base_rate)
    p_start = max(0.0, min(1.0, p_start))
    out = []
    active = False
    for _ in range(n_hours):
        if active:
            active = rng.random() < p_stay_active
        else:
            active = rng.random() < p_start
        out.append(active)
    return out


def _hours_to_events(active_flags: list[bool], anchor: dt.datetime,
                      rng: random.Random) -> list[tuple[dt.datetime, dt.datetime]]:
    """Compress a sequence of hourly active flags into discrete
    (started_at, finished_at) events. Each run of consecutive
    actives becomes one event whose duration is a single sampled
    exponential value (so the hourly grid is the *consequence* of
    discrete events, not their parent)."""
    events = []
    i = 0
    n = len(active_flags)
    while i < n:
        if not active_flags[i]:
            i += 1
            continue
        # Found a run start.
        start_hour = i
        while i < n and active_flags[i]:
            i += 1
        end_hour = i  # exclusive
        # Sample a real-valued duration inside the run window.
        run_h = end_hour - start_hour
        # Add a fractional offset for "started_at" within the first hour.
        start_offset_min = rng.randint(0, 59)
        started = anchor + dt.timedelta(hours=start_hour,
                                         minutes=start_offset_min)
        d_h = max(0.25, min(run_h - 0.1, _sample_duration(rng)))
        finished = started + dt.timedelta(hours=d_h)
        events.append((started, finished))
    return events


def generate(start: dt.date, end: dt.date, out_path: Path,
             seed: int = 42) -> None:
    rng = random.Random(seed)
    anchor = dt.datetime(start.year, start.month, start.day,
                          tzinfo=dt.timezone.utc)
    total_hours = int((dt.datetime(end.year, end.month, end.day,
                                    tzinfo=dt.timezone.utc) - anchor)
                       .total_seconds() // 3600)
    print(f"[synthetic] Generating {total_hours:,} hours "
          f"({start} → {end}) for {len(OBLAST_BASE_RATES)} oblasts...")

    rows: list[dict] = []
    for oblast, base_rate in OBLAST_BASE_RATES:
        local_rng = random.Random(rng.randint(0, 2**31 - 1))
        flags = _markov_alert_stream(local_rng, total_hours, base_rate)
        events = _hours_to_events(flags, anchor, local_rng)
        for started, finished in events:
            rows.append({
                "oblast": oblast,
                "raion":  "",
                "hromada": "",
                "level":  "oblast",
                "started_at":  started.isoformat(sep=" "),
                "finished_at": finished.isoformat(sep=" "),
                "source":      "synthetic",
            })
        print(f"  {oblast:30s} → {len(events):,} events "
              f"(base rate ≈ {base_rate:.2%})")

    rows.sort(key=lambda r: (r["started_at"], r["oblast"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "oblast", "raion", "hromada", "level",
            "started_at", "finished_at", "source",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[synthetic] Wrote {len(rows):,} events to {out_path}")
    print(f"[synthetic] Total span: {total_hours:,} hours over "
          f"{len(OBLAST_BASE_RATES)} oblasts.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=lambda s: dt.date.fromisoformat(s),
                    default=dt.date(2022, 3, 15),
                    help="Start date (default: 2022-03-15)")
    p.add_argument("--end",   type=lambda s: dt.date.fromisoformat(s),
                    default=dt.date(2025, 10, 13),
                    help="End date (default: 2025-10-13)")
    p.add_argument("--out",   type=Path,
                    default=Path("data/synthetic_data_uk.csv"),
                    help="Output CSV path (default: data/synthetic_data_uk.csv)")
    p.add_argument("--seed",  type=int, default=42,
                    help="RNG seed (default: 42)")
    args = p.parse_args()

    if args.end <= args.start:
        print("error: --end must be after --start", file=sys.stderr)
        sys.exit(2)

    generate(args.start, args.end, args.out, args.seed)


if __name__ == "__main__":
    main()
