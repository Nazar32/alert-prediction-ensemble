# Data

Place the dataset file here before running any experiment scripts.

## Required file

| Filename | Description |
|---|---|
| `official_data_uk.csv` | Ukrainian air-alert event log |

## Format

The CSV must contain at least the following columns:

| Column | Type | Description |
|---|---|---|
| `started_at` | datetime | Alert start timestamp (parseable by pandas) |
| `finished_at` | datetime | Alert end timestamp |
| `oblast` | string | Name of the administrative region (oblast) |
| `level` | string | Administrative level (`"oblast"`, `"raion"`, `"hromada"`) |

Only rows where `level == "oblast"` are used.  The loading code (`src/data_utils.load_data`) expands each alert event into individual hourly indicators, so the CSV stores one row per *event*, not per hour.

## Data availability

The dataset is sourced from the public Ukrainian air-alert API.  Due to potential redistribution restrictions, the file is not included in this repository.  Contact the authors if you require access to the exact dataset used in the paper.

## Expected statistics (paper dataset)

| Property | Value |
|---|---|
| Date range | March 2022 – October 2025 (~44 months) |
| Oblasts | 25 (24 regions + Kyiv city) |
| Total hourly observations | 785,000 (25 × 31,400 h) |
| Alert-hours (active) | 142,647 |
| Overall alert rate | ~18% |
