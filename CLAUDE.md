# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Consolidates Particle Image Velocimetry (PIV) velocity fields and paired mould sensor traces from Submerged Entry Nozzle (SEN) steel-casting experiments into a single Zarr v3 archive for downstream analysis.

## Commands

```bash
# Run preprocessing pipeline (builds the Zarr archive)
uv run python preprocess.py

# Explore the resulting Zarr archive interactively
uv run python explore_zarr.py

# Common explore_zarr.py flags
uv run python explore_zarr.py --list                                          # list all experiments
uv run python explore_zarr.py --tree                                          # print tree structure
uv run python explore_zarr.py --experiment SEN07/lclog75/1p2mpm/6lpm/PIV01   # show one experiment

# Install dependencies
uv sync
```

## Architecture

### Data Pipeline

Raw inputs live under `data/raw/` and `data/intermediate/`; processed outputs go to `data/processed/`.

1. **`preprocess.py`** — the main pipeline script. It:
   - Reads `data/intermediate/experiments_manifest.csv` to determine which experiments are included/excluded.
   - Loads PIV velocity fields from `.csv` files (one per frame) stored in `data/intermediate/piv_velocity/<SEN>/<variant>/<speed>/<gas_run>/`.
   - Loads aligned sensor traces from `.parquet` files in `data/intermediate/sensor_logs_aligned/`.
   - Writes everything into `data/processed/all_experiments.zarr` (Zarr v3 format, local store).

2. **`explore_zarr.py`** — interactive and CLI-driven explorer for the output Zarr archive.

### Zarr Archive Layout

```
all_experiments.zarr/
└── runs/
    └── <SEN>/          e.g. SEN06, SEN07
        └── <variant>/  e.g. baseline, lclog75
            └── <speed>/  e.g. 1p2mpm, 1p8mpm
                └── <gas>lpm/
                    └── <PIVNN>/   e.g. PIV01
                        ├── piv/
                        │   ├── u          (frames × height × width float32)
                        │   ├── v          (frames × height × width float32)
                        │   ├── x_mm       (width,)
                        │   ├── y_mm       (height,)
                        │   └── time_s     (frames,)
                        └── sensor/
                            ├── table      (rows × columns float32)
                            └── attrs: columns list
```

Experiment-level attributes (stored in Zarr group `.attrs`): `sen`, `variant`, `strand_speed_slug`, `gas_flow_lpm`, `piv_run`, `n_frames`, `date_iso`, etc.

### Key Constants in `preprocess.py`

- `SENSOR_TIME_COLUMN = "time[s]"` — time column name in sensor parquets.
- `SENSOR_WINDOW_FLAG = "PIV_ON"` — boolean column flagging rows that overlap the PIV acquisition window.
- `CANONICAL_FILENAME_RE` / `LEGACY_FILENAME_RE` — regex patterns for matching sensor workbook filenames.
- `BUBBLE_COLUMNS` — two specific mould-clogging prediction columns extracted from sensor data.

### Manifest File

`data/intermediate/experiments_manifest.csv` is the source of truth for which experiments are processed. Key columns: `sen`, `variant`, `strand_speed_slug`, `gas_flow_lpm`, `piv_run`, `status` (`included`/`excluded`), `exclusion_reason`, `source_path`, `output_path`.

### Intermediate Data

- `data/intermediate/piv_velocity_manifest.csv` — maps PIV source directories to normalized paths and frame counts.
- `data/intermediate/sensor_logs_manifest.csv` — maps raw sensor workbooks to canonical paths.
- `data/intermediate/sensor_logs_aligned_manifest.csv` — maps aligned sensor parquets.
- `data/processed/coverage/SEN*_data_coverage.md` — per-SEN markdown summaries of included/excluded runs.
