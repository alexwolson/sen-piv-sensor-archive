# Amalgamate Steel Flow Data

## Repository Overview
- Purpose: consolidate particle image velocimetry (PIV) velocity fields and their paired mould sensor traces for submerged entry nozzle (SEN) experiments.
- Primary scripts: `preprocess.py` (batch aggregation of PIV CSV sequences with sensor alignment) and `main.py` (placeholder CLI entry). Supporting modules are expected under `src/` (physics metrics, config, data utilities).
- Environment: Python 3.11 managed with `uv`. Install dependencies via `uv sync`; execute commands with `uv run …`.

## Data Inventory
- **PIV Velocity Arrays** (external): OneDrive folder `Non-Aggregated_Data/` organised by SEN, strand speed, gas flow, and run. Captured at 300 Hz (~0.003 s per frame) with 999 CSV frames per experiment.
- **Sensor Logs** (repository): `data/raw/sensor_logs/<SEN>/<variant>/<speed_slug>/YYYY-MM-DD__SENXX__<variant>__<speed>__<gas>LPM__PIV##.xlsx`
  - `variant` is `baseline` or clogging experiments (`lclog75`, `lclog80`, `lclog85`, `lclog90`).
  - `speed_slug` replaces decimal points (`1p2mpm`, `1p4mpm`, `1p8mpm`).
  - Canonical naming ensures sortable timestamps, gas rates, and run IDs (`PIV01–PIV03`).
  - `data/intermediate/sensor_logs_manifest.csv` records original vs. canonical paths with metadata (sensor, variant, speed, gas flow, run, timestamp).

## Sensor Workbook Format
- Sampling: 100 Hz (0.01 s cadence). Baseline runs span ~33 s (≈3300 rows); clogging campaigns ~13 s (≈1300 rows).
- Core columns (present after reconciliation):
  - `time[s]`, analogue (AN) and ultrasonic (US) voltage channels per quadrant, velocity-converted channels, mould level (`ML_*[mm]`), and derived wave heights.
  - `PIV_ON` boolean marking the active imaging window; metadata rows store `PIV Start Time [s]` / `PIV End Time [s]` and quadrant labels (`LL`, `LQ`, `RQ`, `RR`).
  - Summary metrics (`RQ Mean/SD`, `RR Mean/SD` or legacy variants) and XGBoost predictions `data_cleaned_ignore_time_geo_mld_Count_EX1_…` and `…EX2_…` present in every readable workbook.
- Cleaning expectations:
  1. Drop the first four metadata rows.
  2. Harmonise legacy column aliases (e.g., `RQM`, `RQ m`, `RQ M`) to the canonical schema; fill missing columns with `NaN`.
  3. Confirm `time[s]` monotonicity and 0.01 s deltas.

## Aligning Sensor Logs With PIV Frames
1. Load workbook (`pd.read_excel`) and clean metadata rows.
2. Extract the active window: `piv_window = df[df["PIV_ON"]]`, or slice between stored start/end times.
3. Align three sensor samples per PIV frame (as `preprocess.py` expects) using integer division of indices or by following `aligned/frame_idx` in generated HDF5 outputs.
4. Persist cleaned slices as Parquet/HDF5 alongside velocity arrays; store alignment metadata for reproducibility.

## Data Directory Layout
All datasets now live under the `data/` directory:

```
data/
├── raw/
│   ├── sensor_logs/              # original Excel workbooks
│   └── piv_velocity_raw/         # raw Dantec CSV downloads
├── intermediate/
│   ├── sensor_logs_canonical/    # symlink tree of canonicalised workbooks
│   ├── sensor_logs_manifest.csv  # metadata for raw→canonical mapping
│   ├── sensor_logs_aligned/      # aligned Parquet windows
│   ├── sensor_logs_aligned_manifest.csv
│   ├── piv_velocity/             # canonicalised PIV directory tree
│   ├── piv_velocity_manifest.csv
│   └── experiments_manifest.csv  # merged sensor+PIV coverage
└── processed/
    ├── all_experiments.h5        # consolidated archive
    └── coverage/                 # per-SEN coverage reports
```

## Processing Workflow
1. Run `uv run python preprocess.py --rebuild-intermediate` to (re)generate canonical sensor logs, aligned Parquet windows, canonical PIV directories, the merged manifest, SEN coverage reports, and the consolidated HDF5 archive.
2. Use `--limit` when re-running the HDF5 assembly in a constrained environment to process a subset of runs per invocation (the script skips completed groups automatically).
3. All manifests live under `data/intermediate/` and are kept in sync with the HDF5 output; coverage reports land in `data/processed/coverage/`.

### Preprocessing Pipeline
`preprocess.py` now orchestrates the full flow:

1. Normalises raw sensor logs into the canonical layout (`data/intermediate/sensor_logs_canonical/`) and emits `sensor_logs_manifest.csv`.
2. Aligns every workbook into a consistent Parquet schema under `data/intermediate/sensor_logs_aligned/` with the accompanying manifest.
3. Normalises PIV CSV directories into `data/intermediate/piv_velocity/` and records metadata in `piv_velocity_manifest.csv`.
4. Builds the merged experiments manifest (`data/intermediate/experiments_manifest.csv`) and regenerates coverage reports (`data/processed/coverage/`).
5. Writes the consolidated HDF5 archive at `data/processed/all_experiments.h5`.

Use `uv run python preprocess.py --rebuild-intermediate` to force regeneration of all intermediate artefacts before creating the final archive.

To rebuild everything from scratch (including the final HDF5), run:

```
uv run python preprocess.py --rebuild-intermediate --refresh
```

## End-State Consolidation Plan
Goal: publish a single HDF5 archive containing only experiments that have verified PIV velocity frames, aligned sensor traces (including bubble-count predictions), and consistent metadata.

1. **Complete Ingestion**
   - Download every SEN’s PIV sequences into `data/raw/piv_velocity_raw/`, verifying each run has the full 999-frame suite with no missing indices.
   - Maintain/update the sensor manifest in `data/intermediate/sensor_logs_manifest.csv` to capture all available workbooks, ensuring the canonical column set and data types remain stable.
2. **Manifest Integration**
   - Merge sensor and PIV manifests into a unified table keyed by `(SEN, strand_speed, gas_flow, PIV_run)`. Record provenance fields (dates, Originator strings, acquisition times) as metadata.
3. **Automated Validation**
   - Confirm column schemas: harmonise aliases, fill missing columns with `NaN`, and assert every workbook carries the XGBoost prediction fields.
   - Check coordinate geometry: grid dimensions must be 30×22 with consistent physical spacing; capture per-run deltas if observed.
   - Align time windows: the `PIV_ON` boolean and `PIV Start/End Time` must match the PIV CSV timestamp range (0.003333 s increments). Flag discrepancies beyond tolerance and exclude the run until resolved.
   - Ensure monotonic `time[s]` in sensors and continuous frame indices in PIV; a gap or duplicate should mark the run as invalid.
4. **HDF5 Assembly**
   - Design a hierarchy such as `/runs/<SEN>/<speed>/<gas>/PIV##/{piv_data, sensor_data, metadata}` with chunking along time and GZIP compression.
   - Store the raw velocity tensors, aligned sensor tables, bubble-count predictions, and alignment metadata together with manifest attributes (date, variant, acquisition timestamps, processing versions).
5. **Quality Gates & Releases**
   - Run validation scripts as part of the build; fail the job if a run lacks any data source or fails alignment tests.
   - Produce coverage reports under `data/processed/coverage/` for each SEN, tracking missing components until the archive is complete.
   - Version the HDF5 output (checksums, release tags) so collaborators can pin analyses to specific data snapshots.
6. **Documentation & Onboarding**
   - Update this README (and any supplemental docs) with instructions for adding new experiments, rebuilding the archive, and interpreting HDF5 contents.
   - Encourage scripted ingestion only—avoid manual edits to PIV CSVs or sensor workbooks to maintain reproducibility.

## Contributor Guide
- **Setup**: `uv sync` → `.venv`; run scripts with `uv run python …`.
- **Coding style**: Python 3.11, 4-space indentation, type hints on public APIs. Use `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants. Add formatting tools (e.g., `ruff`, `black`) to `pyproject.toml` if adopted.
- **Testing**: Use `pytest`; mirror source structure under `tests/`. Name files `test_<feature>.py` and mark slow data integrations (e.g., `@pytest.mark.slow`). Run `uv run pytest` before PRs and document failing cases if any.
- **Commits & PRs**: Imperative subject lines (`feat: align sensor window`). PR descriptions should summarise changes, reference affected datasets (e.g., `data/raw/sensor_logs/SEN07/...`), list commands run, and link tasks/issues. Flag large data or schema changes early.
- **Data handling**: Treat `data/raw/sensor_logs/` and `data/intermediate/sensor_logs_manifest.csv` as read-only sources. Keep large raw datasets outside git (ensure `.gitignore` covers local paths). Use scripted transformations rather than manual Excel edits.

## Historical Context
- Original notes highlighted two complementary datasets: non-aggregated PIV velocity arrays (999 frames per run) and sensor spreadsheets sampled at 100 Hz with `PIV_ON` flags and XGBoost bubble-count predictions. Renaming from `pivdata` to `sensor_logs` clarifies the content and enforces a consistent layout for downstream tooling.
