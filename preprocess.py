#!/usr/bin/env python3
"""
Build a single HDF5 archive containing only experiments with validated PIV,
sensor, and bubble-count data as described in
data/intermediate/experiments_manifest.csv.
"""

from __future__ import annotations

import argparse
import logging
import textwrap
import csv
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger("amalgamate-preprocess")

SENSOR_TIME_COLUMN = "time[s]"
SENSOR_WINDOW_FLAG = "PIV_ON"
BUBBLE_COLUMNS = [
    "data_cleaned_ignore_time_geo_mld_Count_EX1_OneHot_Geometrical_Clogging_0Lag_Mould",
    "data_cleaned_ignore_time_geo_mld_Count_EX2_OneHot_Geometrical_Clogging_0Lag_Mould",
]
WORKBOOK_SUFFIXES = {".xlsx", ".xlsm"}

CANONICAL_FILENAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})__(?P<sen>SEN\d{2})__(?P<variant>[a-z0-9]+)__"
    r"(?P<speed>[a-z0-9]+)__"
    r"(?P<gas>\d+)LPM__PIV(?P<run>\d{2})\.xlsx$",
    re.IGNORECASE,
)

LEGACY_FILENAME_RE = re.compile(
    r"^(?P<date>\d{2}-\d{2}-\d{4})_(?P<sen>SEN\d{2})(?:_(?P<time>\d{3,4}))?"
    r"_(?P<speed>[\d\.]+(?:mpm)?)_(?P<gas>\d+)LPM_(?P<run>PIV\d+)\.xlsx$",
    re.IGNORECASE,
)

CANONICAL_COLUMNS = [
    "time[s]",
    "AN_1_LL[V]",
    "AN_2_LQ[V]",
    "AN_3_RQ[V]",
    "AN_4_RR[V]",
    "US_1_LL[V]",
    "US_2_LQ[V]",
    "US_3_RQ[V]",
    "US_4_RR[V]",
    "AN_1_LL[m/s]",
    "AN_2_LQ[m/s]",
    "AN_3_RQ[m/s]",
    "AN_4_RR[m/s]",
    "US_1_LL[mm]",
    "US_2_LQ[mm]",
    "US_3_RQ[mm]",
    "US_4_RR[mm]",
    "ML_LL[mm]",
    "ML_LQ[mm]",
    "ML_RQ[mm]",
    "ML_RR[mm]",
    "L_wave_ht[mm]",
    "R_wave_ht[mm]",
    "PIV_ON",
    "Dist from Sensor [mm]",
    "PIV Start Time [s]",
    "PIV Start Y [m/s]",
    "PIV End Time [s]",
    "PIV End Y [m/s]",
    "PIV RQ Time [s]",
    "PIV U RQ [m/s]",
    "PIV RR Time [s]",
    "PIV U RR [m/s]",
    "RQ Mean",
    "RQ SD",
    "RR Mean",
    "RR SD",
    "data_cleaned_ignore_time_geo_mld_Count_EX1_OneHot_Geometrical_Clogging_0Lag_Mould",
    "data_cleaned_ignore_time_geo_mld_Count_EX2_OneHot_Geometrical_Clogging_0Lag_Mould",
]

NUMERIC_COLUMNS = {
    column
    for column in CANONICAL_COLUMNS
    if column not in {"PIV_ON", "Dist from Sensor [mm]"}
}

ALIAS_MAP = {
    "RQ m": "RQ Mean",
    "RQ M": "RQ Mean",
    "RQM": "RQ Mean",
    "RQ Mean": "RQ Mean",
    "AN_RQ_Mean_Velocity [m/s]": "RQ Mean",
    "RQ SD": "RQ SD",
    "RQ STD": "RQ SD",
    "RQS": "RQ SD",
    "AN_RQ_Std_Dev [m/s]": "RQ SD",
    "RR m": "RR Mean",
    "RR M": "RR Mean",
    "RRM": "RR Mean",
    "RR Mean": "RR Mean",
    "AN_RR_Mean_Velocity [m/s]": "RR Mean",
    "RR SD": "RR SD",
    "RR STD": "RR SD",
    "RRS": "RR SD",
    "AN_RR_Std_Dev [m/s]": "RR SD",
    "y": "PIV Start Y [m/s]",
    "x": "PIV RQ Time [s]",
    "y.1": "PIV U RQ [m/s]",
    "x.1": "PIV RR Time [s]",
    "y.2": "PIV U RR [m/s]",
}

NUMERIC_NAME_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")


@dataclass
class WorkbookRecord:
    source_path: Path
    source_relative: Path
    sen: str
    variant: str
    speed_slug: str
    gas_lpm: int
    piv_run: int
    date_iso: str
    start_hhmm: str | None
    canonical_filename: str
    normalized_relative: Path
    issues: list[str] = field(default_factory=list)


@dataclass
class AlignmentRecord:
    sen: str
    variant: str
    speed_slug: str
    gas_lpm: int
    piv_run: int
    date_iso: str
    source_path: Path
    output_path: Path
    row_count: int
    alias_columns: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    all_null_columns: list[str] = field(default_factory=list)


SEN_PATTERN = re.compile(r"^(?P<sen>SEN\d+)(?:_(?P<variant>.+))?$", re.IGNORECASE)
SPEED_PATTERN = re.compile(r"^\d+(?:\.\d+)?mpm$", re.IGNORECASE)
GAS_RUN_PATTERN = re.compile(r"^\d+lpm\d+$", re.IGNORECASE)
GAS_PATTERN = re.compile(r"^\d+lpm$", re.IGNORECASE)
FRAME_NAME_PATTERN = re.compile(
    r"^(?P<stem>.+?)\.(?P<frame>\d{6})-(?P<time>\d+\.\d{4})\.csv$", re.IGNORECASE
)


@dataclass
class RunRecord:
    source_dir: Path
    source_relative: Path
    sen: str
    variant: str
    speed: str
    gas_run: str
    frame_files: list[Path]
    unmatched_files: list[Path] = field(default_factory=list)
    frame_count: int = 0
    first_frame: str = ""
    last_frame: str = ""
    grid_width: int | None = None
    grid_height: int | None = None
    originator: str | None = None
    column_header: Sequence[str] = field(default_factory=tuple)
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def normalized_subdir(self) -> Path:
        return Path(self.sen) / self.variant / self.speed / self.gas_run


@dataclass
class ExperimentRow:
    sen: str
    variant: str
    strand_speed_slug: str
    gas_flow_lpm: int
    piv_run: int
    normalized_dir: Path
    sensor_output_path: Path
    grid_width: int
    grid_height: int
    frame_count: int
    originator: str
    sensor_rows_full: int

    @property
    def group_path(self) -> str:
        return (
            f"runs/{self.sen}/{self.variant}/{self.strand_speed_slug}/"
            f"{self.gas_flow_lpm}lpm/PIV{self.piv_run:02d}"
        )


def load_manifest(path: Path) -> list[ExperimentRow]:
    df = pd.read_csv(path)
    included = df[df["status"] == "included"].copy()
    if included.empty:
        logger.warning("No runs marked as included in %s", path)
        return []

    rows: list[ExperimentRow] = []
    for item in included.itertuples(index=False):
        rows.append(
            ExperimentRow(
                sen=str(item.sen),
                variant=str(item.variant),
                strand_speed_slug=str(item.strand_speed_slug),
                gas_flow_lpm=int(item.gas_flow_lpm),
                piv_run=int(item.piv_run),
                normalized_dir=Path(str(item.normalized_dir)),
                sensor_output_path=Path(str(item.output_path)),
                grid_width=int(item.grid_width),
                grid_height=int(item.grid_height),
                frame_count=int(item.frame_count),
                originator=str(item.originator),
                sensor_rows_full=int(item.sensor_rows),
            )
        )
    return rows


def parse_timestamp(csv_path: Path) -> float:
    """Read the TimeStamp value from the CSV header."""
    pattern = re.compile(r"TimeStamp:\s*#\d+,\s*([+-]?\d+(?:\.\d+)?)")
    with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("TimeStamp:"):
                match = pattern.search(line)
                if not match:
                    raise ValueError(f"Unable to parse timestamp in {csv_path}")
                return float(match.group(1))
            if line.startswith(">>*DATA*<<"):
                break
    raise ValueError(f"No TimeStamp found in {csv_path}")


def load_piv_frames(
    csv_files: Iterable[Path],
    grid_height: int,
    grid_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u_frames: list[np.ndarray] = []
    v_frames: list[np.ndarray] = []
    times: list[float] = []
    x_axis: np.ndarray | None = None
    y_axis: np.ndarray | None = None

    for csv_file in csv_files:
        timestamp = parse_timestamp(csv_file)
        frame_df = pd.read_csv(csv_file, skiprows=8)
        if frame_df.empty:
            raise ValueError(f"{csv_file} has no frame data")
        u = (
            frame_df["U[m/s]"]
            .to_numpy(dtype=np.float32)
            .reshape(grid_height, grid_width)
        )
        v = (
            frame_df["V[m/s]"]
            .to_numpy(dtype=np.float32)
            .reshape(grid_height, grid_width)
        )
        if x_axis is None or y_axis is None:
            x = (
                frame_df["X (mm)[mm]"]
                .to_numpy(dtype=np.float32)
                .reshape(grid_height, grid_width)
            )
            y = (
                frame_df["Y (mm)[mm]"]
                .to_numpy(dtype=np.float32)
                .reshape(grid_height, grid_width)
            )
            x_axis = x[0, :]
            y_axis = y[:, 0]
        u_frames.append(u)
        v_frames.append(v)
        times.append(timestamp)

    return (
        np.stack(u_frames, axis=0),
        np.stack(v_frames, axis=0),
        np.asarray(times, dtype=np.float64),
        x_axis,
        y_axis,
    )


def load_sensor_window(sensor_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(sensor_path)
    if SENSOR_WINDOW_FLAG not in df.columns:
        raise ValueError(f"{sensor_path} missing '{SENSOR_WINDOW_FLAG}' column")
    piv_flag = df[SENSOR_WINDOW_FLAG].fillna(False).astype(bool)
    window = df[piv_flag].reset_index(drop=True)
    if window.empty:
        raise ValueError(f"{sensor_path} lacks any rows with {SENSOR_WINDOW_FLAG}=True")
    if SENSOR_TIME_COLUMN not in window.columns:
        raise ValueError(f"{sensor_path} missing '{SENSOR_TIME_COLUMN}' column")
    time_values = window[SENSOR_TIME_COLUMN].to_numpy(dtype=np.float64)
    if not np.all(np.diff(time_values) > 0):
        raise ValueError(f"{sensor_path} sensor times are not strictly increasing")
    return window, time_values


def align_sensor_to_frames(
    sensor_time: np.ndarray, frame_time: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sensor_rel = sensor_time - sensor_time[0]
    frame_rel = frame_time - frame_time[0]
    index_f = np.interp(
        frame_rel,
        sensor_rel,
        np.arange(sensor_rel.size, dtype=np.float64),
    )
    index = np.clip(np.rint(index_f).astype(np.int32), 0, sensor_rel.size - 1)
    return index, sensor_time[index]


def write_datasets(
    h5f: h5py.File,
    row: ExperimentRow,
    piv_root: Path,
    sensor_root: Path,
) -> None:
    csv_dir = piv_root / row.normalized_dir
    if not csv_dir.exists():
        raise FileNotFoundError(f"PIV directory missing: {csv_dir}")
    csv_files = sorted(csv_dir.glob("*.csv"))
    if len(csv_files) != row.frame_count:
        raise ValueError(
            f"{row.group_path}: expected {row.frame_count} frames but found {len(csv_files)}"
        )

    sensor_path = sensor_root / row.sensor_output_path
    sensor_df, sensor_time = load_sensor_window(sensor_path)
    sensor_numeric = sensor_df.apply(pd.to_numeric, errors="coerce")
    bubble_missing = [col for col in BUBBLE_COLUMNS if col not in sensor_df.columns]
    if bubble_missing:
        raise ValueError(f"{sensor_path} missing bubble prediction columns {bubble_missing}")

    u, v, frame_time, x_axis, y_axis = load_piv_frames(
        csv_files, row.grid_height, row.grid_width
    )
    if x_axis is None or y_axis is None:
        raise ValueError(f"{row.group_path}: missing spatial axes in PIV data")
    dx = float(np.mean(np.diff(x_axis)))
    dy = float(np.mean(np.diff(y_axis)))
    sensor_indices, sensor_time_for_frame = align_sensor_to_frames(sensor_time, frame_time)

    if row.group_path in h5f:
        del h5f[row.group_path]
    grp = h5f.create_group(row.group_path)

    piv_group = grp.create_group("piv")
    piv_group.create_dataset(
        "u",
        data=u.astype(np.float32),
        compression="gzip",
        compression_opts=4,
        chunks=(1, row.grid_height, row.grid_width),
    )
    piv_group.create_dataset(
        "v",
        data=v.astype(np.float32),
        compression="gzip",
        compression_opts=4,
        chunks=(1, row.grid_height, row.grid_width),
    )
    piv_group.create_dataset("time_s", data=frame_time.astype(np.float32), compression="gzip")
    piv_group.create_dataset("x_mm", data=x_axis.astype(np.float32))
    piv_group.create_dataset("y_mm", data=y_axis.astype(np.float32))

    sensor_group = grp.create_group("sensor")
    sensor_data = sensor_numeric.to_numpy(dtype=np.float32)
    sensor_group.create_dataset(
        "table",
        data=sensor_data,
        compression="gzip",
        compression_opts=4,
        chunks=(min(256, sensor_data.shape[0]), sensor_data.shape[1]),
    )
    sensor_group.create_dataset(
        "columns", data=np.asarray(sensor_df.columns, dtype="S")
    )
    sensor_group.create_dataset(
        "time_s",
        data=sensor_time.astype(np.float32),
        compression="gzip",
        compression_opts=4,
    )

    predictions_group = grp.create_group("predictions")
    predictions_group.create_dataset(
        "bubble_counts",
        data=sensor_numeric[BUBBLE_COLUMNS].to_numpy(dtype=np.float32),
        compression="gzip",
        compression_opts=4,
    )
    predictions_group.create_dataset(
        "columns",
        data=np.asarray(BUBBLE_COLUMNS, dtype="S"),
    )

    alignment_group = grp.create_group("aligned")
    alignment_group.create_dataset(
        "sensor_row_index_per_frame",
        data=sensor_indices,
        compression="gzip",
        compression_opts=4,
    )
    alignment_group.create_dataset(
        "sensor_time_for_frame_s",
        data=sensor_time_for_frame.astype(np.float32),
        compression="gzip",
        compression_opts=4,
    )

    grp.attrs.update(
        {
            "sen": row.sen,
            "variant": row.variant,
            "strand_speed_slug": row.strand_speed_slug,
            "gas_flow_lpm": row.gas_flow_lpm,
            "piv_run": row.piv_run,
            "grid_width": row.grid_width,
            "grid_height": row.grid_height,
            "dx_mm": dx,
            "dy_mm": dy,
            "n_frames": row.frame_count,
            "n_sensor_rows_full": row.sensor_rows_full,
            "n_sensor_rows_window": int(sensor_df.shape[0]),
            "sensor_time_start_s": float(sensor_time[0]),
            "sensor_time_end_s": float(sensor_time[-1]),
            "piv_time_start_s": float(frame_time[0]),
            "piv_time_end_s": float(frame_time[-1]),
            "originator": row.originator,
            "piv_directory": str(row.normalized_dir),
            "sensor_parquet": str(row.sensor_output_path),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
            Aggregate validated experiments into a single HDF5 archive. Only runs
            marked as 'included' in data/intermediate/experiments_manifest.csv are processed.
            """
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/intermediate/experiments_manifest.csv"),
        help="Path to the merged experiments manifest CSV.",
    )
    parser.add_argument(
        "--raw-piv-root",
        type=Path,
        default=Path("data/raw/piv_velocity_raw"),
        help="Directory containing raw PIV downloads.",
    )
    parser.add_argument(
        "--piv-root",
        type=Path,
        default=Path("data/intermediate/piv_velocity"),
        help="Root directory containing the normalized PIV directories.",
    )
    parser.add_argument(
        "--piv-manifest",
        type=Path,
        default=Path("data/intermediate/piv_velocity_manifest.csv"),
        help="Path to write/read the PIV manifest.",
    )
    parser.add_argument(
        "--raw-sensor-root",
        type=Path,
        default=Path("data/raw/sensor_logs"),
        help="Directory containing raw sensor workbooks.",
    )
    parser.add_argument(
        "--sensor-canonical-root",
        type=Path,
        default=Path("data/intermediate/sensor_logs_canonical"),
        help="Destination for canonical sensor symlinks.",
    )
    parser.add_argument(
        "--sensor-manifest",
        type=Path,
        default=Path("data/intermediate/sensor_logs_manifest.csv"),
        help="Metadata CSV for canonical sensor workbooks.",
    )
    parser.add_argument(
        "--sensor-root",
        type=Path,
        default=Path("data/intermediate/sensor_logs_aligned"),
        help="Root directory containing aligned sensor Parquet files.",
    )
    parser.add_argument(
        "--sensor-aligned-manifest",
        type=Path,
        default=Path("data/intermediate/sensor_logs_aligned_manifest.csv"),
        help="Metadata CSV describing aligned sensor outputs.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/processed/all_experiments.h5"),
        help="Destination HDF5 file (will be overwritten).",
    )
    parser.add_argument(
        "--coverage-dir",
        type=Path,
        default=Path("data/processed/coverage"),
        help="Directory to write coverage reports.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Delete the output file before processing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many new experiments (useful for chunked runs).",
    )
    parser.add_argument(
        "--rebuild-intermediate",
        action="store_true",
        help="Rebuild normalized/aligned data and manifests before writing HDF5.",
    )
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="Skip regenerating coverage reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rebuild = args.rebuild_intermediate

    normalize_sensor_logs(
        raw_root=args.raw_sensor_root,
        canonical_root=args.sensor_canonical_root,
        manifest_path=args.sensor_manifest,
        refresh=rebuild,
    )
    align_sensor_logs(
        canonical_root=args.sensor_canonical_root,
        output_root=args.sensor_root,
        manifest_path=args.sensor_aligned_manifest,
        refresh=rebuild,
    )
    normalize_piv_runs(
        raw_root=args.raw_piv_root,
        normalized_root=args.piv_root,
        manifest_path=args.piv_manifest,
        refresh=rebuild,
    )
    build_experiments_manifest(
        piv_manifest=args.piv_manifest,
        sensor_manifest=args.sensor_aligned_manifest,
        output=args.manifest,
    )
    if not args.skip_coverage:
        write_coverage_reports(
            manifest_path=args.manifest, coverage_dir=args.coverage_dir
        )

    rows = load_manifest(args.manifest)
    if not rows:
        logger.error("Nothing to process.")
        return

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.refresh and args.output_file.exists():
        args.output_file.unlink()

    file_mode = "a" if args.output_file.exists() else "w"
    with h5py.File(args.output_file, file_mode) as h5f:
        remaining_rows = [
            row for row in rows if row.group_path not in h5f
        ]
        if not remaining_rows:
            logger.info("All experiments already ingested into %s", args.output_file)
            return
        if args.limit is not None:
            remaining_rows = remaining_rows[: args.limit]
        logger.info(
            "Writing %d experiments to %s", len(remaining_rows), args.output_file,
        )
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Experiments", total=len(remaining_rows))
            for row in remaining_rows:
                try:
                    write_datasets(h5f, row, args.piv_root, args.sensor_root)
                except Exception as exc:
                    logger.error("Failed to process %s: %s", row.group_path, exc)
                    raise
                progress.advance(task)


def discover_sensor_workbooks(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in WORKBOOK_SUFFIXES:
            yield path


def clean_speed_slug(raw_speed: str) -> str:
    slug = raw_speed.lower().replace(".", "p")
    if not slug.endswith("mpm"):
        slug = f"{slug}mpm"
    return slug


def parse_canonical_sensor(path: Path) -> WorkbookRecord | None:
    match = CANONICAL_FILENAME_RE.match(path.name)
    if not match:
        return None
    date_iso = match.group("date")
    sen = match.group("sen").upper()
    variant = match.group("variant").lower()
    speed_slug = match.group("speed").lower()
    gas_lpm = int(match.group("gas"))
    piv_run = int(match.group("run"))
    canonical_filename = path.name
    return WorkbookRecord(
        source_path=path,
        source_relative=path,
        sen=sen,
        variant=variant,
        speed_slug=speed_slug,
        gas_lpm=gas_lpm,
        piv_run=piv_run,
        date_iso=date_iso,
        start_hhmm=None,
        canonical_filename=canonical_filename,
        normalized_relative=Path(sen) / variant / speed_slug / canonical_filename,
    )


def parse_legacy_sensor(path: Path, rel_parts: Sequence[str]) -> WorkbookRecord:
    match = LEGACY_FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unrecognised workbook filename: {path.name}")
    date_iso = datetime.strptime(match.group("date"), "%m-%d-%Y").date().isoformat()
    sen = match.group("sen").upper()
    start = match.group("time")
    speed_slug = clean_speed_slug(match.group("speed"))
    gas_lpm = int(match.group("gas"))
    piv_run = int(re.search(r"(\d+)", match.group("run")).group(1))
    variant = "baseline"
    if len(rel_parts) >= 2:
        candidate = rel_parts[1].lower()
        if candidate not in {"baseline", "raw"}:
            variant = candidate
    canonical_filename = (
        f"{date_iso}__{sen}__{variant}__{speed_slug}__{gas_lpm}LPM__PIV{piv_run:02d}.xlsx"
    )
    return WorkbookRecord(
        source_path=path,
        source_relative=path,
        sen=sen,
        variant=variant,
        speed_slug=speed_slug,
        gas_lpm=gas_lpm,
        piv_run=piv_run,
        date_iso=date_iso,
        start_hhmm=start,
        canonical_filename=canonical_filename,
        normalized_relative=Path(sen) / variant / speed_slug / canonical_filename,
        issues=["legacy_filename"],
    )


def resolve_sensor_record(raw_root: Path, workbook_path: Path) -> WorkbookRecord:
    rel = workbook_path.relative_to(raw_root)
    canonical = parse_canonical_sensor(workbook_path)
    if canonical:
        canonical.source_relative = rel
        canonical.normalized_relative = (
            Path(canonical.sen) / canonical.variant / canonical.speed_slug / canonical.canonical_filename
        )
        return canonical
    record = parse_legacy_sensor(workbook_path, rel.parts)
    record.source_relative = rel
    return record


def ensure_directory(dir_path: Path, refresh: bool) -> None:
    if dir_path.exists():
        if not refresh:
            return
        if dir_path.is_symlink():
            dir_path.unlink()
        else:
            shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def safe_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists():
        return
    rel_target = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(rel_target)


def write_sensor_manifest(manifest_path: Path, records: Sequence[WorkbookRecord]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sensor",
        "variant",
        "strand_speed_slug",
        "gas_flow_lpm",
        "piv_run",
        "date_iso",
        "start_time_hhmm",
        "source_path",
        "normalized_path",
        "issues",
    ]
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sensor": record.sen,
                    "variant": record.variant,
                    "strand_speed_slug": record.speed_slug,
                    "gas_flow_lpm": record.gas_lpm,
                    "piv_run": record.piv_run,
                    "date_iso": record.date_iso,
                    "start_time_hhmm": record.start_hhmm or "",
                    "source_path": str(record.source_relative),
                    "normalized_path": str(record.normalized_relative),
                    "issues": ";".join(record.issues),
                }
            )


def normalize_sensor_logs(
    raw_root: Path,
    canonical_root: Path,
    manifest_path: Path,
    refresh: bool,
) -> None:
    if canonical_root.exists() and manifest_path.exists() and not refresh:
        logger.info("Sensor logs already normalized; skipping.")
        return
    if not raw_root.exists():
        raise FileNotFoundError(f"Sensor raw root not found: {raw_root}")
    logger.info("Normalizing sensor logs from %s", raw_root)
    ensure_directory(canonical_root, refresh=refresh)
    records: list[WorkbookRecord] = []
    for workbook in discover_sensor_workbooks(raw_root):
        record = resolve_sensor_record(raw_root, workbook)
        target = canonical_root / record.normalized_relative
        safe_symlink(workbook, target)
        records.append(record)
    write_sensor_manifest(manifest_path, records)
def normalise_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    alias_hits: list[str] = []
    columns_original = [str(col) for col in df.columns]
    for original, target in ALIAS_MAP.items():
        if original in columns_original:
            alias_hits.append(f"{original}->{target}")

    df = df.rename(columns=lambda col: ALIAS_MAP.get(str(col), str(col)))

    drop_candidates = [
        col
        for col in df.columns
        if str(col).startswith("Unnamed:")
        or str(col).lower() == "nan"
        or NUMERIC_NAME_PATTERN.match(str(col))
    ]
    if drop_candidates:
        df = df.drop(columns=drop_candidates)

    aligned = pd.DataFrame(index=df.index)
    missing_columns: list[str] = []
    for column in CANONICAL_COLUMNS:
        matches = [c for c in df.columns if str(c) == column]
        if not matches:
            missing_columns.append(column)
            aligned[column] = pd.NA
            continue
        series = df[matches[0]]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        if column in NUMERIC_COLUMNS:
            aligned[column] = pd.to_numeric(series, errors="coerce")
        else:
            aligned[column] = series

    return aligned, alias_hits, drop_candidates, missing_columns


def align_sensor_logs(
    canonical_root: Path,
    output_root: Path,
    manifest_path: Path,
    refresh: bool,
) -> None:
    if output_root.exists() and manifest_path.exists() and not refresh:
        logger.info("Aligned sensor logs already exist; skipping.")
        return
    if not canonical_root.exists():
        raise FileNotFoundError(f"Canonical sensor root not found: {canonical_root}")
    logger.info("Aligning sensor logs from %s", canonical_root)
    ensure_directory(output_root, refresh=refresh)
    records: list[AlignmentRecord] = []
    for workbook in sorted(canonical_root.rglob("*.xlsx")):
        relative = workbook.relative_to(canonical_root)
        if len(relative.parts) < 4:
            logger.warning("Skipping workbook with unexpected layout: %s", workbook)
            continue
        sen, variant, speed_slug = relative.parts[:3]
        gas_match = re.search(r"(\d+)LPM", workbook.name, re.IGNORECASE)
        piv_match = re.search(r"PIV(\d+)", workbook.name, re.IGNORECASE)
        if not gas_match or not piv_match:
            logger.warning("Unable to parse gas/run from %s; skipping.", workbook.name)
            continue
        record = AlignmentRecord(
            sen=sen,
            variant=variant,
            speed_slug=speed_slug,
            gas_lpm=int(gas_match.group(1)),
            piv_run=int(piv_match.group(1)),
            date_iso=workbook.name.split("__")[0],
            source_path=workbook,
            output_path=Path(sen) / variant / speed_slug / relative.with_suffix(".parquet").name,
            row_count=0,
        )
        df = pd.read_excel(workbook)
        aligned_df, alias_hits, dropped_cols, missing_cols = normalise_columns(df)
        record.alias_columns = alias_hits
        record.dropped_columns = dropped_cols
        record.missing_columns = missing_cols
        record.row_count = len(aligned_df)
        output_path = output_root / record.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        aligned_df.to_parquet(output_path, index=False)
        records.append(record)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sen",
        "variant",
        "strand_speed_slug",
        "gas_flow_lpm",
        "piv_run",
        "date_iso",
        "rows",
        "source_path",
        "output_path",
        "alias_columns",
        "dropped_columns",
        "missing_columns",
        "all_null_columns",
    ]
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sen": record.sen,
                    "variant": record.variant,
                    "strand_speed_slug": record.speed_slug,
                    "gas_flow_lpm": record.gas_lpm,
                    "piv_run": record.piv_run,
                    "date_iso": record.date_iso,
                    "rows": record.row_count,
                    "source_path": str(record.source_path.relative_to(canonical_root)),
                    "output_path": str(record.output_path),
                    "alias_columns": ";".join(record.alias_columns),
                    "dropped_columns": ";".join(record.dropped_columns),
                    "missing_columns": ";".join(record.missing_columns),
                    "all_null_columns": "",
                }
            )
def find_run_directories(raw_root: Path) -> Iterable[Path]:
    for path in sorted(raw_root.rglob("*")):
        if not path.is_dir():
            continue
        try:
            files = [p for p in path.iterdir() if p.is_file()]
        except PermissionError:
            continue
        csvs = [p for p in files if p.suffix.lower() == ".csv"]
        if csvs:
            yield path


def extract_tokens(stem: str) -> list[str]:
    tokens = stem.split("_")
    if "Unaveraged" in tokens:
        return tokens[: tokens.index("Unaveraged")]
    return tokens


def infer_variant(top_dir: str, tokens: Sequence[str]) -> str:
    top_match = SEN_PATTERN.match(top_dir)
    if top_match and top_match.group("variant"):
        return top_match.group("variant").lower()
    if len(tokens) > 1 and not SPEED_PATTERN.match(tokens[1]):
        return tokens[1].lower()
    return "baseline"


def infer_speed(path_parts: Sequence[str], tokens: Sequence[str]) -> str:
    for part in path_parts:
        if SPEED_PATTERN.match(part):
            return part.lower()
    for token in tokens[1:]:
        if SPEED_PATTERN.match(token):
            return token.lower()
    raise ValueError("Unable to infer strand speed from path or filename tokens.")


def infer_gas_run(path_parts: Sequence[str]) -> tuple[str, bool]:
    for part in reversed(path_parts):
        normalized = part.lower().replace(" ", "")
        if GAS_RUN_PATTERN.match(normalized):
            return normalized, normalized != part.lower()
    raise ValueError("Unable to infer gas/run identifier from directory path.")


def read_frame_metadata(
    frame_path: Path,
) -> tuple[
    Sequence[str],
    int | None,
    int | None,
    str | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    headers: list[str] = []
    grid_width: int | None = None
    grid_height: int | None = None
    originator: str | None = None
    x_values: list[float] = []
    y_values: list[float] = []

    with frame_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("GridSize:"):
                match = re.search(r"Width=(\d+), Height=(\d+)", line)
                if match:
                    grid_width = int(match.group(1))
                    grid_height = int(match.group(2))
            elif line.startswith("Originator:"):
                originator = line.split(":", 1)[1]
            elif line.startswith(">>*DATA*<<"):
                break
        reader = csv.reader(fh)
        headers = next(reader, [])
        for row in reader:
            if len(row) < 4:
                continue
            try:
                x_values.append(float(row[2]))
                y_values.append(float(row[3]))
            except ValueError:
                continue

    x_min = min(x_values) if x_values else None
    x_max = max(x_values) if x_values else None
    y_min = min(y_values) if y_values else None
    y_max = max(y_values) if y_values else None
    return headers, grid_width, grid_height, originator, x_min, x_max, y_min, y_max


def describe_run(raw_root: Path, run_dir: Path) -> RunRecord | None:
    relative = run_dir.relative_to(raw_root)
    path_parts = tuple(part.lower() for part in relative.parts)
    candidates: list[tuple[Path, re.Match[str]]] = []
    unmatched: list[Path] = []
    for item in sorted(run_dir.iterdir()):
        if not item.is_file():
            continue
        if item.suffix.lower() != ".csv":
            continue
        match = FRAME_NAME_PATTERN.match(item.name)
        if match:
            candidates.append((item, match))
        else:
            unmatched.append(item)

    if not candidates:
        return None

    first_headers, grid_width, grid_height, originator, x_min, x_max, y_min, y_max = read_frame_metadata(
        candidates[0][0]
    )
    tokens = extract_tokens(candidates[0][0].stem)
    try:
        variant = infer_variant(relative.parts[0], tokens)
        speed = infer_speed(relative.parts, tokens)
        gas_run, normalised = infer_gas_run(relative.parts)
    except ValueError as exc:
        issues = [str(exc)]
        return RunRecord(
            source_dir=run_dir,
            source_relative=relative,
            sen=relative.parts[0].split("_")[0],
            variant="baseline",
            speed="unknown",
            gas_run="unknown",
            frame_files=[],
            unmatched_files=unmatched,
            issues=issues,
        )

    frame_files = [candidate[0] for candidate in candidates]
    frame_files.sort()
    first_frame = frame_files[0].name
    last_frame = frame_files[-1].name
    frame_count = len(frame_files)

    sen = relative.parts[0].split("_")[0].upper()
    return RunRecord(
        source_dir=run_dir,
        source_relative=relative,
        sen=sen,
        variant=variant,
        speed=speed,
        gas_run=gas_run,
        frame_files=frame_files,
        unmatched_files=unmatched,
        frame_count=frame_count,
        first_frame=first_frame,
        last_frame=last_frame,
        grid_width=grid_width,
        grid_height=grid_height,
        originator=originator,
        column_header=first_headers,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        issues=[],
    )


def normalize_piv_runs(
    raw_root: Path,
    normalized_root: Path,
    manifest_path: Path,
    refresh: bool,
) -> None:
    if normalized_root.exists() and manifest_path.exists() and not refresh:
        logger.info("PIV runs already normalized; skipping.")
        return
    if not raw_root.exists():
        raise FileNotFoundError(f"PIV raw root not found: {raw_root}")
    logger.info("Normalizing PIV runs from %s", raw_root)
    ensure_directory(normalized_root, refresh=refresh)
    records: list[RunRecord] = []
    for run_dir in find_run_directories(raw_root):
        record = describe_run(raw_root, run_dir)
        if record is None:
            continue
        records.append(record)
        target_dir = normalized_root / record.normalized_subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for frame_path in record.frame_files:
            link_path = target_dir / frame_path.name
            if not link_path.exists():
                rel_target = os.path.relpath(frame_path, start=link_path.parent)
                link_path.symlink_to(rel_target)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sen",
        "variant",
        "strand_speed",
        "gas_flow_run",
        "source_dir",
        "normalized_dir",
        "frame_count",
        "first_frame",
        "last_frame",
        "grid_width",
        "grid_height",
        "originator",
        "column_header",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "issues",
    ]
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sen": record.sen,
                    "variant": record.variant,
                    "strand_speed": record.speed,
                    "gas_flow_run": record.gas_run,
                    "source_dir": str(record.source_relative),
                    "normalized_dir": str(record.normalized_subdir),
                    "frame_count": record.frame_count,
                    "first_frame": record.first_frame,
                    "last_frame": record.last_frame,
                    "grid_width": record.grid_width,
                    "grid_height": record.grid_height,
                    "originator": record.originator or "",
                    "column_header": "|".join(record.column_header),
                    "x_min": record.x_min or "",
                    "x_max": record.x_max or "",
                    "y_min": record.y_min or "",
                    "y_max": record.y_max or "",
                    "issues": ";".join(record.issues),
                }
            )
def strand_speed_to_slug(value: str) -> str:
    text = value.strip().lower().replace(" ", "")
    if text.endswith("mpm"):
        text = text[:-3]
    text = text.replace(".", "p")
    if not text.endswith("mpm"):
        text = f"{text}mpm"
    return text


def parse_gas_flow_run(value: str) -> tuple[int, int]:
    text = value.strip().lower().replace(" ", "")
    if "lpm" not in text:
        raise ValueError(f"Unrecognised gas/run token '{value}'")
    gas, run = text.split("lpm", 1)
    gas = gas or "0"
    run = run or "1"
    return int(gas), int(run)


def split_semicolon_list(value: str | float | int | None) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def build_experiments_manifest(
    piv_manifest: Path, sensor_manifest: Path, output: Path
) -> None:
    piv_df = pd.read_csv(piv_manifest)
    sensor_df = pd.read_csv(sensor_manifest)

    piv_df = piv_df.copy()
    piv_df["strand_speed_slug"] = piv_df["strand_speed"].apply(strand_speed_to_slug)
    piv_df["gas_flow_lpm"], piv_df["piv_run"] = zip(
        *piv_df["gas_flow_run"].apply(parse_gas_flow_run)
    )
    piv_df["variant"] = piv_df["variant"].str.lower()
    piv_df["sen"] = piv_df["sen"].str.upper()

    sensor_df = sensor_df.rename(columns={"sensor": "sen"}).copy()
    sensor_df["variant"] = sensor_df["variant"].str.lower()
    sensor_df["sen"] = sensor_df["sen"].str.upper()

    merged = piv_df.merge(
        sensor_df,
        how="inner",
        left_on=["sen", "variant", "strand_speed_slug", "gas_flow_lpm", "piv_run"],
        right_on=["sen", "variant", "strand_speed_slug", "gas_flow_lpm", "piv_run"],
        suffixes=("_piv", "_sensor"),
    )

    statuses: list[str] = []
    reasons: list[str] = []
    parsed_missing = merged["missing_columns"].apply(split_semicolon_list)
    sensor_rows = merged["rows"].fillna(0).astype(int)

    for idx, row in merged.iterrows():
        row_reasons: list[str] = []
        if row["frame_count"] != 999:
            row_reasons.append(f"frame_count={row['frame_count']}")
        issues_raw = row.get("issues", "")
        if isinstance(issues_raw, float) and pd.isna(issues_raw):
            issues = ""
        else:
            issues = str(issues_raw).strip()
        if issues:
            row_reasons.append(f"piv_issues={issues}")
        if sensor_rows.iloc[idx] <= 0:
            row_reasons.append("no_sensor_rows")
        missing_cols = set(parsed_missing.iloc[idx])
        missing_required = sorted(
            col
            for col in BUBBLE_COLUMNS
            if col in missing_cols
        )
        if missing_required:
            row_reasons.append(
                "missing_sensor_columns=" + ",".join(missing_required)
            )
        status = "included" if not row_reasons else "excluded"
        statuses.append(status)
        reasons.append(";".join(row_reasons))

    merged["status"] = statuses
    merged["exclusion_reason"] = reasons
    merged["sensor_rows"] = sensor_rows
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)


def write_coverage_reports(manifest_path: Path, coverage_dir: Path) -> None:
    df = pd.read_csv(manifest_path)
    coverage_dir.mkdir(parents=True, exist_ok=True)
    for sen in sorted(df["sen"].unique()):
        subset = df[df["sen"] == sen]
        included = subset[subset["status"] == "included"]
        excluded = subset[subset["status"] != "included"]
        output_path = coverage_dir / f"{sen}_data_coverage.md"
        with output_path.open("w", encoding="utf-8") as fh:
            fh.write(f"# {sen} Data Coverage\n")
            fh.write(
                "Experiments with all validated data sources (PIV frames, aligned sensor traces, bubble predictions).\n"
            )
            if included.empty:
                fh.write("\n_No experiments currently included._\n")
            else:
                fh.write("\n## Complete Experiments\n")
                fh.write(
                    "| Strand Speed | Gas Flow | Run | Sensor Workbook | PIV Directory |\n"
                )
                fh.write("|--------------|----------|-----|------------------|---------------|\n")
                for _, row in included.iterrows():
                    fh.write(
                        f"| {row['strand_speed_slug']} | {row['gas_flow_lpm']} LPM | "
                        f"PIV{int(row['piv_run']):02d} | `{row['output_path']}` | "
                        f"`{row['normalized_dir']}` |\n"
                    )
            if not excluded.empty:
                fh.write("\n## Excluded Experiments\n")
                fh.write("Runs filtered out with their exclusion reasons.\n\n")
                fh.write(
                    "| Strand Speed | Gas Flow | Run | Sensor Workbook | PIV Directory | Reason |\n"
                )
                fh.write("|--------------|----------|-----|------------------|---------------|--------|\n")
                for _, row in excluded.iterrows():
                    fh.write(
                        f"| {row['strand_speed_slug']} | {row['gas_flow_lpm']} LPM | "
                        f"PIV{int(row['piv_run']):02d} | `{row['output_path']}` | "
                        f"`{row['normalized_dir']}` | {row['exclusion_reason'] or ''} |\n"
                    )


if __name__ == "__main__":
    main()
