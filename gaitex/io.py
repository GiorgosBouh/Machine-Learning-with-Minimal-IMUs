
# src/gaitex/io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, List, Dict, Any

import pandas as pd


@dataclass(frozen=True)
class TaskFiles:
    subject: str
    task: str
    imu_csv: Path
    timestamps_csv: Path


def iter_subject_dirs(data_root: Path) -> Iterator[Path]:
    data_root = Path(data_root)
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            yield p


def _find_one(patterns: List[str], folder: Path) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def iter_available_tasks(data_root: Path, tasks: Iterable[str] = ("rd", "rgs")) -> Iterator[TaskFiles]:
    """
    Expected structure:
      data_root/<subject>/<task>/xsens_imu_data_segment_registered_<subject>_<task>.csv
      data_root/<subject>/<task>/timestamps_<subject>_<task>.csv

    Robust via globbing: will accept any xsens_imu_data_segment_registered_*.csv and timestamps_*.csv
    inside each subject/task folder.
    """
    data_root = Path(data_root)
    for subj_dir in iter_subject_dirs(data_root):
        subject = subj_dir.name
        for task in tasks:
            task_dir = subj_dir / task
            if not task_dir.exists():
                continue

            imu = _find_one(
                [
                    f"xsens_imu_data_segment_registered_{subject}_{task}.csv",
                    "xsens_imu_data_segment_registered_*.csv",
                ],
                task_dir,
            )
            ts = _find_one(
                [
                    f"timestamps_{subject}_{task}.csv",
                    "timestamps_*.csv",
                ],
                task_dir,
            )

            if imu is None or ts is None:
                continue

            yield TaskFiles(subject=subject, task=task, imu_csv=imu, timestamps_csv=ts)


def find_time_column(columns: Iterable[str]) -> str:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    candidates = [
        "time",
        "timestamp",
        "t",
        "seconds",
        "sec",
        "time_s",
        "time_sec",
    ]
    for k in candidates:
        if k in lower:
            return lower[k]

    # fallback: first numeric-looking column name often is time; but we cannot know types yet.
    # so just take first column
    return cols[0]


def load_imu_csv(path: Path) -> tuple[pd.Series, pd.DataFrame]:
    """
    Loads IMU segment-registered CSV.
    Returns:
      t: pandas Series of time (float)
      df: DataFrame of the remaining columns (sensor quaternions etc.)
    """
    path = Path(path)
    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError(f"IMU CSV has too few columns: {path}")

    time_col = find_time_column(df.columns)
    t = pd.to_numeric(df[time_col], errors="coerce")

    # Drop rows with missing time
    keep = t.notna()
    t = t.loc[keep].astype(float).reset_index(drop=True)

    df2 = df.loc[keep].reset_index(drop=True).copy()
    df2 = df2.drop(columns=[time_col])

    return t, df2


def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def load_timestamps_csv(path: Path) -> List[Dict[str, Any]]:
    """
    Loads timestamps CSV and returns a list of repetitions:
      [{"label": str, "t_start": float, "t_end": float, "rep_id": int}, ...]
    Robust to column naming differences.
    """
    path = Path(path)
    ts = pd.read_csv(path)
    if ts.empty:
        return []

    # Common possibilities in datasets
    label_col = _pick_first_existing(ts, ["label", "variant", "class", "name", "exercise", "movement"])
    start_col = _pick_first_existing(ts, ["start", "start_time", "t_start", "begin", "onset"])
    end_col = _pick_first_existing(ts, ["end", "end_time", "t_end", "stop", "offset"])

    # Sometimes they use "start [s]" / "end [s]" etc.
    if start_col is None:
        for c in ts.columns:
            if "start" in c.lower():
                start_col = c
                break
    if end_col is None:
        for c in ts.columns:
            if "end" in c.lower() or "stop" in c.lower():
                end_col = c
                break
    if label_col is None:
        for c in ts.columns:
            if "label" in c.lower() or "variant" in c.lower():
                label_col = c
                break

    if start_col is None or end_col is None:
        raise ValueError(f"Could not detect start/end columns in timestamps file: {path}. Columns: {list(ts.columns)}")

    # If no label column, create a default label
    if label_col is None:
        label_col = "__label__"
        ts[label_col] = "unknown"

    starts = pd.to_numeric(ts[start_col], errors="coerce")
    ends = pd.to_numeric(ts[end_col], errors="coerce")
    labels = ts[label_col].astype(str)

    reps: List[Dict[str, Any]] = []
    rep_id = 0
    for lab, s, e in zip(labels, starts, ends):
        if pd.isna(s) or pd.isna(e):
            continue
        s = float(s)
        e = float(e)
        if e <= s:
            continue
        reps.append({"label": lab, "t_start": s, "t_end": e, "rep_id": rep_id})
        rep_id += 1

    return reps
