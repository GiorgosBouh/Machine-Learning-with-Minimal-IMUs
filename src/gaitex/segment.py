# src/gaitex/segment.py
from __future__ import annotations

from typing import Dict, Any, Iterator, List

import numpy as np
import pandas as pd


def cut_repetitions(
    t: pd.Series,
    imu_df: pd.DataFrame,
    reps: List[Dict[str, Any]],
    min_samples: int = 10,
) -> Iterator[Dict[str, Any]]:
    """
    Cuts IMU dataframe into repetition windows based on reps list.

    Inputs:
      t: time series (seconds), length N
      imu_df: IMU data columns, length N
      reps: list of dicts with keys: label, t_start, t_end, rep_id

    Yields dict with:
      label, rep_id, t_start, t_end, t_segment (np.ndarray), imu_segment_df (pd.DataFrame)
    """
    if len(t) != len(imu_df):
        raise ValueError(f"t and imu_df length mismatch: {len(t)} vs {len(imu_df)}")

    t_np = np.asarray(t, dtype=float)

    for rep in reps:
        t_start = float(rep["t_start"])
        t_end = float(rep["t_end"])
        label = str(rep.get("label", "unknown"))
        rep_id = int(rep.get("rep_id", -1))

        # Boolean mask for window
        mask = (t_np >= t_start) & (t_np <= t_end)
        idx = np.where(mask)[0]
        if idx.size < min_samples:
            continue

        seg_t = t_np[idx]
        seg_df = imu_df.iloc[idx].reset_index(drop=True)

        yield {
            "label": label,
            "rep_id": rep_id,
            "t_start": t_start,
            "t_end": t_end,
            "t_segment": seg_t,
            "imu_segment_df": seg_df,
        }
