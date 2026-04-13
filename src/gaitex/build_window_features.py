from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import pandas as pd

from gaitex.features import compute_rep_features, detect_quaternion_groups, extract_quat
from gaitex.io import iter_available_tasks, load_imu_csv, load_timestamps_csv
from gaitex.segment import cut_repetitions


logger = logging.getLogger("gaitex.project32.windows")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build window-level GAITEX segment features for Project 3.2."
    )
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default="output/features")
    p.add_argument("--tasks", nargs="+", default=["rd", "rgs"])
    p.add_argument("--window_s", type=float, default=0.50)
    p.add_argument("--step_s", type=float, default=0.10)
    p.add_argument("--min_samples", type=int, default=20)
    p.add_argument("--max_subjects", type=int, default=None)
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def iter_windows(
    t: np.ndarray,
    df: pd.DataFrame,
    window_s: float,
    step_s: float,
    min_samples: int,
) -> Iterator[Dict[str, Any]]:
    if t.size == 0:
        return

    start = float(t[0])
    end = float(t[-1])
    cur = start
    window_idx = 0
    total_duration = max(end - start, 1e-12)

    while cur + window_s <= end + 1e-12:
        mask = (t >= cur) & (t < cur + window_s)
        idx = np.where(mask)[0]
        if idx.size >= min_samples:
            yield {
                "window_id": window_idx,
                "t_start": float(cur),
                "t_end": float(cur + window_s),
                "window_progress": float((cur + 0.5 * window_s - start) / total_duration),
                "t_window": t[idx],
                "df_window": df.iloc[idx].reset_index(drop=True),
            }
        cur += step_s
        window_idx += 1


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    n_subjects = 0

    for tf in iter_available_tasks(data_root=data_root, tasks=args.tasks):
        if args.max_subjects is not None and n_subjects >= args.max_subjects:
            break

        logger.info("Subject=%s Task=%s", tf.subject, tf.task)
        t, imu_df = load_imu_csv(tf.imu_csv)
        reps = load_timestamps_csv(tf.timestamps_csv)

        for rep in cut_repetitions(t=t, imu_df=imu_df, reps=reps, min_samples=args.min_samples):
            groups = detect_quaternion_groups(list(rep["imu_segment_df"].columns))
            if not groups:
                continue

            rep_duration = float(rep["t_end"] - rep["t_start"])
            rep_total_windows = 0

            rep_windows = list(
                iter_windows(
                    t=np.asarray(rep["t_segment"], dtype=float),
                    df=rep["imu_segment_df"],
                    window_s=args.window_s,
                    step_s=args.step_s,
                    min_samples=args.min_samples,
                )
            )
            rep_total_windows = len(rep_windows)
            if rep_total_windows == 0:
                continue

            for win in rep_windows:
                for g in groups:
                    try:
                        q = extract_quat(win["df_window"], g)
                        feats = compute_rep_features(q=q, t=np.asarray(win["t_window"], dtype=float))
                    except Exception:
                        continue

                    rows.append(
                        {
                            "subject": tf.subject,
                            "task": tf.task,
                            "label": rep["label"],
                            "rep_id": rep["rep_id"],
                            "segment": g.name,
                            "rep_duration_s": rep_duration,
                            "rep_total_windows": rep_total_windows,
                            "window_id": win["window_id"],
                            "window_t_start": win["t_start"],
                            "window_t_end": win["t_end"],
                            "window_progress": win["window_progress"],
                            **feats,
                        }
                    )

        n_subjects += 1

    if not rows:
        raise SystemExit("No window features produced.")

    out_csv = out_dir / "window_features.csv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d rows)", out_csv, len(df_out))


if __name__ == "__main__":
    main()
