# src/gaitex/build_features.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Dict, Any

import pandas as pd

from gaitex.io import iter_available_tasks, load_imu_csv, load_timestamps_csv
from gaitex.segment import cut_repetitions
from gaitex.features import compute_features_for_all_segments
from gaitex.qc_plots import plot_qc


logger = logging.getLogger("gaitex")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build GAITEX segment-level movement quality features (CLI)."
    )
    p.add_argument("--data_root", required=True, help="Path to GAITEX data root (contains subject folders).")
    p.add_argument("--out_dir", required=True, help="Output directory for features.")
    p.add_argument("--tasks", nargs="+", default=["rd", "rgs"], help="Tasks to process, e.g., rd rgs.")
    p.add_argument("--max_subjects", type=int, default=None, help="Optional: limit number of subjects for quick tests.")
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_rows: List[Dict[str, Any]] = []
    durations_rows: List[Dict[str, Any]] = []

    logger.info("Scanning data_root=%s for tasks=%s", data_root, args.tasks)

    n_subjects = 0
    for tf in iter_available_tasks(data_root=data_root, tasks=args.tasks):
        if args.max_subjects is not None and n_subjects >= args.max_subjects:
            break

        logger.info("Subject=%s Task=%s", tf.subject, tf.task)

        # Load data
        t, imu_df = load_imu_csv(tf.imu_csv)
        reps = load_timestamps_csv(tf.timestamps_csv)

        # Cut repetitions and compute features
        for rep in cut_repetitions(t=t, imu_df=imu_df, reps=reps):
            durations_rows.append(
                {
                    "subject": tf.subject,
                    "task": tf.task,
                    "label": rep["label"],
                    "rep_id": rep["rep_id"],
                    "duration_s": float(rep["t_end"] - rep["t_start"]),
                }
            )

            rep_features = compute_features_for_all_segments(
                t=rep["t_segment"],
                df=rep["imu_segment_df"],
                subject=tf.subject,
                task=tf.task,
                label=rep["label"],
                rep_id=rep["rep_id"],
            )
            features_rows.extend(rep_features)

        n_subjects += 1

    if not features_rows:
        logger.error("No features were produced. Check that timestamps and IMU CSVs are found and parsed.")
        return

    features_df = pd.DataFrame(features_rows)
    durations_df = pd.DataFrame(durations_rows)

    features_csv = out_dir / "features.csv"
    features_df.to_csv(features_csv, index=False)
    logger.info("Wrote %s (%d rows)", features_csv, len(features_df))

    # Optional parquet if available
    try:
        features_parquet = out_dir / "features.parquet"
        features_df.to_parquet(features_parquet, index=False)
        logger.info("Wrote %s", features_parquet)
    except Exception as e:
        logger.warning("Parquet not written (missing engine). CSV is fine. (%s)", e)

    # QC plots
    figures_dir = out_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        plot_qc(features_df=features_df, durations_df=durations_df, out_dir=figures_dir)
        logger.info("Wrote QC figures to %s", figures_dir)
    except Exception as e:
        logger.warning("QC plots failed (not fatal): %s", e)


if __name__ == "__main__":
    main()
