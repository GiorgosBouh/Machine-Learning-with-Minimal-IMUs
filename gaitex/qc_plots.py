# src/gaitex/qc_plots.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_qc(features_df: pd.DataFrame, durations_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Durations histogram
    if not durations_df.empty and "duration_s" in durations_df.columns:
        plt.figure()
        durations_df["duration_s"].dropna().hist(bins=30)
        plt.xlabel("Duration (s)")
        plt.ylabel("Count")
        plt.title("Repetition durations")
        plt.tight_layout()
        plt.savefig(out_dir / "durations_hist.png", dpi=150)
        plt.close()

    # 2) Boxplots for selected segments + 2 features
    if features_df.empty:
        return

    # pick two likely segment name substrings (best-effort)
    seg_candidates = ["lower_leg_right", "foot_right"]
    feat_candidates = ["rms_speed", "mean_speed"]

    df = features_df.copy()
    df["segment_l"] = df["segment"].astype(str).str.lower()

    sel = df[df["segment_l"].apply(lambda s: any(c in s for c in seg_candidates))]
    sel = sel.dropna(subset=feat_candidates, how="all")
    if sel.empty:
        return

    # one figure: boxplot of rms_speed by label for selected segments
    plt.figure(figsize=(10, 4))
    # build data arrays per label
    labels = sorted(sel["label"].astype(str).unique())
    data = [sel.loc[sel["label"] == lab, "rms_speed"].dropna().values for lab in labels]
    # filter empty groups
    pairs = [(lab, arr) for lab, arr in zip(labels, data) if len(arr) > 0]
    if not pairs:
        return
    labels_f, data_f = zip(*pairs)

    plt.boxplot(data_f, labels=labels_f, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("RMS angular speed (rad/s)")
    plt.title("RMS speed by execution variant (selected right-leg segments)")
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_speed_by_variant.png", dpi=150)
    plt.close()
