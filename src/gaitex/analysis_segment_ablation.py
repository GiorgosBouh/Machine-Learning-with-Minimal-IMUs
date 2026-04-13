from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("gaitex.ablation")
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger


FEATURES = ["rms_speed", "peak_speed", "rms_accel", "mean_speed", "rot_range"]


def subject_level_table(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Mean over repetitions: one row per (subject, label, segment)."""
    d = df[df["task"].str.lower() == task.lower()].copy()
    g = d.groupby(["subject", "label", "segment"], as_index=False)[FEATURES].mean()
    return g


def make_wide(g: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per (subject, label), columns = segment__feature."""
    X = g.pivot_table(index=["subject", "label"], columns="segment", values=FEATURES, aggfunc="mean")
    X.columns = [f"{seg}__{feat}" for feat, seg in X.columns]
    X = X.reset_index()
    return X


def pick_columns_by_segments(X: pd.DataFrame, segments: list[str]) -> list[str]:
    cols = []
    for c in X.columns:
        if "__" in c:
            seg, feat = c.split("__", 1)
            if seg in segments and feat in FEATURES:
                cols.append(c)
    return cols


def infer_segments_present(X: pd.DataFrame) -> list[str]:
    segs = set()
    for c in X.columns:
        if "__" in c:
            seg = c.split("__", 1)[0]
            segs.add(seg)
    return sorted(segs)


def loso_binary_accuracy(X: pd.DataFrame, feature_cols: list[str], task: str) -> dict:
    """
    Binary: correct vs variant.
    Groups = subject (LOSO).
    """
    y = X["label"].astype(str)
    correct_label = f"{task.lower()}_correct"
    y_bin = (y != correct_label).astype(int).to_numpy()  # 0 correct, 1 variant
    groups = X["subject"].astype(str).to_numpy()

    Xnum = X[feature_cols].to_numpy(dtype=float).copy()
    # Fill NaNs with column means
    col_means = np.nanmean(Xnum, axis=0)
    inds = np.where(np.isnan(Xnum))
    if inds[0].size > 0:
        Xnum[inds] = np.take(col_means, inds[1])

    logo = LeaveOneGroupOut()
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]
    )

    accs = []
    baccs = []
    for train_idx, test_idx in logo.split(Xnum, y_bin, groups=groups):
        pipe.fit(Xnum[train_idx], y_bin[train_idx])
        pred = pipe.predict(Xnum[test_idx])
        accs.append(accuracy_score(y_bin[test_idx], pred))
        baccs.append(balanced_accuracy_score(y_bin[test_idx], pred))

    return {
        "n_subjects": int(len(np.unique(groups))),
        "n_samples": int(len(y_bin)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "balanced_accuracy_mean": float(np.mean(baccs)),
        "balanced_accuracy_std": float(np.std(baccs, ddof=1)) if len(baccs) > 1 else 0.0,
    }


def main() -> int:
    logger = _setup_logger()

    features_csv = Path("output") / "features" / "features.csv"
    out_csv = Path("output") / "features" / "segment_ablation_results.csv"
    out_fig = Path("output") / "figures" / "segment_ablation_accuracy.png"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)
    # Normalize task strings
    df["task"] = df["task"].astype(str)

    results = []

    for task in ["rd", "rgs"]:
        logger.info(f"Task={task}: preparing subject-level table")
        g = subject_level_table(df, task)
        X = make_wide(g)

        segments_all = infer_segments_present(X)

        # Lower-limb heuristic segments (try XSens naming from your data)
        lower_limb = [s for s in segments_all if ("Foot" in s) or ("LowerLeg" in s) or ("UpperLeg" in s)]
        trunk = [s for s in segments_all if ("Pelvis" in s) or ("Sternum" in s)]

        if task == "rd":
            task_relevant = [s for s in segments_all if ("Foot" in s) or ("LowerLeg" in s) or ("Sternum" in s) or ("Pelvis" in s)]
        else:
            task_relevant = [s for s in segments_all if ("Pelvis" in s) or ("UpperLeg" in s) or ("LowerLeg" in s) or ("Sternum" in s)]

        sets = {
            "ALL": segments_all,
            "LOWER_LIMB_ONLY": lower_limb,
            "TASK_RELEVANT": task_relevant,
        }

        for set_name, segs in sets.items():
            cols = pick_columns_by_segments(X, segs)
            if len(cols) < 2:
                logger.warning(f"Skipping {task} {set_name}: not enough columns ({len(cols)})")
                continue

            metrics = loso_binary_accuracy(X, cols, task)
            row = {
                "task": task,
                "segment_set": set_name,
                "n_segments": int(len(segs)),
                "n_feature_cols": int(len(cols)),
                **metrics,
            }
            results.append(row)
            logger.info(f"{task.upper()} | {set_name} | acc={metrics['accuracy_mean']:.3f} bacc={metrics['balanced_accuracy_mean']:.3f}")

    res_df = pd.DataFrame(results).sort_values(["task", "segment_set"])
    res_df.to_csv(out_csv, index=False)
    logger.info(f"Wrote: {out_csv}")

    # Plot accuracy means
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, task in enumerate(["rd", "rgs"]):
        sub = res_df[res_df["task"] == task]
        x = np.arange(len(sub))
        ax.errorbar(
            x + (i * 0.06),
            sub["accuracy_mean"],
            yerr=sub["accuracy_std"],
            fmt="o",
            label=task.upper(),
            capsize=3,
        )
    ax.set_xticks(np.arange(len(res_df[res_df["task"] == "rd"])))
    ax.set_xticklabels(res_df[res_df["task"] == "rd"]["segment_set"].tolist(), rotation=15, ha="right")
    ax.set_ylabel("LOSO accuracy (mean ± SD)")
    ax.set_title("Segment ablation: correct vs variant (subject-level features)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)
    logger.info(f"Wrote: {out_fig}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
