# src/gaitex/analysis_feature_relevance.py
"""
GAITEX #5 — Task-specific feature relevance (reviewer-safe, fixed feature space).

Goal
-----
Estimate which segments/features matter for separating "correct" vs "variant"
movement quality, per task (RD, RGS), using only simple rotational features.

Approach
--------
1) Aggregate repetition-level features to subject-level:
   subject × label × segment → mean(features)

2) Build a fixed wide table:
   subject × (segment__feature) with one row per (subject, class)
   class = 0 (correct), 1 (variant)

3) Handle missing segment/feature entries robustly:
   - Remove columns that are all-NaN across the entire dataset
   - Replace remaining NaNs with 0.0 ONCE, before CV (fixed feature space)

4) Leave-One-Subject-Out (LOSO) logistic regression:
   - Standardize features in each fold
   - Fit LR, collect coefficients
   - Report mean and std of coefficients across folds

Outputs
-------
- output/features/feature_relevance_coeffs.csv
- output/figures/feature_relevance_rd.png
- output/figures/feature_relevance_rgs.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


IN_CSV = Path("output/features/features.csv")
OUT_CSV = Path("output/features/feature_relevance_coeffs.csv")
FIG_DIR = Path("output/figures")

FIG_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["rd", "rgs"]
FEATURES = ["mean_speed", "rms_speed", "peak_speed", "rms_accel", "rot_range"]


def prepare_subject_table(df_task: pd.DataFrame) -> pd.DataFrame:
    """
    Subject-level aggregation:
      subject × label × segment → mean(features)
    """
    return (
        df_task
        .groupby(["subject", "label", "segment"], as_index=False)[FEATURES]
        .mean()
    )


def _find_correct_label(labels: np.ndarray) -> str:
    candidates = [l for l in labels if "correct" in str(l).lower()]
    if not candidates:
        raise ValueError("Could not find a 'correct' label in labels.")
    return sorted(candidates, key=lambda s: len(str(s)))[0]


def run_feature_relevance(df_subj: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Binary classification (correct vs variant) with LOSO.
    Returns long table: task, segment, feature, coef_mean, coef_std, n_folds, n_subjects.
    """
    correct_label = _find_correct_label(df_subj["label"].unique())

    df_bin = df_subj.copy()
    df_bin["y"] = (df_bin["label"] != correct_label).astype(int)

    # Wide pivot: index=(subject, y), columns=(feature, segment)
    Xw = df_bin.pivot_table(
        index=["subject", "y"],
        columns="segment",
        values=FEATURES,
        aggfunc="mean",
    )

    # Flatten columns: "SEG__FEAT"
    Xw.columns = [f"{seg}__{feat}" for feat, seg in Xw.columns]
    Xw = Xw.reset_index()

    # Keep fixed feature space:
    meta_cols = ["subject", "y"]
    feat_cols = [c for c in Xw.columns if c not in meta_cols]

    # Drop columns that are all-missing across the entire dataset
    keep = [c for c in feat_cols if Xw[c].notna().any()]
    Xw = Xw[meta_cols + keep]

    if len(keep) < 5:
        raise ValueError(f"{task}: too few non-missing features after filtering ({len(keep)})")

    # Build matrices
    y = Xw["y"].to_numpy()
    groups = Xw["subject"].to_numpy()

    Xmat = Xw.drop(columns=meta_cols).to_numpy(dtype=float)
    feat_names = Xw.drop(columns=meta_cols).columns.tolist()

    # Manual imputation ONCE to guarantee same dimensions in all folds
    Xmat = np.nan_to_num(Xmat, nan=0.0, posinf=0.0, neginf=0.0)

    logo = LeaveOneGroupOut()

    coefs = []
    for tr, te in logo.split(Xmat, y, groups):
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=5000)),
            ]
        )
        clf.fit(Xmat[tr], y[tr])
        beta = clf.named_steps["lr"].coef_[0]
        coefs.append(beta)

    coefs = np.vstack(coefs)
    mean_beta = np.mean(coefs, axis=0)
    std_beta = np.std(coefs, axis=0, ddof=1)

    rows = []
    for name, m, s in zip(feat_names, mean_beta, std_beta):
        seg, feat = name.split("__", 1)
        rows.append(
            {
                "task": task,
                "segment": seg,
                "feature": feat,
                "coef_mean": float(m),
                "coef_std": float(s),
                "n_folds": int(coefs.shape[0]),
                "n_subjects": int(len(np.unique(groups))),
            }
        )

    return pd.DataFrame(rows)


def plot_feature_relevance(df_rel: pd.DataFrame, task: str) -> None:
    """
    Plot mean |coef| per segment (averaged across features).
    """
    g = (
        df_rel.assign(abs_coef=lambda d: d["coef_mean"].abs())
        .groupby("segment", as_index=False)["abs_coef"]
        .mean()
        .sort_values("abs_coef", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(g["segment"], g["abs_coef"])
    ax.set_xlabel("Mean |standardized coefficient| (LOSO-averaged)")
    ax.set_title(f"{task.upper()} – Feature relevance by segment")
    ax.invert_yaxis()

    out = FIG_DIR / f"feature_relevance_{task}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out}")


def main() -> int:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Could not find {IN_CSV}. Run build_features first.")

    df = pd.read_csv(IN_CSV)

    all_rows = []
    for task in TASKS:
        df_task = df[df["task"] == task].copy()
        if df_task.empty:
            print(f"{task.upper()} | no rows found, skipping.")
            continue

        df_subj = prepare_subject_table(df_task)
        rel = run_feature_relevance(df_subj, task)
        all_rows.append(rel)
        plot_feature_relevance(rel, task)

    out = pd.concat(all_rows, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

