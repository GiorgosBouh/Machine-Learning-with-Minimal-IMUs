from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


IN_CSV = Path("output/features/features.csv")
OUT_DIR = Path("output/features")

TASKS = ["rd", "rgs"]
FEATURE_COLS = ["mean_speed", "rms_speed", "peak_speed", "rms_accel", "rot_range"]


def make_subject_table(df_task: pd.DataFrame) -> pd.DataFrame:
    return df_task.groupby(["subject", "label"], as_index=False)[FEATURE_COLS].mean()


def run_loso_multiclass(df_subj: pd.DataFrame, task: str) -> pd.DataFrame:
    X = df_subj[FEATURE_COLS].to_numpy(dtype=float)
    y = df_subj["label"].astype(str).to_numpy()
    groups = df_subj["subject"].astype(str).to_numpy()

    logo = LeaveOneGroupOut()
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000)),
        ]
    )

    fold_rows: list[dict[str, float | int | str]] = []
    y_true_all: list[str] = []
    y_pred_all: list[str] = []

    for fold_idx, (tr, te) in enumerate(logo.split(X, y, groups=groups), start=1):
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[te])
        y_true = y[te]

        fold_rows.append(
            {
                "task": task,
                "fold": fold_idx,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            }
        )

        y_true_all.extend(list(y_true))
        y_pred_all.extend(list(y_pred))

    labels_sorted = sorted(np.unique(y_true_all))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels_sorted)
    rep = classification_report(
        y_true_all,
        y_pred_all,
        labels=labels_sorted,
        output_dict=True,
        zero_division=0,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true:{label}" for label in labels_sorted],
        columns=[f"pred:{label}" for label in labels_sorted],
    )
    cm_path = OUT_DIR / f"classifier_confusion_{task}.csv"
    cm_df.to_csv(cm_path)

    rep_rows = []
    for label, stats in rep.items():
        if isinstance(stats, dict) and all(metric in stats for metric in ["precision", "recall", "f1-score", "support"]):
            rep_rows.append(
                {
                    "task": task,
                    "label": label,
                    "precision": stats["precision"],
                    "recall": stats["recall"],
                    "f1": stats["f1-score"],
                    "support": stats["support"],
                }
            )

    rep_df = pd.DataFrame(rep_rows).sort_values(["task", "label"])
    rep_path = OUT_DIR / f"classifier_report_{task}.csv"
    rep_df.to_csv(rep_path, index=False)

    print(f"{task.upper()} | saved confusion: {cm_path}")
    print(f"{task.upper()} | saved per-class report: {rep_path}")
    return pd.DataFrame(fold_rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)

    needed = {"subject", "task", "label", "segment"} | set(FEATURE_COLS)
    missing = sorted(needed - set(df.columns))
    if missing:
        raise SystemExit(f"Missing columns in features.csv: {missing}")

    all_folds = []
    for task in TASKS:
        df_task = df[df["task"] == task].copy()
        if df_task.empty:
            print(f"WARNING: no rows for task={task}")
            continue

        df_subj = make_subject_table(df_task)
        if df_subj["label"].nunique() < 2:
            print(f"WARNING: not enough labels for task={task}")
            continue

        folds = run_loso_multiclass(df_subj, task)
        all_folds.append(folds)
        print(
            f"{task.upper()} | LOSO folds: "
            f"acc={folds['accuracy'].mean():.3f}+-{folds['accuracy'].std(ddof=1):.3f} | "
            f"bacc={folds['balanced_accuracy'].mean():.3f}+-{folds['balanced_accuracy'].std(ddof=1):.3f} | "
            f"macroF1={folds['macro_f1'].mean():.3f}+-{folds['macro_f1'].std(ddof=1):.3f}"
        )

    if all_folds:
        out = pd.concat(all_folds, ignore_index=True)
        out_path = OUT_DIR / "classifier_loso_folds.csv"
        out.to_csv(out_path, index=False)
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
