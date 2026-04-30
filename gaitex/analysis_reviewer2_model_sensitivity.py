from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


FEATURES = ["mean_speed", "rms_speed", "peak_speed", "rms_accel", "rot_range"]
COMMON_COMPACT = ["XSens_Hand_Right", "XSens_LowerLeg_Left"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reviewer 2 model-family sensitivity analysis.")
    p.add_argument("--features_csv", default="output/features/features.csv")
    p.add_argument("--out_csv", default="output/features/reviewer2_model_subset_checks.csv")
    return p.parse_args()


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    alt = Path("..") / p
    if alt.exists():
        return alt
    if p.exists():
        return p
    return p


def build_subject_level_binary_table(
    df: pd.DataFrame,
    task: str,
    segments: list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    d = df[df["task"].astype(str).str.lower() == task.lower()].copy()
    if segments is not None:
        d = d[d["segment"].isin(segments)].copy()

    grouped = d.groupby(["subject", "label", "segment"], as_index=False)[FEATURES].mean()
    wide = grouped.pivot_table(
        index=["subject", "label"],
        columns="segment",
        values=FEATURES,
        aggfunc="mean",
    )
    wide.columns = [f"{seg}__{feat}" for feat, seg in wide.columns]
    wide = wide.reset_index()
    wide["y"] = (wide["label"].astype(str) != f"{task.lower()}_correct").astype(int)
    model_cols = [c for c in wide.columns if c not in {"subject", "label", "y"} and wide[c].notna().any()]
    return wide, model_cols


def loso_metrics(wide: pd.DataFrame, model_cols: list[str], pipe: Pipeline) -> dict[str, float]:
    X = wide[model_cols].to_numpy(dtype=float)
    y = wide["y"].to_numpy(dtype=int)
    groups = wide["subject"].astype(str).to_numpy()
    logo = LeaveOneGroupOut()
    accs: list[float] = []
    baccs: list[float] = []

    for tr, te in logo.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr = X[tr]
        Xte = X[te]
        valid_cols = np.isfinite(Xtr).any(axis=0)
        Xtr = Xtr[:, valid_cols]
        Xte = Xte[:, valid_cols]
        pipe.fit(Xtr, y[tr])
        pred = pipe.predict(Xte)
        accs.append(float(accuracy_score(y[te], pred)))
        baccs.append(float(balanced_accuracy_score(y[te], pred)))

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_sd": float(np.std(accs, ddof=1)),
        "balanced_accuracy_mean": float(np.mean(baccs)),
        "balanced_accuracy_sd": float(np.std(baccs, ddof=1)),
    }


def main() -> int:
    args = parse_args()
    features_path = resolve_path(args.features_csv)
    df = pd.read_csv(features_path)
    out_rows: list[dict[str, object]] = []

    models: dict[str, Pipeline] = {
        "LogisticRegression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, solver="liblinear", random_state=0)),
            ]
        ),
        "DecisionTree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", DecisionTreeClassifier(max_depth=3, random_state=0)),
            ]
        ),
        "GaussianNB": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", GaussianNB()),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=300, max_depth=4, random_state=0)),
            ]
        ),
    }

    for task in ["rd", "rgs"]:
        for cfg_name, segments in [("full", None), ("common_compact", COMMON_COMPACT)]:
            wide, model_cols = build_subject_level_binary_table(df, task=task, segments=segments)
            for model_name, pipe in models.items():
                metrics = loso_metrics(wide, model_cols, pipe)
                out_rows.append(
                    {
                        "task": task,
                        "configuration": cfg_name,
                        "model": model_name,
                        "n_features": len(model_cols),
                        **metrics,
                    }
                )

    out = pd.DataFrame(out_rows)
    out_path = resolve_path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
