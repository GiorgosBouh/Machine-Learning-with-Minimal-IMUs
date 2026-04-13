from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = ["mean_speed", "rms_speed", "peak_speed", "rms_accel", "rot_range"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate binary correct-vs-variant performance across sensor subsets."
    )
    p.add_argument("--features_csv", default="output/features/features.csv")
    p.add_argument("--tasks", nargs="+", default=["rd", "rgs"])
    p.add_argument("--max_segments", type=int, default=4)
    p.add_argument("--top_tolerance", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="output/features")
    p.add_argument("--fig_dir", default="output/figures")
    return p.parse_args()


def load_features(path: str, feature_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"subject", "task", "label", "segment"} | set(feature_cols)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in features CSV: {missing}")
    return df


def task_segments(df: pd.DataFrame, task: str) -> list[str]:
    segs = sorted(df.loc[df["task"].astype(str).str.lower() == task.lower(), "segment"].astype(str).unique().tolist())
    if not segs:
        raise ValueError(f"No segments found for task={task}")
    return segs


def lower_body_segments(segments: list[str]) -> list[str]:
    keep_tokens = ("Foot", "LowerLeg", "UpperLeg", "Pelvis")
    return [seg for seg in segments if any(tok in seg for tok in keep_tokens)]


def build_subject_level_binary_table(
    df: pd.DataFrame,
    task: str,
    segments: Iterable[str],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    d = df[df["task"].astype(str).str.lower() == task.lower()].copy()
    d = d[d["segment"].isin(list(segments))].copy()
    if d.empty:
        return pd.DataFrame(), []

    g = (
        d.groupby(["subject", "label", "segment"], as_index=False)[feature_cols]
        .mean()
    )

    wide = g.pivot_table(
        index=["subject", "label"],
        columns="segment",
        values=feature_cols,
        aggfunc="mean",
    )

    flat_cols = []
    for col in wide.columns:
        feat, seg = col
        flat_cols.append(f"{seg}__{feat}")
    wide.columns = flat_cols
    wide = wide.reset_index()

    correct_label = f"{task.lower()}_correct"
    wide["y"] = (wide["label"].astype(str) != correct_label).astype(int)
    wide = wide.drop(columns=["label"])

    model_cols = [c for c in wide.columns if c not in {"subject", "y"} and wide[c].notna().any()]
    return wide, model_cols


def loso_binary_metrics(
    wide: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> dict[str, float]:
    n_subjects = int(wide["subject"].nunique()) if not wide.empty else 0
    if wide.empty or not feature_cols:
        return {
            "accuracy_mean": float("nan"),
            "accuracy_std": float("nan"),
            "balanced_accuracy_mean": float("nan"),
            "balanced_accuracy_std": float("nan"),
            "n_folds": 0,
            "n_subjects": n_subjects,
            "n_rows": int(len(wide)),
        }

    X = wide[feature_cols].to_numpy(dtype=float)
    y = wide["y"].to_numpy(dtype=int)
    groups = wide["subject"].astype(str).to_numpy()

    if n_subjects < 2:
        return {
            "accuracy_mean": float("nan"),
            "accuracy_std": float("nan"),
            "balanced_accuracy_mean": float("nan"),
            "balanced_accuracy_std": float("nan"),
            "n_folds": 0,
            "n_subjects": n_subjects,
            "n_rows": int(len(wide)),
        }

    logo = LeaveOneGroupOut()

    accs: list[float] = []
    baccs: list[float] = []
    for tr, te in logo.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr = X[tr]
        Xte = X[te]

        valid_cols = np.isfinite(Xtr).any(axis=0)
        if not np.any(valid_cols):
            continue

        Xtr = Xtr[:, valid_cols]
        Xte = Xte[:, valid_cols]

        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, solver="liblinear", random_state=seed)),
            ]
        )

        pipe.fit(Xtr, y[tr])
        pred = pipe.predict(Xte)
        accs.append(accuracy_score(y[te], pred))
        baccs.append(balanced_accuracy_score(y[te], pred))

    if not accs:
        return {
            "accuracy_mean": float("nan"),
            "accuracy_std": float("nan"),
            "balanced_accuracy_mean": float("nan"),
            "balanced_accuracy_std": float("nan"),
            "n_folds": 0,
            "n_subjects": int(np.unique(groups).size),
            "n_rows": int(len(wide)),
        }

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "balanced_accuracy_mean": float(np.mean(baccs)),
        "balanced_accuracy_std": float(np.std(baccs, ddof=1)) if len(baccs) > 1 else 0.0,
        "n_folds": int(len(accs)),
        "n_subjects": int(np.unique(groups).size),
        "n_rows": int(len(wide)),
    }


def evaluate_frontier(
    df: pd.DataFrame,
    task: str,
    feature_cols: list[str],
    max_segments: int,
    seed: int,
    search_space_name: str,
    candidate_segments: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    max_k = min(max_segments, len(candidate_segments))
    for k in range(1, max_k + 1):
        for subset in itertools.combinations(candidate_segments, k):
            wide, cols = build_subject_level_binary_table(df, task, subset, feature_cols)
            metrics = loso_binary_metrics(wide, cols, seed=seed)
            rows.append(
                {
                    "task": task,
                    "search_space": search_space_name,
                    "subset_type": f"k={k}",
                    "n_segments": k,
                    "segments": "|".join(subset),
                    "n_feature_cols": len(cols),
                    **metrics,
                }
            )

    wide_all, cols_all = build_subject_level_binary_table(df, task, candidate_segments, feature_cols)
    metrics_all = loso_binary_metrics(wide_all, cols_all, seed=seed)
    rows.append(
        {
            "task": task,
            "search_space": search_space_name,
            "subset_type": "ALL",
            "n_segments": len(candidate_segments),
            "segments": "|".join(candidate_segments),
            "n_feature_cols": len(cols_all),
            **metrics_all,
        }
    )

    return pd.DataFrame(rows)


def best_by_count(frontier: pd.DataFrame) -> pd.DataFrame:
    if frontier.empty:
        return frontier.copy()
    order = frontier.sort_values(
        ["task", "search_space", "n_segments", "balanced_accuracy_mean", "accuracy_mean", "n_feature_cols"],
        ascending=[True, True, True, False, False, True],
    )
    best = order.groupby(["task", "search_space", "n_segments"], as_index=False).head(1).reset_index(drop=True)
    return best


def stability_summary(frontier: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frontier.empty:
        return pd.DataFrame(rows)

    for (task, search_space, n_segments), dfg in frontier[frontier["subset_type"] != "ALL"].groupby(["task", "search_space", "n_segments"]):
        best_bacc = float(dfg["balanced_accuracy_mean"].max())
        keep = dfg[dfg["balanced_accuracy_mean"] >= best_bacc - tolerance].copy()
        total = int(len(keep))
        counts: dict[str, int] = {}
        for segs in keep["segments"]:
            for seg in str(segs).split("|"):
                counts[seg] = counts.get(seg, 0) + 1

        for seg, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            rows.append(
                {
                    "task": task,
                    "search_space": search_space,
                    "n_segments": int(n_segments),
                    "segment": seg,
                    "selected_subsets": count,
                    "candidate_subsets": total,
                    "inclusion_rate": float(count / total) if total else float("nan"),
                    "best_balanced_accuracy_mean": best_bacc,
                    "tolerance": tolerance,
                }
            )
    return pd.DataFrame(rows)


def plot_frontier(best: pd.DataFrame, out_path: Path) -> None:
    if best.empty:
        return

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for (task, search_space), dft in best.groupby(["task", "search_space"]):
        dft = dft.sort_values("n_segments")
        label = f"{task.upper()} ({search_space.replace('_', ' ')})"
        ax.errorbar(
            dft["n_segments"],
            dft["balanced_accuracy_mean"],
            yerr=dft["balanced_accuracy_std"],
            marker="o",
            capsize=3,
            linewidth=1.6,
            label=label,
        )

    ax.set_xlabel("Number of sensors / body segments")
    ax.set_ylabel("Binary LOSO balanced accuracy (mean ± SD)")
    ax.set_title("Sensor frontier: best subset at each sensor count")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_features(args.features_csv, DEFAULT_FEATURES)

    all_frontiers = []
    for task in args.tasks:
        all_segments = task_segments(df, task)
        spaces = {
            "all_segments": all_segments,
            "lower_body_only": lower_body_segments(all_segments),
        }

        for search_space_name, candidate_segments in spaces.items():
            if not candidate_segments:
                continue
            frontier = evaluate_frontier(
                df=df,
                task=task,
                feature_cols=DEFAULT_FEATURES,
                max_segments=args.max_segments,
                seed=args.seed,
                search_space_name=search_space_name,
                candidate_segments=candidate_segments,
            )
            all_frontiers.append(frontier)

    frontier_all = pd.concat(all_frontiers, ignore_index=True)
    best = best_by_count(frontier_all)
    stability = stability_summary(frontier_all, tolerance=args.top_tolerance)

    frontier_csv = out_dir / "sensor_frontier_all_subsets.csv"
    best_csv = out_dir / "sensor_frontier_best_by_count.csv"
    stability_csv = out_dir / "sensor_frontier_stability.csv"
    frontier_all.to_csv(frontier_csv, index=False)
    best.to_csv(best_csv, index=False)
    stability.to_csv(stability_csv, index=False)

    plot_frontier(best, fig_dir / "sensor_frontier_accuracy.png")

    print(f"Wrote: {frontier_csv}")
    print(f"Wrote: {best_csv}")
    print(f"Wrote: {stability_csv}")
    print(f"Wrote: {fig_dir / 'sensor_frontier_accuracy.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
