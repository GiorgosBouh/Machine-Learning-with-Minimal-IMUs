from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
FEATURES_REP = ROOT / "output" / "features" / "features.csv"
OUT_DIR_FEATURES = ROOT / "output" / "features"
OUT_DIR_FIG = ROOT / "output" / "figures"

FEATURES = ["mean_speed", "rms_speed", "peak_speed", "rms_accel", "rot_range"]
TASK_MINIMAL_SEGMENT = {
    "rd": "XSens_LowerLeg_Right",
    "rgs": "XSens_LowerLeg_Left",
}


@dataclass
class Setting:
    name: str
    segments: list[str]
    features: list[str]


def load_features_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"subject", "task", "label", "rep_id", "segment", *FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"features.csv missing columns: {missing}")
    return df


def subject_level_wide(
    df: pd.DataFrame,
    task: str,
    labels_keep: list[str],
    segments: list[str],
    features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    d = df[(df["task"] == task) & (df["label"].isin(labels_keep))].copy()
    d = d[d["segment"].isin(segments)].copy()
    if d.empty:
        return pd.DataFrame(), []

    grouped = d.groupby(["subject", "label", "segment"], as_index=False)[features].mean()
    wide = grouped.pivot_table(
        index=["subject", "label"],
        columns="segment",
        values=features,
        aggfunc="mean",
    )
    wide.columns = [f"{seg}__{feat}" for feat, seg in wide.columns]
    wide = wide.reset_index()
    wide["y"] = (wide["label"] != f"{task}_correct").astype(int)
    wide = wide.drop(columns=["label"])
    feature_cols = [c for c in wide.columns if c not in {"subject", "y"}]
    return wide, feature_cols


def loso_accuracy(wide: pd.DataFrame, feature_cols: list[str], seed: int = 0) -> tuple[float, float, np.ndarray]:
    if wide.empty or not feature_cols:
        return float("nan"), float("nan"), np.array([])

    X = wide[feature_cols].to_numpy(dtype=float)
    y = wide["y"].to_numpy(dtype=int)
    subjects = wide["subject"].astype(str).to_numpy()
    pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, solver="liblinear", random_state=seed)),
        ]
    )

    fold_acc = []
    for subject in np.unique(subjects):
        te = subjects == subject
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        fold_acc.append(float((pred == y[te]).mean()))

    fold_acc = np.asarray(fold_acc, dtype=float)
    if fold_acc.size == 0:
        return float("nan"), float("nan"), fold_acc
    return float(fold_acc.mean()), float(fold_acc.std(ddof=1) if fold_acc.size > 1 else 0.0), fold_acc


def permutation_test_delta(
    wide_min: pd.DataFrame,
    cols_min: list[str],
    wide_all: pd.DataFrame,
    cols_all: list[str],
    n_perm: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    subs = sorted(set(wide_min["subject"].unique()) & set(wide_all["subject"].unique()))
    if len(subs) < 2:
        return {"delta_true": float("nan"), "null_mean": float("nan"), "p_value": float("nan"), "n_subjects": 0.0}

    wm = wide_min[wide_min["subject"].isin(subs)].reset_index(drop=True)
    wa = wide_all[wide_all["subject"].isin(subs)].reset_index(drop=True)

    acc_m, _, _ = loso_accuracy(wm, cols_min, seed=seed)
    acc_a, _, _ = loso_accuracy(wa, cols_all, seed=seed)
    delta_true = acc_a - acc_m

    bm = [np.where(wm["subject"].values == s)[0] for s in subs]
    ba = [np.where(wa["subject"].values == s)[0] for s in subs]
    ym_blocks = [wm.loc[idx, "y"].to_numpy() for idx in bm]
    ya_blocks = [wa.loc[idx, "y"].to_numpy() for idx in ba]

    null = []
    for _ in range(n_perm):
        perm = rng.permutation(len(subs))
        y_m_perm = wm["y"].to_numpy().copy()
        y_a_perm = wa["y"].to_numpy().copy()
        for i, j in enumerate(perm):
            y_m_perm[bm[i]] = ym_blocks[j]
            y_a_perm[ba[i]] = ya_blocks[j]

        wm2 = wm.copy()
        wa2 = wa.copy()
        wm2["y"] = y_m_perm
        wa2["y"] = y_a_perm

        acc_m_p, _, _ = loso_accuracy(wm2, cols_min, seed=seed)
        acc_a_p, _, _ = loso_accuracy(wa2, cols_all, seed=seed)
        null.append(acc_a_p - acc_m_p)

    null_arr = np.asarray(null, dtype=float)
    p_value = float((np.sum(null_arr >= delta_true) + 1) / (n_perm + 1))
    return {
        "delta_true": float(delta_true),
        "null_mean": float(np.nanmean(null_arr)),
        "p_value": p_value,
        "n_subjects": float(len(subs)),
    }


def save_barplot(results: pd.DataFrame, out_png: Path) -> None:
    tasks = results["task"].unique().tolist()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(tasks))
    width = 0.35

    r_min = results[results["setting"] == "MINIMAL"].set_index("task").reindex(tasks)
    r_all = results[results["setting"] == "ALL"].set_index("task").reindex(tasks)

    ax.bar(x - width / 2, r_min["accuracy_mean"].values, width, yerr=r_min["accuracy_std"].values, capsize=4, label="Minimal")
    ax.bar(x + width / 2, r_all["accuracy_mean"].values, width, yerr=r_all["accuracy_std"].values, capsize=4, label="All segments")

    ax.set_xticks(x)
    ax.set_xticklabels([task.upper() for task in tasks])
    ax.set_ylabel("LOSO accuracy (mean +- SD)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    OUT_DIR_FEATURES.mkdir(parents=True, exist_ok=True)
    OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)
    df = load_features_csv(FEATURES_REP)

    results = []
    pair_rows = []

    for task in ["rd", "rgs"]:
        task_df = df[df["task"] == task].copy()
        labels = sorted(task_df["label"].astype(str).unique().tolist())
        task_segments = sorted(task_df["segment"].astype(str).unique().tolist())

        all_setting = Setting("ALL", task_segments, FEATURES)
        min_setting = Setting("MINIMAL", [TASK_MINIMAL_SEGMENT[task]], FEATURES)

        wide_all, cols_all = subject_level_wide(task_df, task, labels, all_setting.segments, all_setting.features)
        wide_min, cols_min = subject_level_wide(task_df, task, labels, min_setting.segments, min_setting.features)

        for setting_name, wide, cols in [
            (all_setting.name, wide_all, cols_all),
            (min_setting.name, wide_min, cols_min),
        ]:
            acc_mean, acc_std, fold_acc = loso_accuracy(wide, cols)
            results.append(
                {
                    "task": task,
                    "setting": setting_name,
                    "accuracy_mean": acc_mean,
                    "accuracy_std": acc_std,
                    "n_subjects": int(fold_acc.size),
                    "n_features": len(cols),
                }
            )
            for subject, acc in zip(sorted(wide["subject"].unique().tolist()), fold_acc):
                pair_rows.append(
                    {
                        "task": task,
                        "setting": setting_name,
                        "subject": subject,
                        "accuracy": float(acc),
                    }
                )

        perm = permutation_test_delta(wide_min, cols_min, wide_all, cols_all, n_perm=500, seed=0)
        print(
            f"{task.upper()} | ALL acc={results[-2]['accuracy_mean']:.3f} | "
            f"MIN acc={results[-1]['accuracy_mean']:.3f} | "
            f"delta={perm['delta_true']:.3f} | p={perm['p_value']:.4f}"
        )

    results_df = pd.DataFrame(results)
    folds_df = pd.DataFrame(pair_rows)

    folds_wide = folds_df.pivot_table(index=["task", "subject"], columns="setting", values="accuracy", aggfunc="mean").reset_index()
    folds_wide = folds_wide.rename(columns={"ALL": "acc_all", "MINIMAL": "acc_minimal"})
    folds_wide["delta_all_minus_min"] = folds_wide["acc_all"] - folds_wide["acc_minimal"]

    results_path = OUT_DIR_FEATURES / "task_complexity_results.csv"
    folds_path = OUT_DIR_FEATURES / "task_complexity_folds_all_vs_minimal.csv"
    fig_path = OUT_DIR_FIG / "task_complexity_results.png"

    results_df.to_csv(results_path, index=False)
    folds_wide.to_csv(folds_path, index=False)
    save_barplot(results_df, fig_path)

    print(f"Wrote: {results_path}")
    print(f"Wrote: {folds_path}")
    print(f"Wrote: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
