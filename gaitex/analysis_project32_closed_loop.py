from __future__ import annotations

import argparse
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


FEATURES = ["mean_speed", "rms_speed", "peak_speed", "rms_accel", "rot_range"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project 3.2 closed-loop analyses.")
    p.add_argument("--window_features_csv", default="output/features/window_features.csv")
    p.add_argument("--out_dir", default="output/features")
    p.add_argument("--fig_dir", default="output/figures")
    p.add_argument("--tasks", nargs="+", default=["rd", "rgs"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--feedback_prob_threshold", type=float, default=0.6)
    p.add_argument("--feedback_persistence", type=int, default=3)
    p.add_argument("--feedback_score_threshold", type=float, default=0.0)
    return p.parse_args()


def wide_window_table(
    df: pd.DataFrame,
    task: str,
    segments: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    d = df[df["task"].astype(str).str.lower() == task.lower()].copy()
    if segments is not None:
        d = d[d["segment"].isin(list(segments))].copy()
    if d.empty:
        return pd.DataFrame(), []

    idx_cols = ["subject", "task", "label", "rep_id", "window_id", "window_progress", "rep_total_windows"]
    wide = d.pivot_table(
        index=idx_cols,
        columns="segment",
        values=FEATURES,
        aggfunc="mean",
    )
    flat_cols = []
    for feat, seg in wide.columns:
        flat_cols.append(f"{seg}__{feat}")
    wide.columns = flat_cols
    wide = wide.reset_index()
    correct_label = f"{task.lower()}_correct"
    wide["y"] = (wide["label"].astype(str) != correct_label).astype(int)
    model_cols = [c for c in wide.columns if "__" in c and wide[c].notna().any()]
    return wide, model_cols


def loso_window_predictions(wide: pd.DataFrame, model_cols: list[str], seed: int) -> pd.DataFrame:
    if wide.empty or not model_cols:
        return pd.DataFrame()

    X = wide[model_cols].to_numpy(dtype=float)
    y = wide["y"].to_numpy(dtype=int)
    groups = wide["subject"].astype(str).to_numpy()
    logo = LeaveOneGroupOut()

    rows: list[dict[str, object]] = []
    for tr, te in logo.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue

        Xtr = X[tr]
        Xte = X[te]
        valid_cols = np.isfinite(Xtr).any(axis=0)
        if not np.any(valid_cols):
            continue

        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, solver="liblinear", random_state=seed)),
            ]
        )
        pipe.fit(Xtr[:, valid_cols], y[tr])
        pred = pipe.predict(Xte[:, valid_cols])
        prob = pipe.predict_proba(Xte[:, valid_cols])[:, 1]

        meta = wide.iloc[te].reset_index(drop=True)
        for i in range(len(meta)):
            rows.append(
                {
                    **meta.iloc[i].to_dict(),
                    "y_pred": int(pred[i]),
                    "y_prob": float(prob[i]),
                }
            )

    return pd.DataFrame(rows)


def window_metrics(pred_df: pd.DataFrame, task: str) -> dict[str, object]:
    d = pred_df[pred_df["task"] == task].copy()
    return {
        "task": task,
        "window_accuracy": float(accuracy_score(d["y"], d["y_pred"])) if not d.empty else float("nan"),
        "window_balanced_accuracy": float(balanced_accuracy_score(d["y"], d["y_pred"])) if not d.empty else float("nan"),
        "n_windows": int(len(d)),
        "n_subjects": int(d["subject"].nunique()) if not d.empty else 0,
    }


def early_detection_summary(pred_df: pd.DataFrame, task: str) -> pd.DataFrame:
    d = pred_df[(pred_df["task"] == task) & (pred_df["y"] == 1)].copy()
    rows: list[dict[str, object]] = []
    for (subject, label, rep_id), g in d.groupby(["subject", "label", "rep_id"]):
        g = g.sort_values("window_id")
        hit = g[g["y_pred"] == 1]
        rows.append(
            {
                "task": task,
                "subject": subject,
                "label": label,
                "rep_id": int(rep_id),
                "detected": int(not hit.empty),
                "first_detect_window": int(hit["window_id"].iloc[0]) if not hit.empty else -1,
                "first_detect_progress": float(hit["window_progress"].iloc[0]) if not hit.empty else float("nan"),
                "n_windows": int(g["rep_total_windows"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def correct_reference(window_df: pd.DataFrame) -> pd.DataFrame:
    correct_rows = []
    for task, dft in window_df.groupby("task"):
        correct_label = f"{task}_correct"
        d = dft[dft["label"] == correct_label].copy()
        if d.empty:
            continue
        grouped = (
            d.groupby(["task", "segment"], as_index=False)[FEATURES]
            .agg(["mean", "std"])
        )
        grouped.columns = ["task", "segment"] + [f"{feat}_{stat}" for feat, stat in grouped.columns.tolist()[2:]]
        correct_rows.append(grouped)
    return pd.concat(correct_rows, ignore_index=True) if correct_rows else pd.DataFrame()


def build_explanations(window_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    ref = correct_reference(window_df)
    if ref.empty:
        return pd.DataFrame()

    d = window_df.merge(
        pred_df[["subject", "task", "label", "rep_id", "window_id", "y_pred", "y_prob"]],
        on=["subject", "task", "label", "rep_id", "window_id"],
        how="inner",
    )
    d = d[d["y_pred"] == 1].copy()
    d = d.merge(ref, on=["task", "segment"], how="left")

    rows: list[dict[str, object]] = []
    for _, r in d.iterrows():
        best_feature = None
        best_score = -np.inf
        best_delta = float("nan")
        for feat in FEATURES:
            mu = float(r.get(f"{feat}_mean", np.nan))
            sd = float(r.get(f"{feat}_std", np.nan))
            val = float(r[feat])
            delta = val - mu
            score = abs(delta / sd) if np.isfinite(sd) and sd > 1e-12 else abs(delta)
            if score > best_score:
                best_score = score
                best_feature = feat
                best_delta = delta

        rows.append(
            {
                "subject": r["subject"],
                "task": r["task"],
                "label": r["label"],
                "rep_id": int(r["rep_id"]),
                "window_id": int(r["window_id"]),
                "window_progress": float(r["window_progress"]),
                "segment": r["segment"],
                "dominant_feature": best_feature,
                "delta_from_correct": float(best_delta),
                "deviation_score": float(best_score),
                "direction": "higher" if best_delta >= 0 else "lower",
            }
        )

    expl = pd.DataFrame(rows)
    if expl.empty:
        return expl

    expl = expl.sort_values(
        ["subject", "task", "label", "rep_id", "window_id", "deviation_score"],
        ascending=[True, True, True, True, True, False],
    )
    expl = expl.groupby(["subject", "task", "label", "rep_id", "window_id"], as_index=False).head(1).reset_index(drop=True)
    expl["explanation_key"] = expl["segment"].astype(str) + "__" + expl["dominant_feature"].astype(str) + "__" + expl["direction"].astype(str)
    return expl


def explanation_stability(expl_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (task, label, subject, rep_id), g in expl_df.groupby(["task", "label", "subject", "rep_id"]):
        g = g.sort_values("window_id")
        if g.empty:
            continue
        counts = g["explanation_key"].value_counts()
        top_key = counts.index[0]
        rows.append(
            {
                "task": task,
                "label": label,
                "subject": subject,
                "rep_id": int(rep_id),
                "n_predicted_incorrect_windows": int(len(g)),
                "top_explanation_key": top_key,
                "top_explanation_share": float(counts.iloc[0] / len(g)),
            }
        )
    return pd.DataFrame(rows)


def explanation_variant_summary(expl_df: pd.DataFrame) -> pd.DataFrame:
    if expl_df.empty:
        return pd.DataFrame()
    d = expl_df[~expl_df["label"].astype(str).str.endswith("_correct")].copy()
    if d.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (task, label), g in d.groupby(["task", "label"]):
        counts = g["explanation_key"].value_counts()
        top_key = counts.index[0]
        rows.append(
            {
                "task": task,
                "label": label,
                "top_explanation_key": top_key,
                "top_explanation_share": float(counts.iloc[0] / len(g)),
                "n_incorrect_windows": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def feedback_simulation(
    pred_df: pd.DataFrame,
    expl_df: pd.DataFrame,
    prob_threshold: float,
    persistence: int,
    score_threshold: float = 0.0,
) -> pd.DataFrame:
    lookup = expl_df.set_index(["subject", "task", "label", "rep_id", "window_id"])
    rows: list[dict[str, object]] = []
    for (subject, task, label, rep_id), g in pred_df.groupby(["subject", "task", "label", "rep_id"]):
        g = g.sort_values("window_id")
        run_key = None
        run_len = 0
        trigger_window = None
        trigger_key = None

        for _, row in g.iterrows():
            key = (subject, task, label, int(rep_id), int(row["window_id"]))
            if row["y_pred"] == 1 and row["y_prob"] >= prob_threshold and key in lookup.index:
                score = float(lookup.at[key, "deviation_score"])
                if score < score_threshold:
                    run_key = None
                    run_len = 0
                    continue
                expl_key = lookup.at[key, "explanation_key"]
                if expl_key == run_key:
                    run_len += 1
                else:
                    run_key = expl_key
                    run_len = 1
                if run_len >= persistence:
                    trigger_window = int(row["window_id"])
                    trigger_key = expl_key
                    break
            else:
                run_key = None
                run_len = 0

        rows.append(
            {
                "subject": subject,
                "task": task,
                "label": label,
                "rep_id": int(rep_id),
                "is_incorrect_rep": int(str(label) != f"{task}_correct"),
                "triggered": int(trigger_window is not None),
                "trigger_window": int(trigger_window) if trigger_window is not None else -1,
                "trigger_progress": float(g.loc[g["window_id"] == trigger_window, "window_progress"].iloc[0]) if trigger_window is not None else float("nan"),
                "trigger_explanation_key": trigger_key if trigger_key is not None else "",
            }
        )
    return pd.DataFrame(rows)


def feedback_simulation_task_rules(
    pred_df: pd.DataFrame,
    expl_df: pd.DataFrame,
    task_rules: dict[str, dict[str, float | int]],
) -> pd.DataFrame:
    parts = []
    for task, dfg in pred_df.groupby("task"):
        rules = task_rules.get(task, {})
        part = feedback_simulation(
            pred_df=dfg,
            expl_df=expl_df[expl_df["task"] == task].copy(),
            prob_threshold=float(rules.get("prob_threshold", 0.6)),
            persistence=int(rules.get("persistence", 3)),
            score_threshold=float(rules.get("score_threshold", 0.0)),
        )
        part["rule_name"] = str(rules.get("rule_name", "custom"))
        part["prob_threshold"] = float(rules.get("prob_threshold", 0.6))
        part["persistence"] = int(rules.get("persistence", 3))
        part["score_threshold"] = float(rules.get("score_threshold", 0.0))
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plot_early_detection(early_df: pd.DataFrame, fig_dir: Path) -> None:
    if early_df.empty:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    for task, dft in early_df.groupby("task"):
        detected = dft[np.isfinite(dft["first_detect_progress"])].copy()
        if detected.empty:
            continue
        xs = np.sort(detected["first_detect_progress"].to_numpy(dtype=float))
        ys = np.arange(1, len(xs) + 1) / len(xs)
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Trial progress at first incorrect detection")
        plt.ylabel("Cumulative fraction of incorrect repetitions")
        plt.title(f"Early detection curve ({task.upper()})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / f"project32_early_detection_{task}.png", dpi=200)
        plt.close()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    window_df = pd.read_csv(args.window_features_csv)

    all_pred_rows = []
    config_rows = []

    best_cfg = {
        "rd": ["XSens_Hand_Right", "XSens_LowerLeg_Left", "XSens_LowerLeg_Right"],
        "rgs": ["XSens_Foot_Right", "XSens_Hand_Right", "XSens_LowerLeg_Left", "XSens_Sternum"],
    }
    lower_cfg = {
        "rd": ["XSens_LowerLeg_Left", "XSens_LowerLeg_Right", "XSens_UpperLeg_Left", "XSens_UpperLeg_Right"],
        "rgs": ["XSens_Foot_Left", "XSens_LowerLeg_Left", "XSens_Pelvis"],
    }

    for task in args.tasks:
        available_segments = sorted(window_df.loc[window_df["task"] == task, "segment"].unique().tolist())
        configs = {
            "full": available_segments,
            "best_compact": best_cfg.get(task, available_segments),
            "best_lower_body": lower_cfg.get(task, available_segments),
        }

        for cfg_name, segments in configs.items():
            wide, model_cols = wide_window_table(window_df, task=task, segments=segments)
            pred = loso_window_predictions(wide=wide, model_cols=model_cols, seed=args.seed)
            if pred.empty:
                continue
            pred["config"] = cfg_name
            all_pred_rows.append(pred)
            metrics = window_metrics(pred, task)
            metrics["config"] = cfg_name
            metrics["n_segments"] = len(segments)
            config_rows.append(metrics)

    pred_df = pd.concat(all_pred_rows, ignore_index=True) if all_pred_rows else pd.DataFrame()
    if pred_df.empty:
        raise SystemExit("No predictions produced.")

    config_df = pd.DataFrame(config_rows)
    config_df.to_csv(out_dir / "project32_window_classification_summary.csv", index=False)

    early_frames = []
    for (task, config), dfg in pred_df.groupby(["task", "config"]):
        early = early_detection_summary(dfg, task)
        if early.empty:
            continue
        early["config"] = config
        early_frames.append(early)
    early_df = pd.concat(early_frames, ignore_index=True) if early_frames else pd.DataFrame()
    early_df.to_csv(out_dir / "project32_early_detection.csv", index=False)
    if not early_df.empty:
        early_summary = (
            early_df[early_df["detected"] == 1]
            .groupby(["task", "config"], as_index=False)
            .agg(
                detected_reps=("detected", "size"),
                median_first_detect_progress=("first_detect_progress", "median"),
                mean_first_detect_progress=("first_detect_progress", "mean"),
            )
        )
    else:
        early_summary = pd.DataFrame()
    early_summary.to_csv(out_dir / "project32_early_detection_summary.csv", index=False)

    full_pred = pred_df[pred_df["config"] == "full"].copy()
    full_window_df = window_df.merge(
        full_pred[["subject", "task", "label", "rep_id", "window_id"]],
        on=["subject", "task", "label", "rep_id", "window_id"],
        how="inner",
    )
    expl_df = build_explanations(full_window_df, full_pred)
    expl_df.to_csv(out_dir / "project32_window_explanations.csv", index=False)

    stability_df = explanation_stability(expl_df)
    stability_df.to_csv(out_dir / "project32_explanation_stability.csv", index=False)
    variant_summary_df = explanation_variant_summary(expl_df)
    variant_summary_df.to_csv(out_dir / "project32_explanation_variant_summary.csv", index=False)

    feedback_df = feedback_simulation(
        pred_df=full_pred,
        expl_df=expl_df,
        prob_threshold=args.feedback_prob_threshold,
        persistence=args.feedback_persistence,
        score_threshold=args.feedback_score_threshold,
    )
    feedback_df.to_csv(out_dir / "project32_feedback_simulation.csv", index=False)

    feedback_summary = (
        feedback_df.groupby(["task", "is_incorrect_rep"], as_index=False)
        .agg(
            trigger_rate=("triggered", "mean"),
            median_trigger_progress=("trigger_progress", "median"),
            n_reps=("triggered", "size"),
        )
    )
    feedback_summary.to_csv(out_dir / "project32_feedback_summary.csv", index=False)

    tuned_rules = {
        "rd": {
            "rule_name": "tuned",
            "prob_threshold": 0.90,
            "persistence": 3,
            "score_threshold": 2.5,
        },
        "rgs": {
            "rule_name": "tuned",
            "prob_threshold": 0.95,
            "persistence": 3,
            "score_threshold": 1.5,
        },
    }
    tuned_feedback_df = feedback_simulation_task_rules(
        pred_df=full_pred,
        expl_df=expl_df,
        task_rules=tuned_rules,
    )
    tuned_feedback_df.to_csv(out_dir / "project32_feedback_simulation_tuned.csv", index=False)

    tuned_feedback_summary = (
        tuned_feedback_df.groupby(
            ["task", "rule_name", "prob_threshold", "persistence", "score_threshold", "is_incorrect_rep"],
            as_index=False,
        )
        .agg(
            trigger_rate=("triggered", "mean"),
            median_trigger_progress=("trigger_progress", "median"),
            n_reps=("triggered", "size"),
        )
    )
    tuned_feedback_summary.to_csv(out_dir / "project32_feedback_summary_tuned.csv", index=False)

    plot_early_detection(early_df[early_df["config"] == "full"].copy() if not early_df.empty else early_df, fig_dir=fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
