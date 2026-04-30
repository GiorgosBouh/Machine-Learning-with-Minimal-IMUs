from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gaitex.analysis_project32_closed_loop import (
    build_explanations,
    feedback_simulation_task_rules,
    loso_window_predictions,
    wide_window_table,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reviewer 2 feedback-rule grid search.")
    p.add_argument("--window_features_csv", default="output/features/window_features.csv")
    p.add_argument("--out_dir", default="output/features")
    return p.parse_args()


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    alt = Path("..") / p
    if alt.exists():
        return alt
    if p.exists():
        return p
    return p


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    window_features_path = resolve_path(args.window_features_csv)
    window_df = pd.read_csv(window_features_path)
    pred_parts = []

    for task in ["rd", "rgs"]:
        segments = sorted(window_df.loc[window_df["task"] == task, "segment"].unique().tolist())
        wide, model_cols = wide_window_table(window_df, task=task, segments=segments)
        pred = loso_window_predictions(wide=wide, model_cols=model_cols, seed=0)
        pred_parts.append(pred)

    pred_df = pd.concat(pred_parts, ignore_index=True)
    full_window_df = window_df.merge(
        pred_df[["subject", "task", "label", "rep_id", "window_id"]],
        on=["subject", "task", "label", "rep_id", "window_id"],
        how="inner",
    )
    expl_df = build_explanations(full_window_df, pred_df)

    rows: list[dict[str, object]] = []
    probs = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    persistences = [2, 3, 4]
    scores = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]

    for task in ["rd", "rgs"]:
        pred_task = pred_df[pred_df["task"] == task].copy()
        expl_task = expl_df[expl_df["task"] == task].copy()
        for prob_threshold in probs:
            for persistence in persistences:
                for score_threshold in scores:
                    rules = {
                        task: {
                            "rule_name": "grid",
                            "prob_threshold": prob_threshold,
                            "persistence": persistence,
                            "score_threshold": score_threshold,
                        }
                    }
                    sim = feedback_simulation_task_rules(pred_task, expl_task, rules)
                    summary = sim.groupby("is_incorrect_rep")["triggered"].mean()
                    correct_rate = float(summary.get(0, float("nan")))
                    incorrect_rate = float(summary.get(1, float("nan")))
                    rows.append(
                        {
                            "task": task,
                            "prob_threshold": prob_threshold,
                            "persistence": persistence,
                            "score_threshold": score_threshold,
                            "trigger_rate_correct": correct_rate,
                            "trigger_rate_incorrect": incorrect_rate,
                            "objective": incorrect_rate - correct_rate,
                        }
                    )

    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "reviewer2_feedback_grid.csv", index=False)

    top_parts = []
    for task, dft in grid.groupby("task"):
        feasible = dft[dft["trigger_rate_correct"] <= 0.10].copy()
        feasible = feasible.sort_values(
            ["trigger_rate_incorrect", "objective", "prob_threshold", "score_threshold"],
            ascending=[False, False, False, False],
        )
        top_parts.append(feasible.head(10))
    top = pd.concat(top_parts, ignore_index=True)
    top.to_csv(out_dir / "reviewer2_feedback_grid_top_feasible.csv", index=False)

    selected_rules = {
        "rd": {"rule_name": "grid_selected", "prob_threshold": 0.95, "persistence": 2, "score_threshold": 1.0},
        "rgs": {"rule_name": "grid_selected", "prob_threshold": 0.95, "persistence": 3, "score_threshold": 1.0},
    }
    selected = feedback_simulation_task_rules(pred_df=pred_df, expl_df=expl_df, task_rules=selected_rules)
    selected_summary = (
        selected.groupby(
            ["task", "rule_name", "prob_threshold", "persistence", "score_threshold", "is_incorrect_rep"],
            as_index=False,
        )
        .agg(
            trigger_rate=("triggered", "mean"),
            median_trigger_progress=("trigger_progress", "median"),
            n_reps=("triggered", "size"),
        )
    )
    selected_summary.to_csv(out_dir / "reviewer2_feedback_grid_selected_summary.csv", index=False)

    print(f"Wrote: {out_dir / 'reviewer2_feedback_grid.csv'}")
    print(f"Wrote: {out_dir / 'reviewer2_feedback_grid_top_feasible.csv'}")
    print(f"Wrote: {out_dir / 'reviewer2_feedback_grid_selected_summary.csv'}")
    print(selected_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
