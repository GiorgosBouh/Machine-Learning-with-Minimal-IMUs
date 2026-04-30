from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["mean_speed", "rms_speed", "peak_speed", "rms_accel", "rot_range"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create interpretable semantic movement fingerprints and feedback text."
    )
    p.add_argument("--features_csv", default="output/features/features.csv")
    p.add_argument("--out_dir", default="output/features")
    p.add_argument("--top_k", type=int, default=3)
    return p.parse_args()


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"subject", "task", "label", "segment"} | set(FEATURES)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in features CSV: {missing}")
    return df


def subject_level(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["subject", "task", "label", "segment"], as_index=False)[FEATURES]
        .mean()
    )


def magnitude_label(abs_dz: float) -> str:
    if not np.isfinite(abs_dz):
        return "unknown"
    if abs_dz < 0.20:
        return "trivial"
    if abs_dz < 0.50:
        return "small"
    if abs_dz < 0.80:
        return "moderate"
    if abs_dz < 1.20:
        return "large"
    return "very large"


def feature_phrase(feature: str, delta: float) -> str:
    direction = "higher" if delta >= 0 else "lower"
    mapping = {
        "mean_speed": f"{direction} average rotational speed",
        "rms_speed": f"{direction} typical rotational speed",
        "peak_speed": f"{direction} peak rotational speed",
        "rms_accel": f"{direction} rotational acceleration intensity",
        "rot_range": f"{direction} rotational excursion",
    }
    return mapping.get(feature, f"{direction} {feature}")


def segment_pretty(segment: str) -> str:
    return segment.replace("XSens_", "").replace("_", " ")


def compute_segment_feature_effects(df_subj: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for task in sorted(df_subj["task"].unique()):
        dft = df_subj[df_subj["task"] == task].copy()
        correct_label = f"{task}_correct"
        variants = sorted([l for l in dft["label"].unique() if l != correct_label])

        for variant in variants:
            corr = dft[dft["label"] == correct_label].set_index(["subject", "segment"])
            var = dft[dft["label"] == variant].set_index(["subject", "segment"])
            common = corr.index.intersection(var.index)
            if len(common) < 2:
                continue

            corr_common = corr.loc[common, FEATURES]
            var_common = var.loc[common, FEATURES]

            for segment in sorted({idx[1] for idx in common}):
                mask = [idx[1] == segment for idx in common]
                x_seg = corr_common.loc[mask, :]
                y_seg = var_common.loc[mask, :]

                for feature in FEATURES:
                    x = x_seg[feature].to_numpy(dtype=float)
                    y = y_seg[feature].to_numpy(dtype=float)
                    d = y - x
                    d = d[np.isfinite(d)]
                    if d.size < 2:
                        continue

                    delta_mean = float(np.mean(d))
                    delta_sd = float(np.std(d, ddof=1))
                    dz = float(delta_mean / delta_sd) if delta_sd > 0 else float("nan")

                    rows.append(
                        {
                            "task": task,
                            "correct_label": correct_label,
                            "variant_label": variant,
                            "segment": segment,
                            "feature": feature,
                            "n_subjects": int(d.size),
                            "delta_mean": delta_mean,
                            "delta_sd": delta_sd,
                            "dz_variant_minus_correct": dz,
                            "abs_dz": abs(dz) if np.isfinite(dz) else float("nan"),
                            "direction_text": feature_phrase(feature, delta_mean),
                            "magnitude_label": magnitude_label(abs(dz) if np.isfinite(dz) else float("nan")),
                        }
                    )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["task", "variant_label", "abs_dz"], ascending=[True, True, False]).reset_index(drop=True)
    return out


def build_variant_feedback(effects: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (task, variant), dfg in effects.groupby(["task", "variant_label"]):
        top = dfg.sort_values("abs_dz", ascending=False).head(top_k)
        bullets = []
        for _, r in top.iterrows():
            bullets.append(
                f"{r['magnitude_label']} change in {r['direction_text']} at the {segment_pretty(r['segment'])} "
                f"(dz={r['dz_variant_minus_correct']:.2f})"
            )

        summary = (
            f"Compared with {task.upper()} correct execution, the {variant.replace(task + '_', '')} variant was characterized by "
            + "; ".join(bullets)
            + "."
        )

        rows.append(
            {
                "task": task,
                "variant_label": variant,
                "top_features": " | ".join(bullets),
                "summary_text": summary,
            }
        )
    return pd.DataFrame(rows)


def build_subject_feedback(df_subj: pd.DataFrame, effects: pd.DataFrame, top_k: int) -> pd.DataFrame:
    effect_lookup = effects.set_index(["task", "variant_label", "segment", "feature"])
    rows: list[dict[str, object]] = []

    for task in sorted(df_subj["task"].unique()):
        dft = df_subj[df_subj["task"] == task].copy()
        correct_label = f"{task}_correct"
        corr = dft[dft["label"] == correct_label].set_index(["subject", "segment"])

        for variant in sorted([l for l in dft["label"].unique() if l != correct_label]):
            var = dft[dft["label"] == variant].set_index(["subject", "segment"])
            common = corr.index.intersection(var.index)
            if len(common) == 0:
                continue

            subjects = sorted({idx[0] for idx in common})
            for subject in subjects:
                candidate_rows = []
                for segment in sorted({idx[1] for idx in common if idx[0] == subject}):
                    idx = (subject, segment)
                    if idx not in corr.index or idx not in var.index:
                        continue
                    for feature in FEATURES:
                        delta = float(var.at[idx, feature] - corr.at[idx, feature])
                        key = (task, variant, segment, feature)
                        dz = float(effect_lookup.at[key, "dz_variant_minus_correct"]) if key in effect_lookup.index else float("nan")
                        abs_dz = abs(dz) if np.isfinite(dz) else 0.0
                        candidate_rows.append(
                            {
                                "segment": segment,
                                "feature": feature,
                                "delta": delta,
                                "direction_text": feature_phrase(feature, delta),
                                "population_dz": dz,
                                "abs_population_dz": abs_dz,
                            }
                        )

                if not candidate_rows:
                    continue

                cand = pd.DataFrame(candidate_rows).sort_values(
                    ["abs_population_dz", "delta"],
                    ascending=[False, False],
                ).head(top_k)

                phrases = [
                    f"{row['direction_text']} at the {segment_pretty(row['segment'])}"
                    for _, row in cand.iterrows()
                ]
                text = (
                    f"For subject {subject}, the {variant.replace(task + '_', '')} trial deviated from the subject's own correct "
                    f"{task.upper()} baseline mainly through " + "; ".join(phrases) + "."
                )

                rows.append(
                    {
                        "subject": subject,
                        "task": task,
                        "variant_label": variant,
                        "feedback_text": text,
                    }
                )

    return pd.DataFrame(rows)


def write_markdown(variant_feedback: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Semantic Movement Fingerprints", ""]
    for _, row in variant_feedback.sort_values(["task", "variant_label"]).iterrows():
        lines.append(f"## {row['task'].upper()} - {row['variant_label']}")
        lines.append("")
        lines.append(row["summary_text"])
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_features(args.features_csv)
    df_subj = subject_level(df)
    effects = compute_segment_feature_effects(df_subj)
    variant_feedback = build_variant_feedback(effects, top_k=args.top_k)
    subject_feedback = build_subject_feedback(df_subj, effects, top_k=args.top_k)

    effects_csv = out_dir / "semantic_fingerprint_effects.csv"
    variant_csv = out_dir / "semantic_feedback_variants.csv"
    subject_csv = out_dir / "semantic_feedback_subject_examples.csv"
    markdown_path = out_dir / "semantic_feedback_examples.md"

    effects.to_csv(effects_csv, index=False)
    variant_feedback.to_csv(variant_csv, index=False)
    subject_feedback.to_csv(subject_csv, index=False)
    write_markdown(variant_feedback, markdown_path)

    print(f"Wrote: {effects_csv}")
    print(f"Wrote: {variant_csv}")
    print(f"Wrote: {subject_csv}")
    print(f"Wrote: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
