import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FEATURES_CSV = Path(r"output\features\features.csv")
OUT_FIG = Path(r"output\figures\segment_cohens_d_heatmap.png")

def cohens_d(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = np.sqrt(((nx-1)*vx + (ny-1)*vy)/(nx+ny-2))
    return (x.mean() - y.mean()) / pooled if pooled > 0 else np.nan

df = pd.read_csv(FEATURES_CSV)

rows = []

for task in sorted(df["task"].unique()):
    df_task = df[df["task"] == task]
    correct_label = [l for l in df_task["label"].unique() if "correct" in l.lower()][0]

    for segment in sorted(df_task["segment"].unique()):
        df_seg = df_task[df_task["segment"] == segment]

        correct = df_seg[df_seg["label"] == correct_label]["rms_speed"]

        for label in df_seg["label"].unique():
            if label == correct_label:
                continue
            variant = df_seg[df_seg["label"] == label]["rms_speed"]

            d = cohens_d(variant, correct)

            rows.append({
                "task": task,
                "segment": segment,
                "variant": label,
                "cohens_d": d,
            })

heat = pd.DataFrame(rows)

# Plot per task
tasks = heat["task"].unique()
fig, axes = plt.subplots(1, len(tasks), figsize=(7*len(tasks), 6), sharey=True)

if len(tasks) == 1:
    axes = [axes]

for ax, task in zip(axes, tasks):
    h = heat[heat["task"] == task]
    pivot = h.pivot(index="segment", columns="variant", values="cohens_d")

    im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
    ax.set_title(task.upper())
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

fig.colorbar(im, ax=axes, label="Cohen's d (variant vs correct)")
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()

print(f"Wrote: {OUT_FIG}")
