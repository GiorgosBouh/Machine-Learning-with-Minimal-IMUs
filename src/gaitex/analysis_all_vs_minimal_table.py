from __future__ import annotations

import os
import numpy as np
import pandas as pd


IN_CSV = r"output\features\task_complexity_folds_all_vs_minimal.csv"
OUT_CSV = r"output\features\all_vs_minimal_summary.csv"
OUT_TEX = r"output\features\all_vs_minimal_summary.tex"


def signflip_pvalue(deltas: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    """
    Paired sign-flip permutation test (two-sided) for mean(delta).
    deltas: per-subject paired differences (ALL - MINIMAL)
    """
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    obs = float(np.mean(deltas))

    signs = rng.choice([-1.0, 1.0], size=(n_perm, deltas.size))
    null_means = signs.dot(deltas) / deltas.size

    p = (np.sum(np.abs(null_means) >= abs(obs)) + 1) / (n_perm + 1)
    return float(p)


def _load_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a standardized paired table with columns:
      task, subject, acc_all, acc_minimal

    Supports:
    A) Wide format (your file):
       task, subject, acc_all, acc_minimal, delta_all_minus_min, ...
    B) Long format:
       task, subject, setting (ALL/MINIMAL), accuracy
    """
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}

    # --- A) wide format ---
    if "acc_all" in low and "acc_minimal" in low and "task" in low and "subject" in low:
        out = df[[low["task"], low["subject"], low["acc_all"], low["acc_minimal"]]].copy()
        out.columns = ["task", "subject", "acc_all", "acc_minimal"]
        return out

    # --- B) long format ---
    needed = {"task", "subject", "setting", "accuracy"}
    if needed.issubset(set(low.keys())):
        tmp = df[[low["task"], low["subject"], low["setting"], low["accuracy"]]].copy()
        tmp.columns = ["task", "subject", "setting", "accuracy"]
        piv = (
            tmp.pivot_table(index=["task", "subject"], columns="setting", values="accuracy", aggfunc="mean")
            .reset_index()
        )
        if "ALL" not in piv.columns or "MINIMAL" not in piv.columns:
            raise ValueError(f"Long-format pivot missing ALL/MINIMAL. Columns: {piv.columns.tolist()}")
        piv = piv.rename(columns={"ALL": "acc_all", "MINIMAL": "acc_minimal"})
        return piv[["task", "subject", "acc_all", "acc_minimal"]]

    raise ValueError(
        "Unexpected columns in input CSV.\n"
        f"Found: {cols}\n"
        "Expected either:\n"
        "  (A) wide: task, subject, acc_all, acc_minimal, ...\n"
        "  (B) long: task, subject, setting, accuracy"
    )


def main() -> int:
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Missing input file: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    pairs = _load_pairs(df)

    rows = []
    for task, dft in pairs.groupby("task"):
        all_acc = dft["acc_all"].to_numpy(dtype=float)
        min_acc = dft["acc_minimal"].to_numpy(dtype=float)

        mask = np.isfinite(all_acc) & np.isfinite(min_acc)
        all_acc = all_acc[mask]
        min_acc = min_acc[mask]
        deltas = all_acc - min_acc

        n = int(deltas.size)
        obs_delta = float(np.mean(deltas)) if n else float("nan")
        p = signflip_pvalue(deltas, n_perm=20000, seed=0) if n else float("nan")

        rows.append(
            {
                "task": str(task),
                "n_subjects": n,
                "accuracy_ALL_mean": float(np.mean(all_acc)) if n else float("nan"),
                "accuracy_ALL_sd": float(np.std(all_acc, ddof=1)) if n > 1 else float("nan"),
                "accuracy_MIN_mean": float(np.mean(min_acc)) if n else float("nan"),
                "accuracy_MIN_sd": float(np.std(min_acc, ddof=1)) if n > 1 else float("nan"),
                "delta_ALL_minus_MIN_mean": obs_delta,
                "p_signflip_two_sided": p,
            }
        )

    out = pd.DataFrame(rows).sort_values("task").reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # LaTeX table
    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(
        r"\caption{Paired comparison of full (ALL) vs minimal representations under leave-one-subject-out evaluation. "
        r"Values are per-subject accuracies aggregated across folds; $\Delta$ is the paired mean difference (ALL$-$MINIMAL). "
        r"The $p$-value is from a two-sided paired sign-flip permutation test on per-subject $\Delta$.}"
    )
    tex.append(r"\label{tab:all_vs_minimal}")
    tex.append(r"\begin{tabular}{lcccc}")
    tex.append(r"\hline")
    tex.append(r"Task & $n$ & ALL (mean$\pm$SD) & MIN (mean$\pm$SD) & $\Delta$ (ALL$-$MIN), $p$ \\")
    tex.append(r"\hline")

    for _, r in out.iterrows():
        task = str(r["task"])
        n = int(r["n_subjects"])

        all_m = r["accuracy_ALL_mean"]
        all_sd = r["accuracy_ALL_sd"]
        min_m = r["accuracy_MIN_mean"]
        min_sd = r["accuracy_MIN_sd"]
        d = r["delta_ALL_minus_MIN_mean"]
        p = r["p_signflip_two_sided"]

        def fmt_ms(m, s):
            if not np.isfinite(m):
                return "NA"
            if np.isfinite(s):
                return f"{m:.3f}$\\pm${s:.3f}"
            return f"{m:.3f}"

        all_txt = fmt_ms(all_m, all_sd)
        min_txt = fmt_ms(min_m, min_sd)
        d_txt = f"{d:.3f}" if np.isfinite(d) else "NA"
        p_txt = f"{p:.4f}" if np.isfinite(p) else "NA"

        tex.append(fr"{task} & {n:d} & {all_txt} & {min_txt} & {d_txt}, {p_txt} \\")
    tex.append(r"\hline")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")

    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))

    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_TEX}")
    print("\n=== SUMMARY (ALL vs MINIMAL) ===")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())