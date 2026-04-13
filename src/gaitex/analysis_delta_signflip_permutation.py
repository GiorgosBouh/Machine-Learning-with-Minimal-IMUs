import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

FEATURES_CSV = r"output\features\features.csv"
OUT_FOLDS = r"output\features\task_complexity_folds_all_vs_minimal.csv"
N_PERM = 5000
RNG_SEED = 42


# --- Define "minimal" representations consistent with what we used ---
# MINIMAL uses ONE segment x 5 features (=> 5 columns).
TASK_MINIMAL_SEGMENT = {
    "rd": "XSens_LowerLeg_Right",
    "rgs": "XSens_LowerLeg_Left",
}
BASE_FEATURES = ["mean_speed", "rms_speed", "peak_speed", "rms_accel", "rot_range"]


@dataclass
class Setting:
    name: str
    segments: List[str]
    features: List[str]


def build_subject_level_wide(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Input df_feat: repetition-level rows with columns:
      subject, task, label, segment, mean_speed, rms_speed, peak_speed, rms_accel, rot_range
    Output: subject-level wide table:
      subject, task, label, <segment__feature columns...>
    """
    need = {"subject", "task", "label", "segment", *BASE_FEATURES}
    missing = need - set(df_feat.columns)
    if missing:
        raise ValueError(f"features.csv missing columns: {sorted(missing)}")

    # subject-level mean across repetitions (rep_id)
    df_sub = (
        df_feat.groupby(["subject", "task", "label", "segment"], as_index=False)[BASE_FEATURES]
        .mean()
    )

    # wide
    wide = df_sub.pivot_table(
        index=["subject", "task", "label"],
        columns="segment",
        values=BASE_FEATURES,
        aggfunc="first",
    )
    # columns: (feature, segment) -> "segment__feature"
    wide.columns = [f"{seg}__{feat}" for feat, seg in wide.columns]
    wide = wide.reset_index()
    return wide


def make_setting(df_wide: pd.DataFrame, task: str, setting_name: str) -> Setting:
    # detect segments present for this task
    cols = [c for c in df_wide.columns if "__" in c]
    segments = sorted({c.split("__", 1)[0] for c in cols})

    if setting_name == "ALL":
        return Setting(name="ALL", segments=segments, features=BASE_FEATURES)

    if setting_name == "MINIMAL":
        seg = TASK_MINIMAL_SEGMENT[task]
        return Setting(name="MINIMAL", segments=[seg], features=BASE_FEATURES)

    raise ValueError("Unknown setting_name")


def select_Xy(df_wide: pd.DataFrame, task: str, setting: Setting) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    df_t = df_wide[df_wide["task"] == task].copy()

    # columns to use
    use_cols = []
    for seg in setting.segments:
        for feat in setting.features:
            col = f"{seg}__{feat}"
            if col in df_t.columns:
                use_cols.append(col)

    if not use_cols:
        raise ValueError(f"No usable columns for task={task} setting={setting.name}")

    X = df_t[use_cols].to_numpy(dtype=float, copy=True)
    y = df_t["label"].astype(str).to_numpy()
    subjects = df_t["subject"].astype(str).to_numpy()

    return X, y, use_cols, subjects


def loso_fold_accuracies_multiclass(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> Dict[str, float]:
    """
    Return dict: subject -> accuracy on that subject's held-out samples.
    Multiclass LOSO. Each subject has multiple samples (labels).
    """
    clf = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000)),  # avoid multi_class arg (your sklearn complained)
        ]
    )

    uniq = np.unique(subjects)
    subj_acc = {}

    for s in uniq:
        te = subjects == s
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue

        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        acc = float(np.mean(pred == y[te]))
        subj_acc[str(s)] = acc

    return subj_acc


def sign_flip_pvalue(deltas: np.ndarray, n_perm: int = 5000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    obs = float(np.mean(deltas))

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(deltas))
        null[i] = float(np.mean(deltas * signs))

    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, float(null.mean()), float(p)


def main():
    df_feat = pd.read_csv(FEATURES_CSV)
    df_wide = build_subject_level_wide(df_feat)

    all_rows = []

    for task in ["rd", "rgs"]:
        logging.info(f"Task={task}: building settings and running LOSO")

        setting_all = make_setting(df_wide, task, "ALL")
        setting_min = make_setting(df_wide, task, "MINIMAL")

        X_all, y_all, cols_all, subj_all = select_Xy(df_wide, task, setting_all)
        X_min, y_min, cols_min, subj_min = select_Xy(df_wide, task, setting_min)

        # sanity: must align (same subjects and same samples order) for fair per-subject comparison
        # We compare per-subject mean fold acc, so alignment only needs subject sets.
        acc_all = loso_fold_accuracies_multiclass(X_all, y_all, subj_all)
        acc_min = loso_fold_accuracies_multiclass(X_min, y_min, subj_min)

        common = sorted(set(acc_all.keys()) & set(acc_min.keys()))
        deltas = np.array([acc_all[s] - acc_min[s] for s in common], dtype=float)

        # save folds
        for s in common:
            all_rows.append(
                {
                    "task": task,
                    "subject": s,
                    "acc_all": acc_all[s],
                    "acc_minimal": acc_min[s],
                    "delta_all_minus_min": acc_all[s] - acc_min[s],
                    "n_cols_all": len(cols_all),
                    "n_cols_minimal": len(cols_min),
                }
            )

        obs, null_mean, p = sign_flip_pvalue(deltas, n_perm=N_PERM, seed=RNG_SEED)

        print("\n===== PAIRED SIGN-FLIP PERMUTATION (ALL vs MINIMAL) =====")
        print(f"{task.upper()} | n_subjects={len(common)}")
        print(f"Observed Δ (ALL - MINIMAL) = {obs:.4f}")
        print(f"Null mean Δ = {null_mean:.4f}")
        print(f"Permutation p-value (two-sided) = {p:.4f}")

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_FOLDS, index=False)
    logging.info(f"Wrote: {OUT_FOLDS}")


if __name__ == "__main__":
    main()