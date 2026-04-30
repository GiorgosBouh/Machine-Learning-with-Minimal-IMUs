# src/gaitex/features.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


# ---------- Quaternion math (w, x, y, z) ----------

def quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize quaternions along last axis."""
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate."""
    qc = q.copy()
    qc[..., 1:] *= -1.0
    return qc


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (w,x,y,z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_relative(q_prev: np.ndarray, q_next: np.ndarray) -> np.ndarray:
    """Relative rotation q_rel = conj(q_prev) * q_next."""
    return quat_mul(quat_conj(q_prev), q_next)


def quat_to_angle(q_rel: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert relative quaternion to rotation angle (rad), range [0, pi].
    Uses angle = 2 * arccos(|w|).
    """
    w = np.clip(np.abs(q_rel[..., 0]), -1.0, 1.0)
    return 2.0 * np.arccos(np.maximum(w, eps))


def angular_speed_from_quat(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute angular speed magnitude (rad/s) from quaternion time series.
    q shape: (N,4) in (w,x,y,z)
    t shape: (N,)
    Returns omega shape: (N-1,) after filtering invalid dt.
    """
    q = quat_normalize(q)
    t = np.asarray(t, dtype=float)
    if q.shape[0] < 2:
        return np.array([], dtype=float)

    q_rel = quat_relative(q[:-1], q[1:])
    ang = quat_to_angle(q_rel)  # rad

    dt = np.diff(t)
    dt = np.where(dt <= 0, np.nan, dt)

    omega = ang / dt
    omega = omega[np.isfinite(omega)]
    return omega


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.nanmean(x ** 2)))


# ---------- Column parsing (robust) ----------

@dataclass(frozen=True)
class QuatGroup:
    name: str
    w: str
    x: str
    y: str
    z: str


_AXIS_PAT = re.compile(r"(?:^|[_\-\s])(w|x|y|z)(?:$|[_\-\s])", re.IGNORECASE)


def _axis_from_col(col: str) -> Optional[str]:
    """
    Infer axis w/x/y/z from column name.
    Supports:
      ..._W/_X/_Y/_Z
      ..._QW/_QX/_QY/_QZ   (XSens style)
      ..._qw/_qx/_qy/_qz
      ...QuatW / ...QuatX / ...QuatY / ...QuatZ
    """
    c = col.strip()
    cl = c.lower()

    # XSens style: _QW/_QX/_QY/_QZ
    if cl.endswith("_qw"):
        return "w"
    if cl.endswith("_qx"):
        return "x"
    if cl.endswith("_qy"):
        return "y"
    if cl.endswith("_qz"):
        return "z"

    # generic: _w/_x/_y/_z
    if cl.endswith("_w"):
        return "w"
    if cl.endswith("_x"):
        return "x"
    if cl.endswith("_y"):
        return "y"
    if cl.endswith("_z"):
        return "z"

    # QuatW / quat_w endings
    for ax in ("w", "x", "y", "z"):
        if cl.endswith(f"quat{ax}") or cl.endswith(f"quat_{ax}"):
            return ax

    # fallback: boundary token match
    m = _AXIS_PAT.search(c)
    if m:
        return m.group(1).lower()

    return None


def detect_quaternion_groups(columns: List[str]) -> List[QuatGroup]:
    """
    Groups columns into quaternions (w,x,y,z) by shared base name.

    For XSens columns like:
      XSens_Pelvis_QX, XSens_Pelvis_QY, XSens_Pelvis_QZ, XSens_Pelvis_QW
    base name becomes:
      XSens_Pelvis
    """
    cols = list(columns)
    buckets: Dict[str, Dict[str, str]] = {}

    for col in cols:
        ax = _axis_from_col(col)
        if ax is None:
            continue

        base = col.strip()

        # remove axis suffixes like _QX/_QY/_QZ/_QW or _X/_Y/_Z/_W
        base = re.sub(r"(_q[wxyz])$", "", base, flags=re.IGNORECASE)
        base = re.sub(r"(_[wxyz])$", "", base, flags=re.IGNORECASE)

        # also handle QuatW/QuatX...
        base = re.sub(r"(quat[_]?[wxyz])$", "", base, flags=re.IGNORECASE)

        base = base.strip("_- ").strip()

        if base not in buckets:
            buckets[base] = {}
        if ax not in buckets[base]:
            buckets[base][ax] = col

    groups: List[QuatGroup] = []
    for base, axes in buckets.items():
        if all(k in axes for k in ("w", "x", "y", "z")):
            groups.append(
                QuatGroup(
                    name=base,
                    w=axes["w"],
                    x=axes["x"],
                    y=axes["y"],
                    z=axes["z"],
                )
            )

    return sorted(groups, key=lambda g: g.name)


def extract_quat(df: pd.DataFrame, g: QuatGroup) -> np.ndarray:
    """
    Returns quaternion array (N,4) in (w,x,y,z) order.
    """
    q = df[[g.w, g.x, g.y, g.z]].to_numpy(dtype=float, copy=False)
    return quat_normalize(q)


# ---------- Feature computation ----------

def compute_rep_features(q: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    """
    Compute features for one quaternion time series segment.
    """
    omega = angular_speed_from_quat(q, t)  # rad/s

    # accel proxy: derivative of omega (simple, robust)
    if omega.size >= 2:
        median_dt = float(np.nanmedian(np.diff(t))) if t.size >= 3 else float("nan")
        if np.isfinite(median_dt) and median_dt > 0:
            alpha = np.diff(omega) / median_dt
        else:
            alpha = np.diff(omega)
    else:
        alpha = np.array([], dtype=float)

    # rotation step magnitudes (rad)
    q_rel = quat_relative(q[:-1], q[1:]) if q.shape[0] >= 2 else np.empty((0, 4))
    angle_steps = quat_to_angle(q_rel) if q_rel.size else np.array([], dtype=float)

    feats = {
        "mean_speed": float(np.nanmean(omega)) if omega.size else float("nan"),
        "rms_speed": rms(omega),
        "peak_speed": float(np.nanmax(omega)) if omega.size else float("nan"),
        "rms_accel": rms(alpha),
        "rot_range": float(np.nanmax(angle_steps) - np.nanmin(angle_steps)) if angle_steps.size else float("nan"),
    }
    return feats


def compute_features_for_all_segments(
    t: np.ndarray,
    df: pd.DataFrame,
    subject: str,
    task: str,
    label: str,
    rep_id: int,
) -> List[Dict[str, Any]]:
    """
    Detect all quaternion groups in df and compute features for each group.
    Returns list of rows for features.csv
    """
    groups = detect_quaternion_groups(list(df.columns))
    rows: List[Dict[str, Any]] = []

    if not groups:
        return rows

    for g in groups:
        try:
            q = extract_quat(df, g)
            feats = compute_rep_features(q, t)
        except Exception:
            continue

        rows.append(
            {
                "subject": subject,
                "task": task,
                "label": label,
                "rep_id": int(rep_id),
                "segment": g.name,
                **feats,
            }
        )

    return rows
