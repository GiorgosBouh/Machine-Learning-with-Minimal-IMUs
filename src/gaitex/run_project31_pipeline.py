from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".mplconfig"


def run_step(module: str, *args: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", module, *args]
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
    env.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> int:
    run_step("gaitex.analysis_sensor_configuration_frontier")
    run_step("gaitex.analysis_semantic_feedback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
