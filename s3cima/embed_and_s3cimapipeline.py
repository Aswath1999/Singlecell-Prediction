#!/usr/bin/env python3
"""
Idempotent pipeline runner:
- Runs S3-CIMA for nrun=10 and nrun=50
- Automatically computes missing embeddings internally
- Safe with nohup and reruns
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path
import sys

# ============================================================
# USER CONFIG
# ============================================================
GPU_ID = 0
SCIMA_WORKERS = 8
K = 50

SCIMA_DIR = Path("s3cimamarkersmasked")
MAIN_SCRIPT = "embed_and_s3cima.py"

# ============================================================
# ENVIRONMENT (critical)
# ============================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ============================================================
def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def run(cmd, env=None):
    log("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


# ============================================================
# MAIN
# ============================================================
def main():
    log("🚀 Pipeline started")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    # --------------------------------------------------------
    # S3-CIMA nrun=10
    # --------------------------------------------------------
    scima_10_dir = SCIMA_DIR / f"K{K}_nrun10"
    if not scima_10_dir.exists():
        log("🧠 Running S3-CIMA (nrun=10)")
        run(
            [
                sys.executable,
                MAIN_SCRIPT,
                "--scima_only",
                "--gpu",
                str(GPU_ID),
                "--K",
                str(K),
                "--nrun",
                "10",
                "--scima_workers",
                str(SCIMA_WORKERS),
            ],
            env=env,
        )
    else:
        log("✅ S3-CIMA nrun=10 already present → skipping")

    # --------------------------------------------------------
    # S3-CIMA nrun=50
    # --------------------------------------------------------
    scima_50_dir = SCIMA_DIR / f"K{K}_nrun50"
    if not scima_50_dir.exists():
        log("🧠 Running S3-CIMA (nrun=50)")
        run(
            [
                sys.executable,
                MAIN_SCRIPT,
                "--scima_only",
                "--gpu",
                str(GPU_ID),
                "--K",
                str(K),
                "--nrun",
                "50",
                "--scima_workers",
                str(SCIMA_WORKERS),
            ],
            env=env,
        )
    else:
        log("✅ S3-CIMA nrun=50 already present → skipping")

    log("🎉 Pipeline finished successfully")


if __name__ == "__main__":
    main()