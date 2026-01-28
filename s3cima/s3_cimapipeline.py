#!/usr/bin/env python3
"""
Master pipeline runner with:
1. Timestamped log filenames
2. Timestamps written inside logs
3. Strict sequential execution
"""

import subprocess
import time
from datetime import datetime
import os
import sys


LOG_DIR = "s3cimalog"
os.makedirs(LOG_DIR, exist_ok=True)


def timestamp_file():
    """Timestamp safe for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamp_human():
    """Human readable timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_step(name, cmd, logfile_prefix):

    logfile = os.path.join(
        LOG_DIR,
        f"{logfile_prefix}_{timestamp_file()}.log"
    )

    print("\n" + "=" * 60)
    print(f" RUNNING: {name}")
    print(f" LOG    : {logfile}")
    print("=" * 60 + "\n")

    with open(logfile, "a") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"START {name} at {timestamp_human()}\n")
        f.write("=" * 60 + "\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=f,
            shell=True
        )
        process.wait()

        f.write("\n")
        f.write(
            f"END {name} at {timestamp_human()} "
            f"(exit={process.returncode})\n"
        )
        f.write("=" * 60 + "\n\n")
        f.flush()

    if process.returncode != 0:
        print(f" ❌ {name} FAILED (exit code {process.returncode})")
        sys.exit(process.returncode)

    print(f" ✅ {name} COMPLETED\n")
    time.sleep(1)


def main():

    # ---------- S3 ANALYSIS (BATCH 2048) ----------
    # run_step(
    #     name="S3_analysis_batch_256_128p",
    #     cmd=(
    #         "python3 -u s3_analysis.py "
    #         "--run_mait "
    #         "--data '/mnt/volumec/Aswath/patchmodel/s3cima/CNNResnetembeddings/128markerawarebatchsize256_2conv.csv' "
    #         "--out 's3cimares/markeraware/batch256_2conv/resultsMAIT/'"
    #     ),
    #     logfile_prefix="s3cima_batch_256_2conv"
    # )

    # ---------- CNN EMBEDDINGS (BATCH 128) ----------
    run_step(
        name="CNNembedding_batch_1024_38ch_128p",
        cmd=(
            "python3 -u cnnembeddings.py "
            "--batchsize 1024 "
            "--out_csv '/mnt/volumec/Aswath/patchmodel/s3cima/"
            "CNNResnetembeddings/38chresnetpretrained.csv'"
        ),
        logfile_prefix="cnn_batch_1024_38ch_128p"
    )

    # ---------- S3 ANALYSIS (BATCH 1024) ----------
    run_step(
        name="S3_analysis_batch_1024_38ch",
        cmd=(
            "python3 -u s3_analysis.py "
            "--run_mait "
            "--data '/mnt/volumec/Aswath/patchmodel/s3cima/"
            "CNNResnetembeddings/38chresnetpretrained.csv' "
            "--out 's3cimares/markeraware/batch1024_38ch_128/resultsMAIT'"
        ),
        logfile_prefix="S3_analysis_batch_1024_38ch"
    )

    # # ---------- CNN EMBEDDINGS (BATCH 256) ----------
    # run_step(
    #     name="CNNembedding_batch_256_256p_2conv",
    #     cmd=(
    #         "python3 -u cnnembeddings.py "
    #         "--batchsize 256 "
    #         "--out_csv '/mnt/volumec/Aswath/patchmodel/s3cima/"
    #         "CNNResnetembeddings/256markerawarebatchsize256_2conv.csv'"
    #     ),
    #     logfile_prefix="CNNembedding_batch_256_256p_2conv"
    # )

    # # ---------- S3 ANALYSIS (BATCH 256) ----------
    # run_step(
    #     name="S3_analysisbatch_256_256p_2conv",
    #     cmd=(
    #         "python3 -u s3_analysis.py "
    #         "--run_mait "
    #         "--data '/mnt/volumec/Aswath/patchmodel/s3cima/"
    #         "CNNResnetembeddings/256markerawarebatchsize256_2conv.csv' "
    #         "--out 's3cimares/markeraware/256batch256_2conv/resultsMAIT'"
    #     ),
    #     logfile_prefix="S3_analysisbatch_256_256p_2conv"
    # )
    
    # run_step(
    #     name="CNNembedding_batch_2048_256p_2conv",
    #     cmd=(
    #         "python3 -u cnnembeddings.py "
    #         "--batchsize 2048 "
    #         "--out_csv '/mnt/volumec/Aswath/patchmodel/s3cima/"
    #         "CNNResnetembeddings/256markerawarebatchsize2048_2conv.csv'"
    #     ),
    #     logfile_prefix="CNNembedding_batch_2048_256p_2conv"
    # )

    # # ---------- S3 ANALYSIS (BATCH 256) ----------
    # run_step(
    #     name="S3_analysisbatch_256_256p_2conv",
    #     cmd=(
    #         "python3 -u s3_analysis.py "
    #         "--run_mait "
    #         "--data '/mnt/volumec/Aswath/patchmodel/s3cima/"
    #         "CNNResnetembeddings/256markerawarebatchsize2048_2conv.csv' "
    #         "--out 's3cimares/markeraware/256batch256_2conv/resultsMAIT'"
    #     ),
    #     logfile_prefix="S3_analysisbatch_2048_256p_2conv"
    # )
    
    

    print("\n" + "=" * 60)
    print(" 🎉 ALL STEPS COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()