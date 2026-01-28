#!/usr/bin/env python3
"""
Master pipeline runner with:
1. Timestamped log filenames
2. Timestamps written inside logs
"""

import subprocess
import time
from datetime import datetime

def timestamp_file():
    """Timestamp safe for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def timestamp_human():
    """Human readable timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_step(name, cmd, logfile_prefix):

    # Create log file name with timestamp automatically
    logfile = f"{logfile_prefix}_{timestamp_file()}.log"

    print(f"\n==============================")
    print(f" RUNNING: {name}")
    print(f" LOG    : {logfile}")
    print(f"==============================\n")

    with open(logfile, "a") as f:
        f.write("\n")
        f.write("="*60 + "\n")
        f.write(f"START {name} at {timestamp_human()}\n")
        f.write("="*60 + "\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=f,
            shell=True
        )
        process.wait()

        f.write("\n")
        f.write(f"END {name} at {timestamp_human()}  (exit={process.returncode})\n")
        f.write("="*60 + "\n\n")
        f.flush()

    if process.returncode != 0:
        print(f" {name} FAILED (exit code {process.returncode})")
        exit(process.returncode)

    print(f" {name} COMPLETED\n")
    time.sleep(1)


def main():

    # run_step(
    #     name="Trainingmultimarker38-3chinstancenorm",
    #     cmd="python -u train_multiheadmarker.py",
    #     logfile_prefix="Trainingmultimarker383chinstancenorm"
    # )
    
    # run_step(
    #     name="train_multiheadbatchnorm",
    #     cmd="python -u train_multiheadbatchnorm.py",
    #     logfile_prefix="train_multiheadbatchnorm"
    # )
    
    run_step(
        name="train_native38chresnet",
        cmd="python -u train_native38chresnet.py",
        logfile_prefix="train_native38chresnet"
    )
    
    # run_step(
    #     name="eval_multiheadmarker",
    #     cmd="python -u eval_multiheadmarker.py",
    #     logfile_prefix="eval_multiheadmarker"
    # )
    
    run_step(
        name="eval_native38ch",
        cmd="python -u eval_native38ch.py",
        logfile_prefix="eval_native38ch"
    )
    
    # run_step(
    #     name="eval_native38ch",
    #     cmd="python -u eval_native38ch.py",
    #     logfile_prefix="eval_native38ch"
    # )
        
    # run_step(
    #     name="eval_multiheadbatchnorm",
    #     cmd="python -u eval_multiheadbatchnorm.py",
    #     logfile_prefix="eval_multiheadbatchnorm"
    # )
    
    
    print("\n==============================")
    print(" ALL STEPS COMPLETED SUCCESSFULLY 🎉")
    print("==============================\n")


if __name__ == "__main__":
    main()