import os
import subprocess
import time
import pandas as pd
from datetime import datetime
import argparse
import glob
import json
from deepRL_for_autonomous_drones.config.cfg_constants import TRAIN_ALGORITHMS

TRAIN_ENVIRONMENTS = ["SafetyDroneLanding-v0"]


def run_training(algorithm, env, output_dir):
    """Run training for a specific algorithm and environment."""
    cmd = [
        "python",
        algorithm,
        "--task",
        env,
        "--logdir",
        output_dir,
        "--seed",
        "20",
    ]

    print(f"Running {algorithm} on {env}")
    try:
        result = subprocess.run(cmd, text=True, check=True)
        if result.returncode != 0:
            print(f"Error running {algorithm} on {env}:")
            print(result.stderr)
            return None

    except Exception as e:
        print(f"Exception running {algorithm} on {env}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=list(TRAIN_ALGORITHMS.keys()),
        help="Algorithms to benchmark (default: all)",
    )
    parser.add_argument(
        "--environments",
        type=str,
        nargs="+",
        default=TRAIN_ENVIRONMENTS,
        help="Environments to benchmark (default: all)",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run benchmarks
    for env in args.environments:
        for _, script in TRAIN_ALGORITHMS.items():
            run_training(script, env, args.output_dir)


if __name__ == "__main__":
    main()
