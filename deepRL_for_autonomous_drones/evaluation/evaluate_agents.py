import os
import subprocess
import time
import pandas as pd
from datetime import datetime
import argparse
import glob
import json
from deepRL_for_autonomous_drones.config.cfg_constants import EVAL_ALGORITHMS

EVAL_ENVIRONMENTS = ["SafetyDroneLanding-v0"]


def find_latest_model(output_dir, algorithm, env):
    """Find the best model file for a given algorithm and environment."""

    group = f"{env}-cost-25"
    if env in ("SafetyBallGather-v0", "SafetyCarGather-v0", "SafetyAntGather-v0", "SafetyDroneGather-v0"):
        group = f"{env}-cost-0.2"

    model_path = os.path.join(output_dir, "fast-safe-rl", group, f"{algorithm}_seed0-*")
    model_path = glob.glob(model_path)

    if not model_path:
        print(f"Warning: No model path found for {algorithm} on {env}")
        return None

    return model_path[0]


def run_evaluation(script, algorithm, env, model_path):
    """Run evaluation for a specific algorithm and environment."""
    cmd = ["python", script, "--path", model_path]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    except Exception as e:
        print(f"Exception evaluating {algorithm} on {env}: {str(e)}")
        return None


def main():
    output_dir = "../training/benchmark_results"

    for env in EVAL_ENVIRONMENTS:
        for algorithm, script in EVAL_ALGORITHMS.items():
            print(f"\nEvaluating {env} with {algorithm}")

            # Find the best model for this algorithm and environment
            model_path = find_latest_model(output_dir, algorithm, env)
            if not model_path:
                print(f"Warning: Could not find model for {algorithm} on {env}, skipping...")
                continue

            run_evaluation(script, env, output_dir, model_path)


if __name__ == "__main__":
    main()
