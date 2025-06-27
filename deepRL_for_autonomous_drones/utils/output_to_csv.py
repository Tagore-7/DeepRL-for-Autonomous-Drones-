import os
import pandas as pd
import numpy as np


def training_csv_output(eval_reward, eval_cost, logdir, name, task, prefix):
    csv_path = os.path.join(logdir, name, "progress.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "task": task,
                "algo": prefix,
                "reward": eval_reward,
                "cost": eval_cost,
            }
        ]
    )
    df.to_csv(csv_path, index=False)
    print(f"Final eval CSV saved to {csv_path}")


def evaluation_csv_output(eval_reward, eval_cost):
    return

def save_eval_results(csv_path: str, rewards, cost):
    """
    Save per-episode reward & cost arrays to *csv_path*.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(
        {
            "episode": np.arange(len(rewards)),
            "reward": rewards,
            "cost": cost
        }
    )
    df.to_csv(csv_path, index=False)
    print(f"[output_to_csv] saved evaluation CSV -> {csv_path}")