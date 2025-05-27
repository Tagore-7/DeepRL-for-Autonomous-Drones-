"""
Evaluate sb3, sb3_contrib models  on SafetyDroneLanding-v0
eposide metrics are saved to ./evaluations.csv
"""

import argparse
import glob
import os
import importlib
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from deepRL_for_autonomous_drones.envs.Drone_Controller_RPM import DroneControllerRPM

ALGOS = {
    "PPO": "PPO",
    "A2C": "A2C",
    "TRPO": "TRPO",
    "DDPG": "DDPG",
    "TD3": "TD3",
    "SAC": "SAC",
    "ARS": "ARS",
    "CrossQ": "CrossQ",
    "TQC": "TQC",
}

COST_KEYS = {
    "total": "cost",
    "tilt": "tilt_cost",
    "spin": "spin_cost",
    "lidar": "lidar_cost",
}


def make_env(task, seed):
    # env = gym.make(task, render_mode="human", graphics=True)
    env = gym.make(task)
    env = FlattenObservation(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def find_latest_checkpoint(root_dir, algo, task):
    pattern = os.path.join(root_dir, f"{algo}_{task}_*/policy.zip")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def rollout_with_cost(model, env, n_episodes: int):
    stats = {k: [] for k in ["reward", *COST_KEYS]}
    lens = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        totals = dict.fromkeys(stats, 0.0)
        ep_len = 0

        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)

            totals["reward"] += r
            for key, inf_key in COST_KEYS.items():
                totals[key] += info.get(inf_key, 0.0)

            ep_len += 1
            done = term or trunc

        for k in totals:
            stats[k].append(totals[k])

        lens.append(ep_len)

    result = {}
    for k, vals in stats.items():
        result[f"{k}_mean"] = float(np.mean(vals))
        result[f"{k}_std"] = float(np.std(vals))
    result["ep_len"] = float(np.mean(lens))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="SafetyDroneLanding-v0")
    p.add_argument("--root", default="sb3_runs_nonconstrained")
    p.add_argument("--algos", nargs="+", default=list(ALGOS.keys()))
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs("evaluation", exist_ok=True)
    csv_path = "evaluation/evaluations_csv.csv"
    all_rows = []

    for algo in args.algos:
        ckpt = find_latest_checkpoint(args.root, algo, args.task)
        if ckpt is None:
            print(f"No checkpoint for {algo}")
            continue

        print(f"Evaluating {algo} -> {ckpt}")
        parent_mod = "sb3_contrib" if algo in {"ARS", "CrossQ", "TQC", "TRPO"} else "stable_baselines3"
        model_cls = getattr(importlib.import_module(parent_mod), ALGOS[algo])
        model = model_cls.load(ckpt, device="cuda")

        env = make_env(args.task, args.seed)
        metrics = rollout_with_cost(model, env, args.episodes)
        env.close()

        row = dict(task=args.task, algo=algo, episodes=args.episodes, checkpoint=os.path.dirname(ckpt), **metrics)

        all_rows.append(row)

        print(
            f"{algo:7s} | R {metrics['reward_mean']:7.2f} ± {metrics['reward_std']:5.2f} | "
            f"C {metrics['total_mean']:5.2f} ± {metrics['total_std']:5.2f} "
            f"(tilt {metrics['tilt_mean']:4.2f}, spin {metrics['spin_mean']:4.2f}, "
            f"lidar {metrics['lidar_mean']:4.2f})"
        )

    if all_rows:
        df = pd.DataFrame(all_rows)
        if os.path.isfile(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
        print(f"\nLogged results to {csv_path}")
    else:
        print("Nothing evaluated")


if __name__ == "__main__":
    main()
