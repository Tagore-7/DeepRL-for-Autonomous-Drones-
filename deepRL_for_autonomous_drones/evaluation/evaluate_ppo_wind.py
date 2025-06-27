import os
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
import glob
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from deepRL_for_autonomous_drones.envs.Drone_Controller_RPM import DroneControllerRPM

COST_KEYS = {
    "total": "cost",
    "tilt": "tilt_cost",
    "spin": "spin_cost",
    "lidar": "lidar_cost",
}


def make_env(task, seed=0, wind_scale=1.0):
    env = gym.make(task, render_mode="human", graphics=True)
    env = FlattenObservation(env)
    env = Monitor(env)
    env.reset(seed=seed)

    if hasattr(env.unwrapped, "setWindEffects"):
        env.unwrapped.setWindEffects(True)
    if hasattr(env.unwrapped, "setWindScale"):
        env.unwrapped.setWindScale(wind_scale)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="SafetyDroneLanding-v0")
    parser.add_argument("--root", default="sb3_runs_nonconstrained")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    wind_levels = np.linspace(0.0, 1.0, num=11)
    all_results = []

    ckpt = find_latest_checkpoint(args.root, "PPO", args.task)
    if ckpt is None:
        print("No checkpoint for PPO")
        return

    env = make_env(args.task, seed=args.seed, wind_scale=0.0)
    model = PPO.load(ckpt, device="cuda")

    for wind in wind_levels:
        print(f"Evaluating PPO with wind level {wind:.1f}")
        if hasattr(env.unwrapped, "setWindScale"):
            env.unwrapped.setWindScale(wind)

        metrics = rollout_with_cost(model, env, args.episodes)

        row = dict(task=args.task, wind=wind, episodes=args.episodes, checkpoint=os.path.dirname(ckpt), **metrics)
        all_results.append(row)

        print(
            f"PPO | Wind {wind:.1f} | R {metrics['reward_mean']:.2f} ± {metrics['reward_std']:.2f} | "
            f"C {metrics['total_mean']:5.2f} ± {metrics['total_std']:5.2f} "
            f"(tilt {metrics['tilt_mean']:4.2f}, spin {metrics['spin_mean']:4.2f}, "
            f"lidar {metrics['lidar_mean']:4.2f})"
        )
    env.close()

    df = pd.DataFrame(all_results)
    os.makedirs("evaluation", exist_ok=True)
    out_path = os.path.join("evaluation", "ppo_wind_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved wind sweep evaluation to {out_path}")


if __name__ == "__main__":
    main()
