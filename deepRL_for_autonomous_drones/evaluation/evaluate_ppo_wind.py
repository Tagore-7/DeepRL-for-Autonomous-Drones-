"""
evaluate_ppo_wind.py
"""



import os
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
import glob
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from envs.Drone_Controller_RPM import DroneControllerRPM
import matplotlib.pyplot as plt 
import seaborn as sns

COST_KEYS = {
    "total": "cost",
    "tilt": "tilt_cost",
    "spin": "spin_cost",
    "lidar": "lidar_cost",
}


def make_env(task, seed=0, wind_scale=1.0, graphics=True, render_mode="human"):
    env = gym.make(task, render_mode=render_mode, graphics=graphics)
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
    # env.render()
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

            # env.render()

            totals["reward"] += r
            for key, inf_key in COST_KEYS.items():
                totals[key] += info.get(inf_key, 0.0)

            ep_len += 1
            done = term or trunc

        for k in totals:
            stats[k].append(totals[k])

        lens.append(ep_len)
    # env.close()

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
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
           "--graphics",
            action=argparse.BooleanOptionalAction,
           default=True,
            help="Show PyBullet GUI (default: True). Use --no-graphics to disable.",
       )    
    parser.add_argument(
            "--render_mode",
            choices=["human", "rgb_array", "none"],
            default="human",
            help="Render‑mode to give gym.make(); 'none' → Python None",
        )    
    args = parser.parse_args()

    wind_levels = np.linspace(0.0, 1.0, num=11)
    all_results = []

    # ckpt = find_latest_checkpoint(args.root, "PPO", args.task)
    ckpt = "deepRL_for_autonomous_drones/training/best_ppo_with_no_wind.zip"

    if ckpt is None:
        print("No checkpoint for PPO")
        return
    
    render_mode = None if args.render_mode.lower() == "none" else args.render_mode

    env = make_env(args.task, seed=args.seed, wind_scale=0.0, graphics=args.graphics, render_mode=render_mode)

    model = PPO.load(ckpt, device="cpu") 

    if args.graphics and render_mode == "human":
        env.render()

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
    out_path = os.path.join("deepRL_for_autonomous_drones/evaluation/csv_files", "ppo_wind_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved wind sweep evaluation to {out_path}")

    sns.set_style("whitegrid")
    fig, (ax_r, ax_c) = plt.subplots(2, 1, sharex=True, figsize=(6, 4))

    ax_r.plot(df["wind"], df["reward_mean"],
            lw=2, color="steelblue", label="Reward")
    ax_r.fill_between(df["wind"],
                    df["reward_mean"] - df["reward_std"],
                    df["reward_mean"] + df["reward_std"],
                    alpha=0.3, color="steelblue")
    ax_r.set_ylabel("Reward")

    ax_c.plot(df["wind"], df["total_mean"],
            lw=2, color="firebrick", label="Cost")
    ax_c.fill_between(df["wind"],
                    df["total_mean"] - df["total_std"],
                    df["total_mean"] + df["total_std"],
                    alpha=0.3, color="firebrick")
    ax_c.set_ylabel("Cost")
    ax_c.set_xlabel("Wind Scale")

    fig.suptitle("Drone‑Run", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)      

    out_img = "deepRL_for_autonomous_drones/pics/ppo_wind_curve.png"
    fig.savefig(out_img, dpi=150)
    print(f"Figure saved to {out_img}")
    # plt.show()  


if __name__ == "__main__":
    main()


