import os
import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass
from typing import Optional
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation
from tqdm import trange

import deepRL_for_autonomous_drones.envs  


def make_env(task: str, wind_level: str = "none", render: bool = False):
    render_mode = "human" if render else None
    env = gym.make(task, render_mode=render_mode)
    env = FlattenObservation(env)
    if hasattr(env.unwrapped, "setWindEffects"):
        env.unwrapped.setWindEffects(True)
    if hasattr(env.unwrapped, "setWindLevel"):
        env.unwrapped.setWindLevel(wind_level)
    return env


@dataclass
class EvalConfig:
    model_path: str = "deepRL_for_autonomous_drones/training/results/fast-safe-rl/SafetyDroneLanding-v0-cost-0/ppo-370b/checkpoint/ppo_model.zip"
    task: str = "SafetyDroneLanding-v0"
    wind_levels: tuple = ("none", "light_breeze", "light", "medium", "high")
    eval_episodes: int = 20
    device: str = "cpu"
    render: bool = False
    output_csv: str = "wind_eval_sb3_ppo.csv"


def evaluate(cfg: EvalConfig):
    print(f"Loading model from: {cfg.model_path}")
    model = PPO.load(cfg.model_path, device=cfg.device)

    results = []

    for wind in cfg.wind_levels:
        print(f"\nEvaluating wind level: {wind}")
        env = DummyVecEnv([lambda: make_env(cfg.task, wind, render=cfg.render)])

        ep_rewards, ep_lengths, ep_costs = [], [], []

        for _ in trange(cfg.eval_episodes, desc=f"Wind: {wind}"):
            obs = env.reset()
            done = False
            total_reward = 0.0
            total_cost = 0.0
            steps = 0

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                total_cost = info[0].get("cost", None)
                steps += 1

                if done[0]:
                    break

            ep_rewards.append(total_reward)
            ep_lengths.append(steps)
            ep_costs.append(total_cost)

        results.append({
            "wind_level": wind,
            "reward": np.mean(ep_rewards),
            "cost" : np.mean(ep_costs),
            "length": np.mean(ep_lengths),
        })

        env.close()

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(cfg.output_csv), exist_ok=True)
    df.to_csv(cfg.output_csv, index=False)
    print(f"\nResults saved to: {cfg.output_csv}")


if __name__ == "__main__":
    config = EvalConfig(
        output_csv="deepRL_for_autonomous_drones/evaluation/evaluations/sb3_ppo_wind_results.csv",
        eval_episodes=20,
        device="cpu",  # or "cpu"
        render=False
    )
    evaluate(config)
