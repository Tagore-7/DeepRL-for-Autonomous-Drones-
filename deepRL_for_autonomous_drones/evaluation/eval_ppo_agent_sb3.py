import numpy as np
import pandas as pd
import random
import torch
import os
from stable_baselines3 import PPO, SAC
from gymnasium.wrappers import FlattenObservation, RecordVideo
import gymnasium as gym

# import seaborn as sns
import matplotlib.pyplot as plt
from deepRL_for_autonomous_drones import envs


def make_env(task: str, render: bool = False, seed: int = 42):
    render_mode = "human" if render else None
    env = gym.make(task, render_mode=render_mode)

    env = FlattenObservation(env)
    env.reset(seed=seed)
    return env


def eval_agent(model_path: str, env_name: str, render: bool = False, episodes: int = 20):

    base_seed = random.randint(1, 1000)
    # base_seed = 42
    model = PPO.load(model_path, device="cpu")
    env = make_env(env_name, render, base_seed)

    for episode in range(episodes):
        # random_seed = random.randint(1, 1000)
        # obs, _ = env.reset(seed=random_seed)
        eval_seed = base_seed + episode
        obs, _ = env.reset(seed=eval_seed)
        env.unwrapped.landed = False
        done = False
        total_reward, total_cost, steps = 0, 0, 0

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cost = info.get("cost", None)

            total_reward += reward
            total_cost += cost
            steps += 1


if __name__ == "__main__":
    eval_agent(
        model_path="../training/benchmark_results/add_seeding_trees/best_model",
        env_name="SafetyDroneLanding-v0",
        episodes=20,
        render=False,
    )
