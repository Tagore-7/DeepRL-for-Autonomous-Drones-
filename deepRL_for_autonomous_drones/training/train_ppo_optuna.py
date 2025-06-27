"""
Multi-objective hyper-parameter tuning for PPO on SafetyDroneLanding-v0
using Optuna.  Objective:
1. Maximise average episodic reward (return)
2. Minimize average episodic safety cost (info["cost"])
"""

import argparse
import os
from typing import Dict, Any
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
import optuna
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from deepRL_for_autonomous_drones import envs

# from deepRL_for_autonomous_drones.envs.Drone_Controller_RPM import DroneControllerRPM


def make_env(task: str, seed: int):
    def _init():
        env = gym.make(task)
        env = FlattenObservation(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def sample_ppo_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_steps": trial.suggest_int("n_steps", 512, 4096, step=512),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999, step=0.001),
        "learning_rate": trial.suggest_float("lr", 1e-5, 3e-4, log=True),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.98),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef": trial.suggest_float("ent_coef", 1e-5, 0.01, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "net_arch": [trial.suggest_categorical("hidden", [64, 128, 256])] * 2,
        "num_envs": trial.suggest_int("num_envs", 4, 48, step=4),
    }


def rollout_cost_reward(model, env, episodes: int = 20):
    rewards, costs = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        ep_r = ep_c = 0.0
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            ep_r += r
            ep_c += info.get("cost", 0.0)
            done = term or trunc
        rewards.append(ep_r)
        costs.append(ep_c)
    return float(np.mean(rewards)), float(np.mean(costs))


def objective(trial: optuna.Trial, task: str, seed: int, train_steps: int):
    params = sample_ppo_hparams(trial)
    net_arch = params.pop("net_arch")
    num_envs = params.pop("num_envs")

    vec_env = SubprocVecEnv([make_env(task, seed + i) for i in range(num_envs)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=net_arch),
        verbose=0,
        seed=seed,
        tensorboard_log=None,
        device="cpu",
        **params,
    )

    model.learn(total_timesteps=train_steps)

    vec_env.close()
    eval_env = make_env(task, seed + 10)()
    reward, cost = rollout_cost_reward(model, eval_env, episodes=10)
    eval_env.close()

    trial.set_user_attr("mean_reward", reward)
    trial.set_user_attr("mean_cost", cost)

    return reward, cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="SafetyDroneLanding-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--train-steps", type=int, default=300_000)
    parser.add_argument("--storage", default="sqlite:///ppo_moo.db", help="Optuna storage URI")
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=f"PPO-{args.task}-MOO",
        directions=("maximize", "minimize"),
        sampler=optuna.samplers.NSGAIISampler(seed=args.seed),
        storage=args.storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda t: objective(t, args.task, args.seed, args.train_steps),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    df = study.trials_dataframe()
    df.to_csv("ppo_moo_trials.csv", index=False)
    print("Parento-optimal (reward, cost):")
    for t in study.best_trials:
        print(f" Trail {t.number}: R ={t.values[0]:.2f}, C={t.values[1]:.2f}")
