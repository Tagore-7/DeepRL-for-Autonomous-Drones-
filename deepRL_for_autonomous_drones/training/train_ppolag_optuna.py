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

from fsrl.agent import PPOLagAgent
from fsrl.utils.exp_util import seed_all
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from tianshou.env import BaseVectorEnv, DummyVectorEnv, RayVectorEnv, ShmemVectorEnv, SubprocVectorEnv

WORKER_MAPPING = {
    "BaseVectorEnv": BaseVectorEnv,
    "DummyVectorEnv": DummyVectorEnv,
    "RayVectorEnv": RayVectorEnv,
    "SubprocVectorEnv": SubprocVectorEnv,
    "ShmemVectorEnv": ShmemVectorEnv,
}


def make_env(task, seed: int = 42, graphics: bool = False):
    render_mode = "human" if graphics else None
    env = gym.make(task, render_mode=render_mode, graphics=graphics)
    env = FlattenObservation(env)
    # env.reset(seed=seed)
    return env


# def sample_ppo_hparams(trial: optuna.Trial) -> Dict[str, Any]:
#     return {
#         "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
#         "gamma": trial.suggest_float("gamma", 0.95, 0.999, step=0.001),
#         "lr": trial.suggest_float("lr", 1e-5, 3e-4, log=True),
#         "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.98),
#         "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
#         "eps_clip": trial.suggest_float("eps_clip", 0.1, 0.3),
#         "hidden_sizes": tuple([trial.suggest_categorical("hidden_sizes", [64, 128, 256])] * 2),
#         "training_num": trial.suggest_int("training_num", 8, 20, step=4),
#         "step_per_epoch": trial.suggest_int("step_per_epoch", 8000, 20000, step=2000),
#         "repeat_per_collect": trial.suggest_int("repeat_per_collect", 1, 10),
#         "cost_limit": trial.suggest_categorical("cost_limit", [5, 10, 20]),
#         "target_kl": trial.suggest_float("target_kl", 0.02, 0.08, step=0.02),
#     }


def sample_ppo_hparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Search only the variables that still look uncertain and
    keep the rest close to the Pareto-winner you selected."""
    return {
        # freeze parameters that were already near-optimal
        "batch_size": 64,
        "hidden_sizes": (128, 128),
        "step_per_epoch": 20_000,
        "cost_limit": 5,
        "gamma": trial.suggest_float("gamma", 0.96, 0.99),
        "repeat_per_collect": trial.suggest_int("repeat_per_collect", 7, 11),
        "lr": trial.suggest_float("lr", 1e-5, 4e-5, log=True),
        "eps_clip": trial.suggest_float("eps_clip", 0.18, 0.26),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.94, 0.98),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "target_kl": trial.suggest_float("target_kl", 0.04, 0.08, step=0.02),
        "training_num": trial.suggest_int("training_num", 8, 16, step=4),
    }


def evaluate_agent(agent, env, episodes=10):
    rews, lens, costs = agent.evaluate(
        test_envs=env,
        eval_episodes=episodes,
        render=False,
        train_mode=False,
    )
    return float(rews), float(costs)


def objective(trial: optuna.Trial, task: str, seed: int, train_epochs: int):
    try:
        params = sample_ppo_hparams(trial)
        training_num = params.pop("training_num")
        batch_size = params.pop("batch_size")
        step_per_epoch = params.pop("step_per_epoch")
        repeat_per_collect = params.pop("repeat_per_collect")

        demo_env = make_env(task, seed=seed)
        seed_all(seed)

        logger = TensorboardLogger("tensorboard", log_txt=True, name="PPOL_optuna")

        agent = PPOLagAgent(
            env=demo_env,
            logger=logger,
            device="cpu",
            seed=seed,
            use_lagrangian=True,
            **params,  # n_steps, lr, etc.
        )

        training_num_workers = min(training_num, 20)
        worker = WORKER_MAPPING.get("SubprocVectorEnv")
        if worker is None:
            raise ValueError("Unknown worker type")

        train_envs = worker([lambda: make_env(task) for _ in range(training_num_workers)])

        agent.learn(
            train_envs=train_envs,
            epoch=train_epochs,
            batch_size=batch_size,
            step_per_epoch=step_per_epoch,
            repeat_per_collect=repeat_per_collect,
            verbose=True,
        )

        demo_env.close()
        train_envs.close()

        eval_env = make_env(task)
        reward, cost = evaluate_agent(agent, eval_env, episodes=10)
        eval_env.close()

        print(f"[Trial {trial.number}] Reward: {reward:.2f}, Cost: {cost:.2f}")
        trial.set_user_attr("mean_reward", reward)
        trial.set_user_attr("mean_cost", cost)

        return reward, cost
    except Exception as e:
        print(f"[Trial {trial.number}] FAILED with error: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="SafetyDroneLanding-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--train-epochs", type=int, default=60)
    parser.add_argument("--storage", default="sqlite:///ppo_moo.db", help="Optuna storage URI")
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=f"PPOL-{args.task}-MOO",
        directions=("maximize", "minimize"),
        sampler=optuna.samplers.NSGAIISampler(seed=args.seed),
        storage=args.storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda t: objective(t, args.task, args.seed, args.train_epochs),
        n_trials=args.trials,
    )

    df = study.trials_dataframe()
    df.to_csv("ppol_moo_trials.csv", index=False)
    print("Parento-optimal (reward, cost):")
    for t in study.best_trials:
        print(f" Trial {t.number}: R ={t.values[0]:.2f}, C={t.values[1]:.2f}")
