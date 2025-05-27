import os
import pandas as pd
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
import numpy as np

import bullet_safety_gym

try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import pyrallis
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv

from fsrl.agent import PPOLagAgent
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, load_config_and_model, seed_all

from deepRL_for_autonomous_drones.envs.Drone_Controller_RPM import DroneControllerRPM
from deepRL_for_autonomous_drones.envs.Base_Drone_Controller import BaseDroneController


def make_env(task, graphics: bool = False, wind_scale: float = 1.0):
    render_mode = "human" if graphics else None
    env = gym.make(task, render_mode=render_mode, graphics=graphics)
    env = FlattenObservation(env)

    env.unwrapped.setWindEffects(True)
    if hasattr(env.unwrapped, "setWindScale"):
        env.unwrapped.setWindScale(wind_scale)

    return env


@dataclass
class EvalConfig:
    path: str = "../training/benchmark_results/fast-safe-rl/SafetyDroneLanding-v0-cost-50/ppol-b434"
    best: bool = True
    eval_episodes: int = 1
    parallel_eval: bool = False
    device: str = "cpu"
    render: bool = True
    train_mode: bool = False


@pyrallis.wrap()
def evaluate(args: EvalConfig = EvalConfig):
    cfg, model = load_config_and_model(args.path, args.best)
    task = cfg["task"]

    # ---- create a single environment once (reusing GUI) ----#
    test_env = make_env(task, graphics=args.render, wind_scale=0.0)

    agent = PPOLagAgent(
        env=test_env,
        logger=BaseLogger(),
        device=args.device,
        use_lagrangian=cfg["use_lagrangian"],
        thread=cfg["thread"],
        seed=cfg["seed"],
        cost_limit=cfg["cost_limit"],
        hidden_sizes=cfg["hidden_sizes"],
        unbounded=cfg["unbounded"],
        last_layer_scale=cfg["last_layer_scale"],
    )

    wind_levels = np.linspace(0.0, 1.0, num=11)
    results = []

    for wind in wind_levels:
        print(f"Evaluating at wind level {wind:.1f}")

        if hasattr(test_env.unwrapped, "setWindScale"):
            test_env.unwrapped.setWindScale(wind)

        rews, lens, cost = agent.evaluate(
            test_envs=test_env,
            state_dict=model["model"],
            eval_episodes=args.eval_episodes,
            render=args.render,
            train_mode=args.train_mode,
        )

        print(f"Eval reward: {rews}, cost: {cost}, length: {lens}")

        results.append(
            {
                "wind": wind,
                "reward": rews,
                "cost": cost,
                "length": lens,
            }
        )

    df = pd.DataFrame(results)
    df["task"] = task
    df["algo"] = cfg["prefix"]
    df["seed"] = cfg["seed"]

    os.makedirs("evaluations", exist_ok=True)
    out_path = os.path.join("evaluations", "wind_results.csv")
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    evaluate()
