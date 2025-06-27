import os
import pandas as pd
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
import numpy as np
from functools import partial

import bullet_safety_gym

try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import pyrallis
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv, DummyVectorEnv

from fsrl.agent import PPOLagAgent
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, load_config_and_model, seed_all

from deepRL_for_autonomous_drones.envs.Drone_Controller_RPM import DroneControllerRPM
from deepRL_for_autonomous_drones.envs.Base_Drone_Controller import BaseDroneController



def make_env(task, graphics: bool = False, wind_level: str = "none"):
    render_mode = "human" if graphics else None
    env = gym.make(task, render_mode=render_mode, graphics=graphics)
    env = FlattenObservation(env)

    env.unwrapped.setWindEffects(True)
    if hasattr(env.unwrapped, "setWindLevel"):
        env.unwrapped.setWindLevel(wind_level)

    return env


@dataclass
class EvalConfig:

    path: str = "deepRL_for_autonomous_drones/training/results/fast-safe-rl/SafetyDroneLanding-v0-cost-0/ppo-working/"
    best: bool = True
    eval_episodes: int = 20
    parallel_eval: bool = False
    device: str = "cpu"
    render: bool = False 
    train_mode: bool = False


@pyrallis.wrap()
def evaluate(args: EvalConfig = EvalConfig):
    cfg, model = load_config_and_model(args.path, args.best)
    task = cfg["task"]

    # ---- create a single environment once (reusing GUI) ----#
    test_env = make_env(task, graphics=args.render, wind_level="none")

    agent = PPOLagAgent(
        env=test_env,
        logger=BaseLogger(),
        device=args.device,
        use_lagrangian=False,
        thread=cfg["thread"],
        seed=cfg["seed"],
        cost_limit=cfg["cost_limit"],
        hidden_sizes=cfg["hidden_sizes"],
        unbounded=cfg["unbounded"],
        last_layer_scale=cfg["last_layer_scale"],
    )

    # wind_levels = np.linspace(0.0, 1.0, num=11)
    results = []

    for level in ["none", "light_breeze", "light", "medium", "high"]:
        # for level in ["light_breeze", "light", "medium", "high"]:
        print(f"Evaluating at wind level: {level}")

        if hasattr(test_env.unwrapped, "setWindLevel"):
            test_env.unwrapped.setWindLevel(level)

        import torch
        ckpt_path = os.path.join(args.path, "checkpoint", "model_best.pt")
        checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=True)

        # Load only policy weights
        state_dict = checkpoint["model"]
        if "_extra_state" in state_dict and state_dict["_extra_state"] is None:
            state_dict["_extra_state"]  = {}# 👈 remove the broken key if it's None
        
        agent.policy.load_state_dict(state_dict)


        rews, lens, cost = agent.evaluate(
            test_envs=test_env,
            eval_episodes=args.eval_episodes,
            render=args.render,
            train_mode=args.train_mode,
        )

        print(f"Eval reward: {rews}, cost: {cost}, length: {lens}")

        results.append(
            {
                "wind_level": level,
                "reward": np.mean(rews),
                "cost": np.mean(cost),
                "length": lens,
                "algo": cfg["prefix"],
                "seed": cfg["seed"],
                "episodes": args.eval_episodes,
                "cost_limit": cfg["cost_limit"],
            }
        )

    df = pd.DataFrame(results)
    os.makedirs("deepRL_for_autonomous_drones/evaluation/evaluations/", exist_ok=True)
    out_path = os.path.join("deepRL_for_autonomous_drones/evaluation/evaluations/", "wind_results.csv")
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    evaluate()
