import os
import pandas as pd
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import bullet_safety_gym

try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import pyrallis
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv

from fsrl.agent import CPOAgent
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, load_config_and_model, seed_all


def make_env(task, graphics: bool):
    render_mode = "human" if graphics else None
    env = gym.make(task, render_mode=render_mode, graphics=graphics)
    env = FlattenObservation(env)
    env.reset()
    return env


@dataclass
class EvalConfig:
    path: str = "../training/benchmark_results"
    best: bool = True
    eval_episodes: int = 20
    parallel_eval: bool = True
    device: str = "cpu"
    render: bool = False
    train_mode: bool = False


@pyrallis.wrap()
def evaluate(args: EvalConfig = EvalConfig):
    cfg, model = load_config_and_model(args.path, args.best)

    task = cfg["task"]
    demo_env = make_env(task, graphics=False)

    agent = CPOAgent(
        env=demo_env,
        logger=BaseLogger(),
        device=args.device,
        thread=cfg["thread"],
        seed=cfg["seed"],
        cost_limit=cfg["cost_limit"],
        hidden_sizes=cfg["hidden_sizes"],
        unbounded=cfg["unbounded"],
        last_layer_scale=cfg["last_layer_scale"],
    )

    if args.parallel_eval:
        assert args.render is False, "please use single env when rendering"
        test_envs = ShmemVectorEnv([lambda: make_env(task, graphics=False) for _ in range(args.eval_episodes)])
    else:
        test_envs = make_env(task, graphics=args.render)

    rews, lens, cost = agent.evaluate(
        test_envs=test_envs,
        state_dict=model["model"],
        eval_episodes=args.eval_episodes,
        render=args.render,
        train_mode=args.train_mode,
    )
    print("Traing mode: ", args.train_mode)
    print(f"Eval reward: {rews}, cost: {cost}, length: {lens}")

    csv_path = "evaluations"
    os.makedirs(csv_path, exist_ok=True)
    df = pd.DataFrame([{"task": task, "algo": cfg["prefix"], "seed": cfg["seed"], "reward": rews, "cost": cost, "length": lens}])
    # Write or append
    csv_file = "evaluations.csv"
    path = os.path.join(csv_path, csv_file)
    if not os.path.isfile(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)
    print(f"Final eval CSV saved to {csv_path}")


if __name__ == "__main__":
    evaluate()
