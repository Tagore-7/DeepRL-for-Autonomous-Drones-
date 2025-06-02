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
    path: str = "../training/benchmark_results/fast-safe-rl/SafetyDroneLanding-v0-cost-50/ppol-c1dd"
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
    os.makedirs("evaluations", exist_ok=True)
    out_path = os.path.join("evaluations", "wind_results.csv")
    df.to_csv(out_path, index=False)


# @pyrallis.wrap()
# def evaluate(args: EvalConfig = EvalConfig):
#     cfg, model = load_config_and_model(args.path, args.best)

#     task = cfg["task"]
#     demo_env = make_env(task, graphics=False)

#     agent = PPOLagAgent(
#         env=demo_env,
#         logger=BaseLogger(),
#         device=args.device,
#         use_lagrangian=cfg["use_lagrangian"],
#         thread=cfg["thread"],
#         seed=cfg["seed"],
#         cost_limit=cfg["cost_limit"],  # Original FSRL Script did not have this for PPOL?
#         hidden_sizes=cfg["hidden_sizes"],
#         unbounded=cfg["unbounded"],
#         last_layer_scale=cfg["last_layer_scale"],
#     )

#     if args.parallel_eval:
#         assert args.render is False, "please use single env when rendering"
#         test_envs = ShmemVectorEnv([lambda: make_env(task, graphics=False) for _ in range(args.eval_episodes)])
#     else:
#         test_envs = make_env(task, graphics=args.render)

#     rews, lens, cost = agent.evaluate(
#         test_envs=test_envs,
#         state_dict=model["model"],
#         eval_episodes=args.eval_episodes,
#         render=args.render,
#         train_mode=args.train_mode,
#     )
#     print("Traing mode: ", args.train_mode)
#     print(f"Eval reward: {rews}, cost: {cost}, length: {lens}")

#     csv_path = "evaluations"
#     os.makedirs(csv_path, exist_ok=True)
#     df = pd.DataFrame([{"task": task, "algo": cfg["prefix"], "seed": cfg["seed"], "reward": rews, "cost": cost, "length": lens}])
#     # Write or append
#     csv_file = "evaluations.csv"
#     path = os.path.join(csv_path, csv_file)
#     if not os.path.isfile(path):
#         df.to_csv(path, index=False)
#     else:
#         df.to_csv(path, mode="a", header=False, index=False)
#     print(f"Final eval CSV saved to {csv_path}")


# @pyrallis.wrap()
# def evaluate(args: EvalConfig = EvalConfig):
#     cfg, model = load_config_and_model(args.path, args.best)

#     task = cfg["task"]
#     demo_env = make_env(task, graphics=False, wind_scale=0.0)

#     agent = PPOLagAgent(
#         env=demo_env,
#         logger=BaseLogger(),
#         device=args.device,
#         use_lagrangian=cfg["use_lagrangian"],
#         thread=cfg["thread"],
#         seed=cfg["seed"],
#         cost_limit=cfg["cost_limit"],  # Original FSRL Script did not have this for PPOL?
#         hidden_sizes=cfg["hidden_sizes"],
#         unbounded=cfg["unbounded"],
#         last_layer_scale=cfg["last_layer_scale"],
#     )

#     wind_levels = np.linspace(0.0, 1.0, num=11)  # [0.0, 0.1, ..., 1.0]
#     results = []

#     test_envs = None  # to store reference for cleanup

#     for wind in wind_levels:
#         print(f"Evaluating at wind level {wind:.1f}")

#         # Close the previous environment explicitly
#         # if test_envs is not None:
#         #     if isinstance(test_envs, SubprocVectorEnv):
#         #         test_envs.close()
#         #     else:
#         #         test_envs.close()

#         #         if hasattr(test_env.unwrapped, "setWindScale"):
#         # #             test_env.unwrapped.setWindScale(wind)

#         # if args.parallel_eval:
#         #     assert args.render is False, "please use single env when rendering"
#         #     # test_envs = ShmemVectorEnv([lambda: make_env(task, graphics=False, wind_scale=wind) for _ in range(args.eval_episodes)])
#         #     env_fns = [lambda w=wind: make_env(task, graphics=False, wind_scale=w) for _ in range(args.eval_episodes)]
#         #     test_envs = ShmemVectorEnv(env_fns)
#         # else:
#         #     test_envs = make_env(task, graphics=args.render, wind_scale=wind)
#         if args.parallel_eval:
#             env_fns = [partial(make_env, task, graphics=False, wind_scale=wind) for _ in range(args.eval_episodes)]
#             test_envs = SubprocVectorEnv(env_fns)
#             # print("worker wind scales:", [e.unwrapped.wind_force_scale for e in test_envs.envs])  # or whatever attr
#         else:
#             test_envs = make_env(task, graphics=args.render, wind_scale=wind)

#         test_envs.reset()  # spawns workers

#         rews, lens, cost = agent.evaluate(
#             test_envs=test_envs,
#             state_dict=model["model"],
#             eval_episodes=args.eval_episodes,
#             render=args.render,
#             train_mode=args.train_mode,
#         )

#         print(f"Eval reward: {rews}, cost: {cost}, length: {lens}")

#         # results.append(
#         #     {
#         #         "wind": wind,
#         #         "reward": rews,
#         #         "cost": cost,
#         #         "length": lens,
#         #         "algo": cfg["prefix"],
#         #         "seed": cfg["seed"],
#         #         "cost_limit": cfg["cost_limit"],
#         #     }
#         # )
#         results.append(
#             {
#                 "wind": wind,
#                 "reward": float(np.mean(rews)),
#                 "cost": float(np.mean(cost)),
#                 "length": float(np.mean(lens)),
#                 "algo": cfg["prefix"],
#                 "seed": cfg["seed"],
#                 "cost_limit": cfg["cost_limit"],
#             }
#         )

#         # csv_path = "evaluations"
#         # os.makedirs(csv_path, exist_ok=True)
#         # df = pd.DataFrame([{"task": task, "algo": cfg["prefix"], "seed": cfg["seed"], "reward": rews, "cost": cost, "length": lens}])
#         # # Write or append
#         # csv_file = "evaluations.csv"
#         # path = os.path.join(csv_path, csv_file)
#         # if not os.path.isfile(path):
#         #     df.to_csv(path, index=False)
#         # else:
#         #     df.to_csv(path, mode="a", header=False, index=False)
#         # print(f"Final eval CSV saved to {csv_path}")
#     df = pd.DataFrame(results)
#     os.makedirs("evaluations", exist_ok=True)
#     out_path = os.path.join("evaluations", "wind_results.csv")
#     df.to_csv(out_path, index=False)


if __name__ == "__main__":
    evaluate()
