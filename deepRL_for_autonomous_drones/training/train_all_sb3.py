"""
Train PPO, A2C, DDPG, TD3, SAC, ARS, CroosQ, and TQC 
on "SafetyDroneLanding-v0", saving each policy in its own folder 
"""

import os
import time
import argparse
import multiprocessing
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from sb3_contrib import ARS, CrossQ, TQC
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from deepRL_for_autonomous_drones.envs.Drone_Controller_RPM import DroneControllerRPM

ALGOS = {
    "PPO": PPO,
    "A2C": A2C,
    "DDPG": DDPG,
    "TD3": TD3,
    "SAC": SAC,
    "ARS": ARS,
    "CrossQ": CrossQ,
    "TQC": TQC,
}


def make_env(task_id: str, seed: int = 42):
    def _init():
        env = gym.make(task_id)
        env = FlattenObservation(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="SafetyDroneLanding-v0")
    parser.add_argument("--algos", nargs="+", default=list(ALGOS.keys()))
    parser.add_argument("--total-steps", type=int, default=5_000_000, help="traning timesetps per algortihm")
    parser.add_argument("--num_envs", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", default="../evaluation/sb3_runs_nonconstrained")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    ts_start = int(time.time())

    vec_env = SubprocVecEnv([make_env(args.task, args.seed + i) for i in range(args.num_envs)])

    for algo_name in args.algos:
        algo_cls = ALGOS[algo_name]
        print(f"\n Training {algo_name} for {args.total_steps:,} steps ...")
        model = algo_cls(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            seed=args.seed,
            tensorboard_log=os.path.join(args.logdir, "tensorboard"),
            # device="cpu",
        )

        model.learn(total_timesteps=args.total_steps, progress_bar=True)

        ckpt_dir = os.path.join(args.logdir, f"{algo_name}_{args.task}_{ts_start}_{args.seed}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save(os.path.join(ckpt_dir, "policy.zip"))
        print(f"{algo_name} saved to {ckpt_dir}")

    vec_env.close()
    print("\nAll done!")


if __name__ == "__main__":
    main()
