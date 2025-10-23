import os
from dataclasses import asdict
from typing import Dict, Callable
import multiprocessing

import gymnasium as gym
import numpy as np
import pyrallis
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.env_util import make_vec_env

import deepRL_for_autonomous_drones
from deepRL_for_autonomous_drones.config.ppo_cfg import DroneLandingCfg
import deepRL_for_autonomous_drones.envs
from gymnasium.wrappers import FlattenObservation

# ────────────────────────────── helpers ───────────────────────────────────


def linear_schedule(initial_value: float, end_value: float):
    def func(progress_remaining: float) -> float:
        return end_value + (initial_value - end_value) * progress_remaining

    return func


WORKERS: Dict[str, Callable] = {
    "SubprocVectorEnv": SubprocVecEnv,
    "ShmemVectorEnv": SubprocVecEnv,
    "DummyVectorEnv": DummyVecEnv,
}


def make_env(
    task: str,
    seed: int = 42,
    wind_level: str = "none",
):
    def _init():
        env = gym.make(task)
        if hasattr(env.unwrapped, "setWindEffects"):
            env.unwrapped.setWindEffects(wind_level != "none")
        if hasattr(env.unwrapped, "setWindLevel"):
            env.unwrapped.setWindLevel(wind_level)
        env = FlattenObservation(env)
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def train_phase(cfg: DroneLandingCfg, wind_level: str, pretrained_model_path: str = None):
    num_cpu = multiprocessing.cpu_count()
    print(f"Number of CPU cores available: {num_cpu}")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    set_random_seed(cfg.seed)

    WorkerCls = WORKERS[cfg.worker]
    env_fns = [make_env(cfg.task, cfg.seed + i, wind_level) for i in range(cfg.training_num)]
    train_envs = WorkerCls(env_fns)
    eval_env = WorkerCls([make_env(cfg.task, cfg.seed + 10000 + i, wind_level) for i in range(cfg.testing_num)])

    log_dir = os.path.join(cfg.logdir, cfg.name)

    eval_callback = EvalCallback(
        eval_env,
        # best_model_save_path=f"{cfg.default_output_path}",
        best_model_save_path=log_dir,
        log_path=log_dir,
        # eval_freq=10000,
        eval_freq=5000,
        # eval_freq=cfg.step_per_epoch * cfg.training_num,
        n_eval_episodes=10,  # match num of eval environments?
        verbose=1,
    )

    checkpoint_dir = os.path.join(log_dir, "./checkpoint")
    checkpoint_callback = CheckpointCallback(
        # save_freq=50_000,
        save_freq=10_000,
        save_path=checkpoint_dir,
        save_replay_buffer=True,
        verbose=1,
    )

    policy_kwargs = dict(net_arch=list(cfg.hidden_sizes))

    tensorboard_log_dir = os.path.join(log_dir, "tensorboard")
    new_logger = configure(tensorboard_log_dir, ["stdout", "log", "tensorboard"])

    clip_schedule = get_linear_fn(0.25, 0.1, 1.0)
    # agent = PPO(
    #     policy="MlpPolicy",
    #     env=train_envs,
    #     # learning_rate=cfg.lr,
    #     learning_rate=linear_schedule(4e-4, 2e-5),
    #     # n_steps=cfg.step_per_epoch // cfg.training_num,
    #     n_steps=cfg.step_per_epoch,
    #     batch_size=cfg.batch_size,
    #     n_epochs=cfg.repeat_per_collect,
    #     gamma=cfg.gamma,
    #     gae_lambda=cfg.gae_lambda,
    #     # clip_range=cfg.eps_clip,
    #     clip_range=clip_schedule,
    #     ent_coef=0.0,
    #     vf_coef=cfg.vf_coef,
    #     max_grad_norm=cfg.max_grad_norm,
    #     normalize_advantage=cfg.norm_adv,
    #     target_kl=cfg.target_kl,
    #     policy_kwargs=policy_kwargs,
    #     seed=cfg.seed,
    #     device=cfg.device,
    #     verbose=cfg.verbose,
    #     # tensorboard_log=log_dir
    # )
    # resume_path = "./benchmark_results/add_ran_pad_dyn_trees/best_model"
    resume_path = "./benchmark_results/add_ran_pad_dyn_trees/checkpoint/model"
    custom_objects = {"learning_rate": 3e-5, "clip_range": 0.1}
    agent = PPO.load(resume_path, env=train_envs, device=cfg.device, custom_objects=custom_objects)

    agent.set_logger(new_logger)

    # if pretrained_model_path:
    #     agent.set_parameters(pretrained_model_path)

    total_timesteps = 10_000_000
    agent.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=[eval_callback, checkpoint_callback],
        # reset_num_timesteps=False
    )

    save_path = os.path.join(log_dir, "checkpoint", f"model")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"Saved model for wind level '{wind_level}' to {save_path}")
    train_envs.close()
    return save_path


def run_curriculum(cfg: DroneLandingCfg):
    cfg.epoch = 1500
    base_model_path = train_phase(cfg, wind_level="none", pretrained_model_path=None)


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=DroneLandingCfg)
    run_curriculum(cfg)
