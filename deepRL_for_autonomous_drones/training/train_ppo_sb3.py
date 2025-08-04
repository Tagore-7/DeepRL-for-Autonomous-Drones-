import os
from dataclasses import asdict
from typing import Dict, Callable

import gymnasium as gym
import numpy as np
import pyrallis
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from fsrl.utils.logger import WandbLogger
from fsrl.utils.exp_util import auto_name
import deepRL_for_autonomous_drones
from deepRL_for_autonomous_drones.config.ppo_cfg import DroneLandingCfg
from deepRL_for_autonomous_drones import envs
from gymnasium.wrappers import FlattenObservation

# ────────────────────────────── helpers ───────────────────────────────────

WORKERS: Dict[str, Callable] = {
    "SubprocVectorEnv": SubprocVecEnv,
    "ShmemVectorEnv": SubprocVecEnv, 
    "DummyVectorEnv": DummyVecEnv,
}

def make_env(task: str, wind_level: str = "none",reward_function_id: int = 1 ):
    def _init():
        env = gym.make(task)
        if hasattr(env.unwrapped, "setWindEffects"):
            env.unwrapped.setWindEffects(wind_level != "none")
        if hasattr(env.unwrapped, "setWindLevel"):
            env.unwrapped.setWindLevel(wind_level)
        env = FlattenObservation(env)
        return Monitor(env)
    return _init

def train_phase(cfg: DroneLandingCfg, wind_level: str, pretrained_model_path: str = None):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    WorkerCls = WORKERS[cfg.worker]
    env_fns = [make_env(cfg.task, wind_level) for _ in range(cfg.training_num)]
    train_envs = WorkerCls(env_fns)

    default_cfg = asdict(DroneLandingCfg())
    name = cfg.name or auto_name(default_cfg, asdict(cfg), cfg.prefix, cfg.suffix)
    group = cfg.group or f"{cfg.task}-cost-{cfg.cost_limit}-wind-{wind_level}"
    log_dir = os.path.join(cfg.logdir, cfg.project, group)
    logger = WandbLogger(cfg, cfg.project, group, name, log_dir)
    logger.save_config(asdict(cfg))

    policy_kwargs = dict(net_arch=list(cfg.hidden_sizes))

    agent = PPO(
        policy="MlpPolicy",
        env=train_envs,
        learning_rate=cfg.lr,
        n_steps=cfg.step_per_epoch // cfg.training_num,
        batch_size=cfg.batch_size,
        n_epochs=cfg.repeat_per_collect,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.eps_clip,
        ent_coef=0.0,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        normalize_advantage=cfg.norm_adv,
        target_kl=cfg.target_kl,
        policy_kwargs=policy_kwargs,
        seed=cfg.seed,
        device=cfg.device,
        verbose=cfg.verbose,
        tensorboard_log=log_dir
    )

    if pretrained_model_path:
        agent.set_parameters(pretrained_model_path)

    agent.learn(total_timesteps=cfg.epoch * cfg.step_per_epoch, progress_bar=True)

    save_path = os.path.join(log_dir, "checkpoint", f"ppo_model_wind_{wind_level}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"Saved model for wind level '{wind_level}' to {save_path}")
    train_envs.close()
    return save_path

def run_curriculum(cfg: DroneLandingCfg):
    # Phase 1: Normal training (no wind)
    cfg.epoch = 750
    base_model_path = train_phase(cfg, wind_level="none", pretrained_model_path=None)

    # Phase 2: Wind curriculum
    wind_stages = ["light_breeze", "light_wind", "medium_wind", "high_wind"]
    cfg.epoch = 750
    for wind_level in wind_stages:
        base_model_path = train_phase(cfg, wind_level=wind_level, pretrained_model_path=base_model_path)

if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=DroneLandingCfg)
    run_curriculum(cfg)
