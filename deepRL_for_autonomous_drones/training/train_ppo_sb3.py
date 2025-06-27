from __future__ import annotations

"""train_ppo_agent_sb3.py

Trains PPO from Stable-Baselines3 on the drone landing task.
Configuration mirrors `train_ppol_agent.py` for comparison.
"""

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
import deepRL_for_autonomous_drones.envs 
from gymnasium.wrappers import FlattenObservation


# ────────────────────────────── helpers ───────────────────────────────────

WORKERS: Dict[str, Callable] = {
    "SubprocVectorEnv": SubprocVecEnv,
    "ShmemVectorEnv": SubprocVecEnv, 
    "DummyVectorEnv": DummyVecEnv,
}

def make_env(task: str):
    """Factory that builds a *training* environment with **wind disabled**."""
    def _init():
        env = gym.make(task)
        if hasattr(env.unwrapped, "setWindEffects"):
            env.unwrapped.setWindEffects(False)
        env = FlattenObservation(env)
        return Monitor(env)
    return _init


# ────────────────────────────── main entry ────────────────────────────────

def main(cfg: DroneLandingCfg):
    """Train SB3 PPO according to *cfg* on the drone-landing task."""

    # reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # build vectorised envs
    WorkerCls = WORKERS[cfg.worker]
    train_envs = WorkerCls([make_env(cfg.task) for _ in range(cfg.training_num)])
    test_envs = WorkerCls([make_env(cfg.task) for _ in range(cfg.testing_num)])

    # logger (WandB by default ‑ change here if you prefer TensorBoard)
    default_cfg = asdict(DroneLandingCfg())
    run_name = cfg.name or auto_name(default_cfg, asdict(cfg), cfg.prefix, cfg.suffix)
    group_name = cfg.group or f"{cfg.task}-cost-{cfg.cost_limit}"
    log_dir = os.path.join(cfg.logdir, cfg.project, group_name) if cfg.logdir else None
    logger = WandbLogger(cfg, cfg.project, group_name, run_name, log_dir)
    logger.save_config(asdict(cfg))

    # SB3-compatible policy architecture
    policy_kwargs = dict(
        net_arch=list(cfg.hidden_sizes)
    )

    # define SB3 PPO agent
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
        tensorboard_log=log_dir,           
    )

    # training
    total_timesteps = cfg.epoch * cfg.step_per_epoch
    agent.learn(
        total_timesteps=total_timesteps, 
        progress_bar=True,
    )

    save_path = os.path.join(log_dir, "checkpoint", "ppo_model")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"Model saved to: {save_path}")

    train_envs.close()
    test_envs.close()


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=DroneLandingCfg)
    main(cfg)
