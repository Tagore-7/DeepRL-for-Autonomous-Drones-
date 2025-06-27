from __future__ import annotations
"""train_ppo_agent.py
Plain **PPO** baseline (no Lagrangian safety terms).
"""
import os
from dataclasses import asdict
from typing import Dict, Callable

import gymnasium as gym
import numpy as np
import pyrallis
import torch
from fsrl.agent import PPOLagAgent  # behaves like plain PPO when ``use_lagrangian=False``
from fsrl.utils.logger import WandbLogger
from fsrl.utils.exp_util import auto_name
from tianshou.env import (
    SubprocVectorEnv,
    ShmemVectorEnv,
    DummyVectorEnv,
    BaseVectorEnv,
)

from deepRL_for_autonomous_drones.config.ppo_cfg import DroneLandingCfg
import deepRL_for_autonomous_drones.envs 
from gymnasium.wrappers import FlattenObservation



# ────────────────────────────── helpers ───────────────────────────────────

WORKERS: Dict[str, Callable] = {
    "SubprocVectorEnv": SubprocVectorEnv,
    "ShmemVectorEnv": ShmemVectorEnv,
    "DummyVectorEnv": DummyVectorEnv,
    "BaseVectorEnv": BaseVectorEnv,
}

def make_env(task: str):
    """Factory that builds a *training* environment with **wind disabled**."""
    env = gym.make(task)
    if hasattr(env.unwrapped, "setWindEffects"):
        env.unwrapped.setWindEffects(False)
    env = FlattenObservation(env)
    return env

# ────────────────────────────── main entry ────────────────────────────────

# @pyrallis.wrap(DroneLandingCfg)
def main(cfg: DroneLandingCfg):
    """Train PPO according to *cfg* on the drone‑landing task."""

    # reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # build vectorised envs
    WorkerCls = WORKERS[cfg.worker]
    train_envs = WorkerCls([lambda: make_env(cfg.task) for _ in range(cfg.training_num)])
    test_envs = WorkerCls([lambda: make_env(cfg.task) for _ in range(cfg.testing_num)])

    # logger (WandB by default ‑ change here if you prefer TensorBoard)
    default_cfg = asdict(DroneLandingCfg())
    run_name = cfg.name or auto_name(default_cfg, asdict(cfg), cfg.prefix, cfg.suffix)
    group_name = cfg.group or f"{cfg.task}-cost-{cfg.cost_limit}"
    log_dir = os.path.join(cfg.logdir, cfg.project, group_name) if cfg.logdir else None
    logger = WandbLogger(cfg, cfg.project, group_name, run_name, log_dir)
    logger.save_config(asdict(cfg))
    dummy_env = make_env(cfg.task)

    # agent – identical net sizes etc. as PPOL but ``use_lagrangian=False``
    agent = PPOLagAgent(
        env=dummy_env,
        logger=logger,
        device=cfg.device,
        thread=cfg.thread,
        seed=cfg.seed,
        # PPO‑specific
        lr=cfg.lr,
        hidden_sizes=cfg.hidden_sizes,
        unbounded=cfg.unbounded,
        last_layer_scale=cfg.last_layer_scale,
        target_kl=cfg.target_kl,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        gae_lambda=cfg.gae_lambda,
        eps_clip=cfg.eps_clip,
        dual_clip=cfg.dual_clip,
        value_clip=cfg.value_clip,
        advantage_normalization=cfg.norm_adv,
        recompute_advantage=cfg.recompute_adv,
        # disable Lagrangian parts → plain PPO
        use_lagrangian=False,
        cost_limit=cfg.cost_limit,
        rescaling=False,
        # generic
        gamma=cfg.gamma,
        max_batchsize=cfg.max_batchsize,
        reward_normalization=cfg.rew_norm,
        deterministic_eval=cfg.deterministic_eval,
        action_scaling=cfg.action_scaling,
        action_bound_method=cfg.action_bound_method,
    )

    # training loop (leverages FSRL's built‑in utility)
    agent.learn(
        train_envs=train_envs,
        test_envs=test_envs,
        epoch=cfg.epoch,
        step_per_epoch=cfg.step_per_epoch,
        repeat_per_collect=cfg.repeat_per_collect,
        episode_per_collect=cfg.episode_per_collect,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        testing_num=cfg.testing_num,
        reward_threshold=cfg.reward_threshold,
        save_interval=cfg.save_interval,
        resume=cfg.resume,
        save_ckpt=cfg.save_ckpt,
        verbose=cfg.verbose,
    )

    train_envs.close(); test_envs.close()


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=DroneLandingCfg)
    main(cfg)

