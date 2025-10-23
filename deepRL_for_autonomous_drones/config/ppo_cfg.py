from dataclasses import dataclass
from typing import Optional, Tuple

"""
Plain‑PPO configuration (no Lagrangian / safety terms)
Style intentionally mirrors `ppol_cfg.py` so that experiments can be compared
one‑to‑one by keeping the same field names (`epoch`, `step_per_epoch`, etc.).
"""


@dataclass
class TrainCfg:
    # ───────────────────────── general task params ──────────────────────────
    task: str = "SafetyDroneLanding-v0"
    device: str = "cpu"
    thread: int = 128
    seed: int = 10

    # ─────────────────────────── algorithm params ───────────────────────────
    lr: float = 3e-4
    hidden_sizes: Tuple[int, ...] = (128, 128)

    # ───────────── PPO‑specific arguments (identical names to PPOL) ─────────
    target_kl: float = 0.04
    vf_coef: float = 0.25
    max_grad_norm: Optional[float] = 0.5
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    norm_adv: bool = True
    recompute_adv: bool = False

    # ─────────────────────── Base policy common arguments ───────────────────
    gamma: float = 0.99
    rew_norm: bool = False
    deterministic_eval: bool = True

    # ─────────────────────── collecting / training params ───────────────────
    episode_per_collect: int = 20
    step_per_epoch: int = 10000
    n_epochs: int = 4
    buffer_size: int = 100000
    worker: str = "ShmemVectorEnv"
    training_num: int = 256
    testing_num: int = 10

    # ───────────────────────────── misc params ──────────────────────────────
    batch_size: int = 256
    save_interval: int = 20
    resume: bool = False
    save_ckpt: bool = True
    verbose: bool = True
    render: bool = False

    # ──────────────────────────── logger params ─────────────────────────────
    logdir: str = "benchmark_results"
    project: str = "fast-safe-rl"
    group: Optional[str] = "changes"
    name: Optional[str] = "PPO"
    prefix: Optional[str] = "ppo"
    suffix: Optional[str] = ""


@dataclass
class DroneLandingCfg(TrainCfg):
    """Overrides tuned for the drone‑landing task."""

    name: Optional[str] = "PPO_1"
    training_num: int = 256
    testing_num: int = 10

    step_per_epoch: int = 1024
    batch_size: int = 4096

    n_epochs: int = 6
    lr: float = 2.5e-5
    target_kl: float = 0.04
    vf_coef: float = 0.6
    hidden_sizes: Tuple[int, ...] = (256, 256)
    resume: bool = False
