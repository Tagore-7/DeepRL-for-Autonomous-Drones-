from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainCfg:
    # general task params
    task: str = "SafetyDroneLanding-v0"
    cost_limit: float = 10
    # cost_limit: float = 80
    device: str = "cpu"
    thread: int = 4  # if use "cpu" to train
    seed: int = 10
    # algorithm params
    # lr: float = 5e-4
    lr: float = 3e-4
    hidden_sizes: Tuple[int, ...] = (128, 128)
    unbounded: bool = False
    last_layer_scale: bool = False
    # PPO specific arguments
    # target_kl: float = 0.02
    target_kl: float = 0.04
    vf_coef: float = 0.25
    max_grad_norm: Optional[float] = 0.5
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    dual_clip: Optional[float] = None
    value_clip: bool = False  # no need
    norm_adv: bool = True  # good for improving training stability
    recompute_adv: bool = False
    # Lagrangian specific arguments
    use_lagrangian: bool = True
    lagrangian_pid: Tuple[float, ...] = (0.05, 0.0005, 0.1)
    rescaling: bool = True
    # Base policy common arguments
    gamma: float = 0.99
    max_batchsize: int = 100000
    rew_norm: bool = False  # no need, it will slow down training and decrease final perf
    deterministic_eval: bool = True
    action_scaling: bool = True
    action_bound_method: str = "clip"
    # collecting params
    epoch: int = 200
    episode_per_collect: int = 20
    step_per_epoch: int = 10000
    repeat_per_collect: int = 4  # increasing this can improve efficiency, but less stability
    buffer_size: int = 100000
    worker: str = "ShmemVectorEnv"
    # worker: str = "SubprocVectorEnv"
    training_num: int = 20
    testing_num: int = 2
    # general params
    batch_size: int = 256
    reward_threshold: float = 10000  # for early stop purpose
    save_interval: int = 20
    resume: bool = False  # TODO
    save_ckpt: bool = True  # set this to True to save the policy model
    verbose: bool = True
    render: bool = False
    # logger params
    logdir: str = "benchmark_results"
    project: str = "fast-safe-rl"
    group: Optional[str] = None
    name: Optional[str] = "ppol-no-wind-5"
    prefix: Optional[str] = "ppol"
    suffix: Optional[str] = ""


@dataclass
class DroneLandingCfg(TrainCfg):
    epoch: int = 1500
    # cost_limit = 25
    cost_limit: float = 6
    repeat_per_collect: int = 7
    lr: float = 2.5e-5
    # lr: float = 1e-4
    gae_lambda: float = 0.95
    vf_coef: float = 0.8
    # hidden_sizes: Tuple[int, ...] = (64, 64)
    step_per_epoch: int = 14000
    target_kl: float = 0.06
    lagrangian_pid: Tuple[float, ...] = (0.15, 0.005, 0.1)
