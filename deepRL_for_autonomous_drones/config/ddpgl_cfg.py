from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainCfg:
    # general task params
    task: str = "SafetyDroneLanding-v0"
    cost_limit: float = 10
    device: str = "cpu"
    thread: int = 4  # if use "cpu" to train
    seed: int = 10
    # algorithm params
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    hidden_sizes: Tuple[int, ...] = (128, 128)
    tau: float = 0.05
    exploration_noise: float = 0.1
    n_step: int = 2
    # Lagrangian specific arguments
    use_lagrangian: bool = True
    lagrangian_pid: Tuple[float, ...] = (0.05, 0.0005, 0.1)
    rescaling: bool = True
    # Base policy common arguments
    gamma: float = 0.97
    deterministic_eval: bool = True
    action_scaling: bool = True
    action_bound_method: str = "clip"
    # collecting params
    epoch: int = 200
    episode_per_collect: int = 2
    step_per_epoch: int = 10000
    update_per_step: float = 0.2
    buffer_size: int = 100000
    worker: str = "ShmemVectorEnv"
    training_num: int = 10
    testing_num: int = 2
    # general params
    batch_size: int = 256
    reward_threshold: float = 10000  # for early stop purpose
    save_interval: int = 4
    resume: bool = False  # TODO
    save_ckpt: bool = True  # set this to True to save the policy model
    verbose: bool = True
    render: bool = False
    # logger params
    logdir: str = "benchmark_results"
    project: str = "fast-safe-rl"
    group: Optional[str] = None
    name: Optional[str] = None
    prefix: Optional[str] = "ddpgl"
    suffix: Optional[str] = ""


@dataclass
class DroneLandingCfg(TrainCfg):
    epoch: int = 500
    cost_limit = 25
    # cost_limit: float = 80
