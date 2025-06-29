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
    # CVPO arguments
    estep_iter_num: int = 1
    estep_kl: float = 0.02
    estep_dual_max: float = 20
    estep_dual_lr: float = 0.02
    sample_act_num: int = 16
    mstep_iter_num: int = 1
    mstep_kl_mu: float = 0.005
    mstep_kl_std: float = 0.0005
    mstep_dual_max: float = 0.5
    mstep_dual_lr: float = 0.1
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    gamma: float = 0.97
    n_step: int = 2
    tau: float = 0.05
    hidden_sizes: Tuple[int, ...] = (128, 128)
    double_critic: bool = False
    conditioned_sigma: bool = True
    unbounded: bool = False
    last_layer_scale: bool = False
    # collecting params
    epoch: int = 200
    episode_per_collect: int = 10
    step_per_epoch: int = 10000
    update_per_step: float = 0.2
    buffer_size: int = 200000
    worker: str = "ShmemVectorEnv"
    # worker: str = "SubprocVectorEnv"
    training_num: int = 20
    testing_num: int = 2
    # general train params
    batch_size: int = 256
    reward_threshold: float = 10000  # for early stop purpose
    save_interval: int = 50
    deterministic_eval: bool = True
    action_scaling: bool = True
    action_bound_method: str = "clip"
    resume: bool = False  # TODO
    save_ckpt: bool = True  # set this to True to save the policy model
    verbose: bool = True
    render: bool = False
    # logger params
    logdir: str = "benchmark_results"
    project: str = "fast-safe-rl"
    group: Optional[str] = None
    name: Optional[str] = None
    prefix: Optional[str] = "cvpo"
    suffix: Optional[str] = ""


@dataclass
class DroneLandingCfg(TrainCfg):
    epoch: int = 500
    cost_limit = 25
    # cost_limit: float = 80
