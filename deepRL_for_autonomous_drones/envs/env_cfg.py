from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class EnvCfg:
    launch_pad_position: Tuple[int, ...] = (0, 0, 0)
    distance_reward_weight: float = 2.0
    tensorboard_log_dir: str = "./logs_metrics_benchmark_tensorboard/"
    boundary_limits: int = 20
    gravity: float = -9.8
    leg_contact_reward: int = 100
    model_name_to_save: str = "drone_landing_model_using_ppo"
    visual_mode: str = "Direct"
    discount_factor: float = 0.99
    reward_function: int = 1
    debug_axes: bool = False
    add_obstacles: bool = True
    enable_wind: bool = True
    enable_curriculum_learning: bool = False
    observation_type: int = 2
