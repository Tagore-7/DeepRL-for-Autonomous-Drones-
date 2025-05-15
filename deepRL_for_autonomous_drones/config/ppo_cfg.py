from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DroneEnvCfg:
    algorithm_name: str = "PPO"
    visual_mode: str = "Direct"
    launch_pad_position: tuple = (0, 0, 0)
    boundary_limits: int = 20
    distance_reward_weight: float = 2.0
    leg_contact_reward: int = 100
    tensorboard_log_dir: str = "./logs_metrics_benchmark_tensorboard/"
    model_name_to_save: str = "drone_landing_model_using_ppo"
    gravity: float = -9.8
    discount_factor: float = 0.99
    add_obstacles: bool = True
    enable_wind: bool = False
    debug_axes: bool = False
    enable_curriculum_learning: bool = False
    observation_type: int = 1
    reward_function: int = 1
