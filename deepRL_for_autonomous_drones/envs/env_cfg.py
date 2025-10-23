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
    visual_mode: str = "DIRECT"
    discount_factor: float = 0.99
    reward_function: int = 4
    cost_function: int = 2
    debug_axes: bool = False
    add_obstacles: bool = True
    enable_wind: bool = False
    enable_curriculum_learning: bool = False
    observation_type: int = 2

    use_dyn_landing_pad: bool = True
    enable_dynamic_tree_sizing: bool = False
    use_dyn_trees: bool = True  # For using the seed layout pool

    # ------- Seed / layout_pool -------#
    use_layout_pool: bool = True
    layout_pool_size: int = 64
    eval_pool_size: int = 20

    # ------- Tree configs -------#
    num_trees: int = 50
    forest_span: float = 5.0  # Note: this sets the forest as an NxN (ie, if the value is 5.0, it'll go from -5.0 to 5.0 in both X and Y)
    pad_clear_margin: float = 0.15  # For how close the trees can come to the landing pad
    min_tree_spacing: float = 0.0
    launch_pad_clearance: float = 1.2
    landing_pad_clearance: float = 1.2
