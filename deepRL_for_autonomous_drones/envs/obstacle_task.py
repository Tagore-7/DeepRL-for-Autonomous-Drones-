import numpy as np
import gymnasium as gym
import time
from gymnasium.envs.registration import register
from deepRL_for_autonomous_drones.envs.Drone_Controller_RPM import DroneControllerRPM
from deepRL_for_autonomous_drones.envs.obstacles import generateStaticTrees


class ObstacleTask(DroneControllerRPM):
    """
    Subtask for obstacle avoidance: single tree between spawn and pad; agent must pass around.
    """

    def __init__(
        self,
        render_mode=None,
        graphics=False,
        task_type: str = "Obstacle",
        forest_type: str = "Original",
        spawn_offset_x: float = 7.0,
        spawn_offset_y: float = 0.0,
        tree_spawn_x: float = 0.0,
        tree_spawn_y: float = 0.0,
        spawn_height: float = 1.0,
        obstacle_offset: float = 4.0,
        pass_radius: float = 3.0,
        # --- NEW knobs for randomized obstacle lineups ---#
        n_at_x5: int = 5,
        n_at_x4: int = 5,
        n_at_x3: int = 5,
        y_span: float = 5.0,  # sample Y in [-y_span, +y_span]
        min_gap: float = 1.0,  # min |Δy| between trees at the same X
    ):
        super().__init__(render_mode=render_mode, graphics=graphics, task_type=task_type)
        self.forest_type = forest_type
        self.args.task_type = task_type
        self.add_obstacles = True
        # self.enable_wind = False
        if self.forest_type == "forest_shifted" or self.forest_type == "forest_mixed_shifted":
            self.spawn_offset_x = 6.0
        else:
            self.spawn_offset_x = spawn_offset_x
        # self.spawn_offset_x = spawn_offset_x
        self.spawn_offset_y = spawn_offset_y
        self.tree_spawn_x = tree_spawn_x
        self.tree_spawn_y = tree_spawn_y
        self.spawn_height = spawn_height
        self.obstacle_offset = obstacle_offset
        self.pass_radius = pass_radius
        self.n_at_x5 = int(n_at_x5)
        self.n_at_x4 = int(n_at_x4)
        self.n_at_x3 = int(n_at_x3)
        self.y_span = float(y_span)
        self.min_gap = float(min_gap)

        self.launch_zone_left = True
        self.launch_gate_given = True

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)

        self.landing_pad_position = [0, 0, 0]
        self._p.resetBasePositionAndOrientation(self.landing_pad, self.landing_pad_position, [0, 0, 0, 1])

        pad_x, pad_y, pad_z = self.landing_pad_position
        # spawn left of pad
        start_pos = [pad_x + self.spawn_offset_x, pad_y + self.spawn_offset_y, pad_z + self.spawn_height]
        self._p.resetBasePositionAndOrientation(self.drone.getDroneID(), start_pos, [0, 0, 0, 1])
        self._p.resetBaseVelocity(self.drone.getDroneID(), (0, 0, 0), (0, 0, 0))
        self.drone.resetDrone()

        self.drone.action_buffer.clear()
        for _ in range(self.drone.ACTION_BUFFER_SIZE):
            self.drone.action_buffer.append(np.zeros(4, dtype=np.float32))
        self.drone.updateAndStoreKinematicInformation()

        # ---- Regenerate trees ----#
        self._generateStaticTrees()

        self.after_reset()

        return self._getObservation(), info

    def _computeTerminated(self):
        if self.at_reset:
            return

        pos = self.drone.getDroneStateVector()[0:3]
        # tree_pos, _ = self._p.getBasePositionAndOrientation(self.trees[0])
        # tree_x = tree_pos[0]
        tree_x = 4.0

        if abs(pos[0]) <= abs(tree_x) - abs(self.pass_radius):
            # if abs(pos[0]) <= 1:
            self.task_achieved = True
            # print("Task achieved")
            return True
        return False
        # return super()._computeTerminated()

    def _computeReward(self, observation, action, reward_function, tilt_cost=0, spin_cost=0, lidar_cost=0):
        self.launch_zone_left = True
        self.launch_gate_given = True

        pos = self.drone.getDroneStateVector()[0:3]
        # tree_pos, _ = self._p.getBasePositionAndOrientation(self.trees[0])
        # tree_x = tree_pos[0]
        tree_x = 4.0

        reward = 0.0
        if abs(pos[0]) <= abs(tree_x) - abs(self.pass_radius):
            reward = 1.0
        return reward

    def _generateStaticTrees(self, layout_seed: int = 42):
        # pad_x, pad_y, _ = self.landing_pad_position
        pad_x, pad_y, _ = [0, 0, 0]

        # TODO: Fix this rushed mess...
        if self.forest_type == "1_reg_square_tree":
            ######################## ONE TREE, REGULAR SQUARE #############################
            self.fixed_tree_positions = [(pad_x + 4, pad_y, 0)]
            self.fixed_tree_types = ["assets/trees/tree_tall_slim.urdf"]
        elif self.forest_type == "3_reg_square_trees":
            ######################## THREE TREES, REGULAR SQUARE #############################
            self.fixed_tree_positions = [
                (pad_x + 4, pad_y, 0),
                (pad_x + 4, pad_y + 1.0, 0),
                (pad_x + 4, pad_y - 1.0, 0),
            ]
            self.fixed_tree_types = [
                "assets/trees/tree_tall_slim.urdf",
                "assets/trees/tree_tall_slim.urdf",
                "assets/trees/tree_tall_slim.urdf",
            ]
        elif self.forest_type == "forest" or self.forest_type == "forest_shifted":
            ######################## FOREST OF TREES, SQUARE #############################
            if self.forest_type == "forest_shifted":
                self.fixed_tree_positions = [
                    # Col 4
                    (pad_x + 3, pad_y, 0),
                    (pad_x + 3, pad_y + 1.0, 0),
                    (pad_x + 3, pad_y - 1.0, 0),
                    # Col 3
                    (pad_x + 2, pad_y + 2.0, 0),
                    (pad_x + 2, pad_y - 2.0, 0),
                    # Col 5
                    (pad_x + 4, pad_y + 2.0, 0),
                    (pad_x + 4, pad_y - 2.0, 0),
                    (pad_x + 4, pad_y + 1, 0),
                    (pad_x + 4, pad_y - 1, 0),
                    # Col 6
                    (pad_x + 5, pad_y + 2.0, 0),
                    (pad_x + 5, pad_y - 2.0, 0),
                    (pad_x + 5, pad_y + 0.5, 0),
                    (pad_x + 5, pad_y - 0.5, 0),
                ]
            else:
                self.fixed_tree_positions = [
                    # Col 4
                    (pad_x + 4, pad_y, 0),
                    (pad_x + 4, pad_y + 1.0, 0),
                    (pad_x + 4, pad_y - 1.0, 0),
                    # Col 3
                    (pad_x + 3, pad_y + 2.0, 0),
                    (pad_x + 3, pad_y - 2.0, 0),
                    # Col 5
                    (pad_x + 5, pad_y + 2.0, 0),
                    (pad_x + 5, pad_y - 2.0, 0),
                    (pad_x + 5, pad_y + 1, 0),
                    (pad_x + 5, pad_y - 1, 0),
                    # Col 6
                    (pad_x + 6, pad_y + 2.0, 0),
                    (pad_x + 6, pad_y - 2.0, 0),
                    (pad_x + 6, pad_y + 0.5, 0),
                    (pad_x + 6, pad_y - 0.5, 0),
                ]

            self.fixed_tree_types = [
                # Col 4
                "assets/trees/tree_tall_wide.urdf",
                "assets/trees/tree_mid_mid.urdf",
                "assets/trees/tree_mid_slim.urdf",
                # Col 3
                "assets/trees/tree_tall_wide.urdf",
                "assets/trees/tree_tall_slim.urdf",
                # Col 5
                "assets/trees/tree_tall_wide.urdf",
                "assets/trees/tree_tall_slim.urdf",
                "assets/trees/tree_mid_slim.urdf",
                "assets/trees/tree_mid_wide.urdf",
                "assets/trees/tree_tall_wide.urdf",
                "assets/trees/tree_tall_slim.urdf",
                "assets/trees/tree_mid_slim.urdf",
                "assets/trees/tree_mid_wide.urdf",
            ]
        elif self.forest_type == "forest_mixed" or self.forest_type == "forest_mixed_shifted":
            ######################## FOREST OF TREES, MIXED #############################
            if self.forest_type == "forest_mixed_shifted":
                self.fixed_tree_positions = [
                    # Col 4
                    (pad_x + 3, pad_y, 0),
                    (pad_x + 3, pad_y + 1.0, 0),
                    (pad_x + 3, pad_y - 1.0, 0),
                    # Col 3
                    (pad_x + 2, pad_y + 2.0, 0),
                    (pad_x + 2, pad_y - 2.0, 0),
                    # Col 5
                    (pad_x + 4, pad_y + 2.0, 0),
                    (pad_x + 4, pad_y - 2.0, 0),
                    (pad_x + 4, pad_y + 1, 0),
                    (pad_x + 4, pad_y - 1, 0),
                    # Col 6
                    (pad_x + 5, pad_y + 2.0, 0),
                    (pad_x + 5, pad_y - 2.0, 0),
                    (pad_x + 5, pad_y + 0.5, 0),
                    (pad_x + 5, pad_y - 0.5, 0),
                ]
            else:
                self.fixed_tree_positions = [
                    # Col 4
                    (pad_x + 4, pad_y, 0),
                    (pad_x + 4, pad_y + 1.0, 0),
                    (pad_x + 4, pad_y - 1.0, 0),
                    # Col 3
                    (pad_x + 3, pad_y + 2.0, 0),
                    (pad_x + 3, pad_y - 2.0, 0),
                    # Col 5
                    (pad_x + 5, pad_y + 2.0, 0),
                    (pad_x + 5, pad_y - 2.0, 0),
                    (pad_x + 5, pad_y + 1, 0),
                    (pad_x + 5, pad_y - 1, 0),
                    # Col 6
                    (pad_x + 6, pad_y + 2.0, 0),
                    (pad_x + 6, pad_y - 2.0, 0),
                    (pad_x + 6, pad_y + 0.5, 0),
                    (pad_x + 6, pad_y - 0.5, 0),
                ]

            self.fixed_tree_types = [
                # Col 4
                "assets/trees/tree_cylinder_wide.urdf",
                "assets/trees/tree_mid_mid.urdf",
                "assets/trees/tree_mid_slim.urdf",
                # Col 3
                "assets/trees/tree_tall_wide.urdf",
                "assets/trees/tree_cylinder_slim.urdf",
                # Col 5
                "assets/trees/tree_tall_wide.urdf",
                "assets/trees/tree_tall_slim.urdf",
                "assets/trees/tree_cylinder_slim.urdf",
                "assets/trees/tree_mid_wide.urdf",
                "assets/trees/tree_tall_wide.urdf",
                "assets/trees/tree_cylinder_slim.urdf",
                "assets/trees/tree_mid_slim.urdf",
                "assets/trees/tree_mid_wide.urdf",
            ]
        else:
            self.fixed_tree_positions = [(pad_x + 4, pad_y, 0)]
            self.fixed_tree_types = ["assets/trees/tree_tall_slim.urdf"]

        ######################################################################################
        ######################################################################################
        ######################################################################################

        ######################## FOREST OF NORMAL TREES, SQUARE #############################
        # self.fixed_tree_positions = [
        #     # Col 4
        #     (pad_x + 4, pad_y, 0),
        #     (pad_x + 4, pad_y + 1.0, 0),
        #     (pad_x + 4, pad_y - 1.0, 0),
        #     # (pad_x + 3.5, pad_y, 0),
        #     # (pad_x + 3.5, pad_y + 1.0, 0),
        #     # (pad_x + 3.5, pad_y - 1.0, 0),
        #     # Col 3
        #     (pad_x + 3, pad_y + 2.0, 0),
        #     (pad_x + 3, pad_y - 2.0, 0),
        #     # (pad_x + 3, pad_y + 0.5, 0),
        #     # (pad_x + 3, pad_y - 0.5, 0),
        #     # Col 5
        #     (pad_x + 5, pad_y + 2.0, 0),
        #     (pad_x + 5, pad_y - 2.0, 0),
        #     (pad_x + 5, pad_y + 1, 0),
        #     (pad_x + 5, pad_y - 1, 0),
        #     # Col 6
        #     (pad_x + 6, pad_y + 2.0, 0),
        #     (pad_x + 6, pad_y - 2.0, 0),
        #     (pad_x + 6, pad_y + 0.5, 0),
        #     (pad_x + 6, pad_y - 0.5, 0),
        # ]

        # self.fixed_tree_types = [
        #     # Col 4
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     # Col 3
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     # Col 5
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        # ]

        ######################## ONE TREE, REGULAR SQUARE #############################
        # self.fixed_tree_positions = [(pad_x + 4, pad_y, 0)]
        # self.fixed_tree_types = ["assets/trees/tree_tall_slim.urdf"]

        ######################## ONE TREE, WIDE SQUARE #############################
        # self.fixed_tree_positions = [(pad_x + 4, pad_y, 0)]
        # self.fixed_tree_types = ["assets/trees/tree_tall_wide.urdf"]

        ######################## ONE TREE, REGULAR CIRCLE #############################
        # self.fixed_tree_positions = [(pad_x + 4, pad_y, 0)]
        # self.fixed_tree_types = ["assets/trees/tree_cylinder.urdf"]

        ######################## ONE TREE, WIDE CIRCLE #############################
        # self.fixed_tree_positions = [(pad_x + 4, pad_y, 0)]
        # self.fixed_tree_types = ["assets/trees/tree_cylinder_wide.urdf"]

        ######################## THREE TREES, REGULAR SQUARE #############################
        # self.fixed_tree_positions = [
        #     (pad_x + 4, pad_y, 0),
        #     (pad_x + 4, pad_y + 1.0, 0),
        #     (pad_x + 4, pad_y - 1.0, 0),
        # ]
        # self.fixed_tree_types = [
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        # ]

        ######################## THREE TREES, WIDE SQUARE #############################
        # self.fixed_tree_positions = [
        #     (pad_x + 4, pad_y, 0),
        #     (pad_x + 4, pad_y + 1.0, 0),
        #     (pad_x + 4, pad_y - 1.0, 0),
        # ]
        # self.fixed_tree_types = [
        #     "assets/trees/tree_tall_wide.urdf",
        #     "assets/trees/tree_tall_wide.urdf",
        #     "assets/trees/tree_tall_wide.urdf",
        # ]

        ######################## THREE TREES, MIX-MATCH SQUARE #############################
        # self.fixed_tree_positions = [
        #     (pad_x + 4, pad_y, 0),
        #     (pad_x + 4, pad_y + 1.0, 0),
        #     (pad_x + 4, pad_y - 1.0, 0),
        # ]
        # self.fixed_tree_types = [
        #     "assets/trees/tree_mid_slim.urdf",
        #     "assets/trees/tree_mid_mid.urdf",
        #     "assets/trees/tree_tall_wide.urdf",
        # ]

        ######################## THREE TREES, REGULAR CIRCLE #############################
        # self.fixed_tree_positions = [
        #     (pad_x + 4, pad_y, 0),
        #     (pad_x + 4, pad_y + 1.0, 0),
        #     (pad_x + 4, pad_y - 1.0, 0),
        # ]
        # self.fixed_tree_types = [
        #     "assets/trees/tree_cylinder.urdf",
        #     "assets/trees/tree_cylinder.urdf",
        #     "assets/trees/tree_cylinder.urdf",
        # ]

        ######################## THREE TREES, WIDE CIRCLE #############################
        # self.fixed_tree_positions = [
        #     (pad_x + 4, pad_y, 0),
        #     (pad_x + 4, pad_y + 1.0, 0),
        #     (pad_x + 4, pad_y - 1.0, 0),
        # ]
        # self.fixed_tree_types = [
        #     "assets/trees/tree_cylinder_wide.urdf",
        #     "assets/trees/tree_cylinder_wide.urdf",
        #     "assets/trees/tree_cylinder_wide.urdf",
        # ]

        ######################## THREE TREES, MIX-MATCH CIRCLE #############################
        # self.fixed_tree_positions = [
        #     (pad_x + 4, pad_y, 0),
        #     (pad_x + 4, pad_y + 1.0, 0),
        #     (pad_x + 4, pad_y - 1.0, 0),
        # ]
        # self.fixed_tree_types = [
        #     "assets/trees/tree_cylinder_slim.urdf",
        #     "assets/trees/tree_cylinder_wide.urdf",
        #     "assets/trees/tree_cylinder.urdf",
        # ]

        ######################## FOREST OF TREES, SQUARE #############################
        # self.fixed_tree_positions = [
        #     # Col 4
        #     (pad_x + 4, pad_y, 0),
        #     (pad_x + 4, pad_y + 1.0, 0),
        #     (pad_x + 4, pad_y - 1.0, 0),
        #     # (pad_x + 3.5, pad_y, 0),
        #     # (pad_x + 3.5, pad_y + 1.0, 0),
        #     # (pad_x + 3.5, pad_y - 1.0, 0),
        #     # Col 3
        #     (pad_x + 3, pad_y + 2.0, 0),
        #     (pad_x + 3, pad_y - 2.0, 0),
        #     # (pad_x + 3, pad_y + 0.5, 0),
        #     # (pad_x + 3, pad_y - 0.5, 0),
        #     # Col 5
        #     (pad_x + 5, pad_y + 2.0, 0),
        #     (pad_x + 5, pad_y - 2.0, 0),
        #     (pad_x + 5, pad_y + 1, 0),
        #     (pad_x + 5, pad_y - 1, 0),
        #     # Col 6
        #     (pad_x + 6, pad_y + 2.0, 0),
        #     (pad_x + 6, pad_y - 2.0, 0),
        #     (pad_x + 6, pad_y + 0.5, 0),
        #     (pad_x + 6, pad_y - 0.5, 0),
        # ]

        # self.fixed_tree_types = [
        #     # Col 4
        #     "assets/trees/tree_tall_wide.urdf",
        #     "assets/trees/tree_mid_mid.urdf",
        #     "assets/trees/tree_mid_slim.urdf",
        #     # Col 3
        #     "assets/trees/tree_tall_wide.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     # "assets/trees/tree_mid_slim.urdf",
        #     # "assets/trees/tree_mid_wide.urdf",
        #     # Col 5
        #     "assets/trees/tree_tall_wide.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_mid_slim.urdf",
        #     "assets/trees/tree_mid_wide.urdf",
        #     "assets/trees/tree_tall_wide.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_mid_slim.urdf",
        #     "assets/trees/tree_mid_wide.urdf",
        # ]

        ######################## FOREST OF NORMAL TREES, SQUARE #############################
        # self.fixed_tree_positions = [
        #     # Col 4
        #     (pad_x + 4, pad_y, 0),
        #     (pad_x + 4, pad_y + 1.0, 0),
        #     (pad_x + 4, pad_y - 1.0, 0),
        #     # (pad_x + 3.5, pad_y, 0),
        #     # (pad_x + 3.5, pad_y + 1.0, 0),
        #     # (pad_x + 3.5, pad_y - 1.0, 0),
        #     # Col 3
        #     (pad_x + 3, pad_y + 2.0, 0),
        #     (pad_x + 3, pad_y - 2.0, 0),
        #     # (pad_x + 3, pad_y + 0.5, 0),
        #     # (pad_x + 3, pad_y - 0.5, 0),
        #     # Col 5
        #     (pad_x + 5, pad_y + 2.0, 0),
        #     (pad_x + 5, pad_y - 2.0, 0),
        #     (pad_x + 5, pad_y + 1, 0),
        #     (pad_x + 5, pad_y - 1, 0),
        #     # Col 6
        #     (pad_x + 6, pad_y + 2.0, 0),
        #     (pad_x + 6, pad_y - 2.0, 0),
        #     (pad_x + 6, pad_y + 0.5, 0),
        #     (pad_x + 6, pad_y - 0.5, 0),
        # ]

        # self.fixed_tree_types = [
        #     # Col 4
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     # Col 3
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     # Col 5
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        # ]

        # self.fixed_tree_positions = [
        #     (pad_x + 4, pad_y, 0),
        #     (pad_x + 4, pad_y + 0.75, 0),
        #     (pad_x + 4, pad_y - 0.75, 0),
        #     # (pad_x + 4, pad_y + 1.50, 0),
        #     # (pad_x + 4, pad_y - 1.50, 0),
        # ]

        # # self.fixed_tree_types = [
        # #     "assets/trees/tree_cylinder.urdf",
        # #     # "assets/trees/tree_dynamic.urdf",
        # #     # "assets/tree_five.urdf",
        # #     # "assets/trees/tree_tall_wide.urdf",
        # # ]

        # # Cylinder - 3 trees
        # self.fixed_tree_types = [
        #     # "assets/trees/tree_cylinder_slim.urdf",
        #     # "assets/trees/tree_cylinder_wide.urdf",
        #     # "assets/trees/tree_cylinder.urdf",
        #     "assets/trees/tree_tall_slim.urdf",
        #     "assets/trees/tree_tall_wide.urdf",
        #     # "assets/trees/tree_cylinder_wide.urdf",
        #     "assets/trees/tree_mid_mid.urdf",
        #     # "assets/trees/tree_cylinder_slim.urdf",
        # ]

        ######################## THREE TREES, MIX-MATCH SQUARE close #############################
        # self.fixed_tree_positions = [
        #     (pad_x + 4, pad_y, 0),
        #     (pad_x + 4, pad_y + 1.0, 0),
        #     (pad_x + 4, pad_y - 1.0, 0),
        # ]
        # self.fixed_tree_types = [
        #     "assets/trees/tree_mid_slim.urdf",
        #     "assets/trees/tree_mid_mid.urdf",
        #     "assets/trees/tree_tall_wide.urdf",
        # ]
        ######################## THREE TREES, MIX-MATCH SQUARE closer #############################
        # self.fixed_tree_positions = [
        #     (pad_x + 4, pad_y, 0),
        #     (pad_x + 4, pad_y + 1.0, 0),
        #     (pad_x + 4, pad_y - 1.0, 0),
        # ]
        # self.fixed_tree_types = [
        #     "assets/trees/tree_mid_slim.urdf",
        #     "assets/trees/tree_mid_mid.urdf",
        #     "assets/trees/tree_tall_wide.urdf",
        # ]

        ######################## THREE TREES, MIX-MATCH CIRCLE CLOSE #############################
        # self.fixed_tree_positions = [
        #     (pad_x + 4, pad_y, 0),
        #     (pad_x + 4, pad_y + 0.75, 0),
        #     (pad_x + 4, pad_y - 0.75, 0),
        # ]
        # self.fixed_tree_types = [
        #     "assets/trees/tree_cylinder_slim.urdf",
        #     "assets/trees/tree_cylinder_wide.urdf",
        #     "assets/trees/tree_cylinder.urdf",
        # ]

        ######################## THREE TREES, MIX-MATCH CIRCLE CLOSER #############################
        # self.fixed_tree_positions = [
        #     (pad_x + 4, pad_y, 0),
        #     (pad_x + 4, pad_y + 0.5, 0),
        #     (pad_x + 4, pad_y - 0.5, 0),
        # ]
        # self.fixed_tree_types = [
        #     "assets/trees/tree_cylinder_slim.urdf",
        #     "assets/trees/tree_cylinder_wide.urdf",
        #     "assets/trees/tree_cylinder.urdf",
        # ]

        self.trees = generateStaticTrees(self.fixed_tree_positions, self.fixed_tree_types, self._p, random_scaling=False)
