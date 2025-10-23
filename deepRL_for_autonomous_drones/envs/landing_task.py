import numpy as np
import gymnasium as gym
from deepRL_for_autonomous_drones.envs.Drone_Controller_RPM import DroneControllerRPM
from deepRL_for_autonomous_drones.envs.obstacles import generateStaticTrees


class LandingTask(DroneControllerRPM):
    """
    Subtask for landing: agent starts above pad, evaluated only on landing.
    """

    def __init__(
        self,
        render_mode=None,
        graphics=False,
        task_type: str = "Landing",
        spawn_height: float = 1.0,
    ):
        super().__init__(render_mode=render_mode, graphics=graphics, task_type=task_type)
        self.args.task_type = task_type
        # self.enable_wind = False
        self.spawn_height = spawn_height

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        pad_x, pad_y, pad_z = self.landing_pad_position
        start_pos = [pad_x, pad_y, pad_z + self.spawn_height]
        self._p.resetBasePositionAndOrientation(self.drone.getDroneID(), start_pos, [0, 0, 0, 1])
        self._p.resetBaseVelocity(self.drone.getDroneID(), (0, 0, 0), (0, 0, 0))
        self.drone.resetDrone()

        self.drone.action_buffer.clear()
        for _ in range(self.drone.ACTION_BUFFER_SIZE):
            self.drone.action_buffer.append(np.zeros(4, dtype=np.float32))
        self.drone.updateAndStoreKinematicInformation()

        self.after_reset()
        return self._getObservation(), info

    def _computeReward(self, observation, action, reward_function=None):
        state = self.drone.getDroneStateVector()
        pos = state[0:3]
        rel_pos = pos - self.landing_pad_position
        distance_penalty = -1 * np.linalg.norm(rel_pos)

        land_reward = 0.0
        vx, vy, vz = state[10:13]
        contact_points = self._p.getContactPoints(self.drone.getDroneID(), self.landing_pad)
        if contact_points:
            print("Landed")
            if abs(vx) <= 0.3 and abs(vy) <= 0.3 and abs(vz) <= 0.3:
                land_reward = 10.0
                self.task_achieved = True
                self.landed = True
            else:
                land_reward = -10.0
                self.task_achieved = True
                self.hard_landing = True

        shaping = distance_penalty + land_reward

        reward = shaping if self.previous_shaping is None else shaping - self.previous_shaping
        self.previous_shaping = shaping
        return reward

    def _generateStaticTrees(self, layout_seed: int = 42):
        super()._generateStaticTrees(random_scaling=False)
