import numpy as np
import gymnasium as gym
from deepRL_for_autonomous_drones.envs.Drone_Controller_RPM import DroneControllerRPM


class NavigationTask(DroneControllerRPM):
    """
    Subtask for point-to-point navigation: agent must reach within `nav_radius` of the pad.
    """

    def __init__(
        self,
        render_mode=None,
        graphics=False,
        task_type: str = "Navigation",
        # spawn_offset: float = -6.7,
        spawn_offset: float = 7.5,
        spawn_height: float = 0.2,
        nav_radius: float = 5.2,
    ):
        super().__init__(render_mode=render_mode, graphics=graphics, task_type=task_type)
        self.args.task_type = task_type
        self.add_obstacles = True
        # self.enable_wind = False
        self.nav_radius = nav_radius
        self.spawn_offset = spawn_offset
        self.spawn_height = spawn_height

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)

        # self.wall_id = self._add_axis_wall(axis="y", coord=self.nav_radius, span=20.0, height=2.0, thickness=0.05)

        # ---- Launch pad reset ----#
        self.launch_pad_position = [self.spawn_offset, 0, 0]
        self._p.resetBasePositionAndOrientation(self.launch_pad, self.launch_pad_position, [0, 0, 0, 1])

        # ---- Drone reset ----#
        start_pos = [self.launch_pad_position[0], self.launch_pad_position[1], self.launch_pad_position[2] + self.spawn_height]
        self._p.resetBasePositionAndOrientation(self.drone.getDroneID(), start_pos, [0, 0, 0, 1])
        self._p.resetBaseVelocity(self.drone.getDroneID(), (0, 0, 0), (0, 0, 0))
        self.drone.resetDrone()

        self.drone.action_buffer.clear()
        for _ in range(self.drone.ACTION_BUFFER_SIZE):
            self.drone.action_buffer.append(np.zeros(4, dtype=np.float32))
        self.drone.updateAndStoreKinematicInformation()

        self.after_reset()
        return self._getObservation(), info

    def _computeTerminated(self):
        pos = self.drone.getDroneStateVector()[0:3]
        if np.linalg.norm(pos) <= self.nav_radius:
            self.task_achieved = True
            print("Task achieved")
            return True
        # return False
        return super()._computeTerminated()

    def _generateStaticTrees(self, layout_seed: int = 42):
        super()._generateStaticTrees(random_scaling=False)

    def _add_axis_wall(
        self,
        axis: str = "x",  # "x" -> wall stretches along X (fixed y); "y" -> along Y (fixed x)
        coord: float = -8.0,  # y or x coordinate where the wall sits
        span: float = 20.0,  # total length of the wall along its axis
        height: float = 2.0,
        thickness: float = 0.05,
        rgba=(1.0, 0.0, 0.0, 0.35),
    ):
        """
        Adds a visible, non-colliding wall. For axis="x", the wall lies at y=coord
        and stretches along X; for axis="y", it lies at x=coord and stretches along Y.
        """
        assert axis in ("x", "y"), "axis must be 'x' or 'y'"

        # Half-extents for the visual box (PyBullet boxes take half extents)
        if axis == "x":
            hx, hy, hz = span * 0.5, thickness * 0.5, height * 0.5
            base_pos = [0.0, float(coord), hz]  # center at z = height/2 so it sits on the ground
        else:
            hx, hy, hz = thickness * 0.5, span * 0.5, height * 0.5
            base_pos = [float(coord), 0.0, hz]

        vis = self._p.createVisualShape(
            shapeType=self._p.GEOM_BOX,
            halfExtents=[hx, hy, hz],
            rgbaColor=list(rgba),
        )

        wall_id = self._p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,  # no collision
            baseVisualShapeIndex=vis,
            basePosition=base_pos,
            baseOrientation=[0, 0, 0, 1],
        )

        # zero out any collision filtering just in case
        try:
            self._p.setCollisionFilterGroupMask(wall_id, -1, collisionFilterGroup=0, collisionFilterMask=0)
        except Exception:
            pass

        return wall_id
