# import math
# import random
from collections import deque

# import pkg_resources
from importlib.resources import files
import numpy as np
import pybullet as p
import random
from deepRL_for_autonomous_drones.utils.drone_urdf_parser import parseURDFParameters
from deepRL_for_autonomous_drones.config.robust_settings_cfg import RobustCfg


class Drone(object):
    def __init__(
        self,
        bullet_client,
        rng=None,
        launch_pad_position=None,
        gravity=-9.81,
        ctrl_freq=30,
        # pyb_client=p,
        normalized_rl_action_space=True,
    ):

        self.args = RobustCfg

        self.delay_buf: deque[np.ndarray] = deque(maxlen=self.args.delay_steps or 1)
        self._fail_idx: int | None = None  # which rotor (0–3) will fail
        self._fail_remaining: int = 0  # how many more control steps to keep it disabled
        self._fail_timer: int = 0  # countdown before failure starts
        self._flip_set: set[int] = set()  # which rotors are inverted
        self._base_yaw_dir = np.array([-1, 1, -1, 1], dtype=float)
        self._yaw_dir = self._base_yaw_dir.copy()

        self._p = bullet_client
        self.rng = rng
        self.launch_pad_position = launch_pad_position
        self.gravity = gravity
        self.CTRL_FREQ = ctrl_freq
        # self.pyb_client = pyb_client
        self.NORMALIZED_RL_ACTION_SPACE = normalized_rl_action_space

        # ---- Load drone properties from the .urdf file ---- #
        (
            self.MASS,  # (mass of drone in kilograms)
            self.ARM,  # ("Arm length" or distance from center to rotor)
            self.THRUST2WEIGHT_RATIO,  # (Ratio of maximum total thrust over the drone's weight)
            self.J,  # [IXX, IYY, IZZ] (Inertia matrix)
            self.J_INV,  # (Inverse inertia matrix)
            self.KF,  # (thrust coefficient - how rotor speed squared translates into thrust force.)
            self.KM,  # (Torque (moment) coefficient - how rotor speed squared translates into rotor torque.)
            self.COLLISION_H,
            self.COLLISION_R,
            self.COLLISION_Z_OFFSET,
            self.MAX_SPEED_KMH,
            self.GND_EFF_COEFF,  # (ground effect coefficient)
            self.PROP_RADIUS,  # (The physical radius of the propellers)
            self.DRAG_COEFF,  # [DRAG_COEFF_XY, DRAG_COEFF, XY, DRAG_COEFF_Z]
            self.DW_COEFF_1,
            self.DW_COEFF_2,
            self.DW_COEFF_3,
        ) = parseURDFParameters("assets/cf2x.urdf")
        # ) = parseURDFParameters("assets/cf21x_bullet.urdf")

        # ---- Compute constants ----#
        self.G = -self.gravity * self.MASS
        self.HOVER_RPM = np.sqrt(self.G / (4 * self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.G) / (4 * self.KF))
        self.MAX_THRUST = 4 * self.KF * self.MAX_RPM**2
        self.MAX_XY_TORQUE = (2 * self.ARM * self.KF * self.MAX_RPM**2) / np.sqrt(2)
        self.MAX_Z_TORQUE = 2 * self.KM * self.MAX_RPM**2
        self.HOVER_THRUST = 9.81 * self.MASS / 4  # Gravity compensation per motor
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000.0
        self.MAX_PWM = 65535.0

        # ---- Rotor positions in URDF file ---- #
        self.rotor_positions_local = np.array(
            [
                [0.028, -0.028, 0.0],  # prop0 - Back right rotor: clockwise
                [-0.028, -0.028, 0.0],  # prop1 - Back left rotor: counterclockwise
                [-0.028, 0.028, 0.0],  # prop2 - Front left rotor: clockwise
                [0.028, 0.028, 0.0],  # prop3 - Front right rotor: counterclockwise
            ]
        )

        # ---- Action sent by controller, possibly normalized and unclipped ----#
        self.current_raw_action = None
        # ---- Current_raw_action unnormalized if it was normalized ----#
        self.current_physical_action = None
        # ---- Current_noisy_physical_action clipped to physical action bounds ----#
        self.current_clipped_action = None

        self.norm_act_scale = 0.1

        # ---- Create a buffer for the last .5 sec of actions ----#
        self.ACTION_BUFFER_SIZE = int(self.CTRL_FREQ // 2)

        # ---- Stores the most recent actions performed          ----#
        # ---- which is added into the Observation space and     ----#
        # ---- helps the RL agent learn by giving it information ----#
        # ---- about what was recently commanded, enabling it to ----#
        # ---- learn better stabilization and complex actions    ----#
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        self.last_action_err = 0.0
        self.pos = np.zeros(3)
        self.quat = np.zeros(4)
        self.rpy = np.zeros(3)
        self.vel = np.zeros(3)
        self.ang_v = np.zeros(3)
        self.last_clipped_action = np.zeros(4)

        act_lower_bound = self.KF * 1 * (self.PWM2RPM_SCALE * self.MIN_PWM + self.PWM2RPM_CONST) ** 2
        act_upper_bound = self.KF * 1 * (self.PWM2RPM_SCALE * self.MAX_PWM + self.PWM2RPM_CONST) ** 2
        self.physical_action_bounds = (
            np.full(4, act_lower_bound, np.float32),
            np.full(4, act_upper_bound, np.float32),
        )

        self.resetDrone()
        self.loadDrone()

    def armRotorFailure(self, idx: int, steps: int, delay_sec: float, ctrl_freq: int):
        """
        Schedule a rotor failure:
            idx       - rotor index 0..3 that will be zero-thrust
            steps     - how many control steps to keep it disabled
            delay_sec - how many seconds after reset before it starts
            ctrl_freq - controller frequency (e.g. 30, 60) to convert seconds→steps
        Called once at the beginning of every episode.
        """
        self._fail_idx = idx
        self._fail_remaining = 0  # nothing yet
        self._fail_timer = int(delay_sec * ctrl_freq)
        self._fail_steps_cfg = steps  # remember for when timer hits 0

    def resetRotorFailure(self):
        self._fail_remaining = 0
        self._fail_idx = None
        self._fail_timer = 0

    def setSignFlips(self, idxs: set[int]):
        self._yaw_dir = self._base_yaw_dir.copy()
        for i in idxs:
            self._yaw_dir[i] *= -1  # flip spin direction
        self._flip_set = idxs.copy()

    def applyActionDisruptor(self, thrust):

        if not self.args.enabled:
            return thrust

        if self.args.delay_steps > 0:
            if len(self.delay_buf) < self.args.delay_steps:
                self.delay_buf.append(thrust.copy())
                delayed = thrust  # no old cmd yet
            else:
                delayed = self.delay_buf.popleft()  # oldest cmd
                self.delay_buf.append(thrust.copy())
            thrust = delayed

        if self._fail_timer > 0:
            # Failure not active yet
            self._fail_timer -= 1
        elif self._fail_remaining == 0 and self._fail_idx is not None:
            # Time elapse, start failure window
            self._fail_remaining = self._fail_steps_cfg

        if self._fail_remaining > 0 and self._fail_idx is not None:
            thrust = thrust.copy()
            thrust[self._fail_idx] = 0.0
            self._fail_remaining -= 1

        t = thrust.copy()
        if self.args.noise_sigma > 0:
            mode = self.args.noise_type
            mu = self.args.noise_mu
            sigma = self.args.noise_sigma * self.HOVER_THRUST

            if mode == "gauss":
                t += self.rng.normal(mu, sigma, size=t.shape)

        return t

    def set_bullet_client(self, bullet_client):
        self._p = bullet_client

    def getDroneID(self):
        """Gets the drone's ID"""
        return self.drone

    def get_position(self):
        return self.pos

    def get_orientation(self):
        return self.rpy

    def get_quaternion(self):
        return self.quat

    def get_linear_velocity(self):
        return self.vel

    def setCurrentRawAction(self, action):
        """Sets the current raw action"""
        self.current_raw_action = action

    def setCurrentPhysicalAction(self, action):
        """Sets the current physical action"""
        self.current_physical_action = action

    def setCurrentClippedAction(self, action):
        """Sets the current clipped action"""
        self.current_clipped_action = action

    def cmd2pwm(self, thrust):
        """
        Generic cmd to pwm function.
        Thrust is thrust of each motor.
        """
        n_motor = 4 // int(thrust.size)
        thrust = np.clip(thrust, np.zeros_like(thrust), None)  # Make sure thrust is not negative.
        motor_pwm = (np.sqrt(thrust / n_motor / self.KF) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        motor_pwm = np.array(motor_pwm)
        motor_pwm = np.clip(motor_pwm, self.MIN_PWM, self.MAX_PWM)
        return motor_pwm

    def pwm2rpm(self, pwm):
        """Computes motor squared rpm from pwm"""
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        return rpm

    def normalizeAction(self, action):
        """Converts a physical action into a normalized action if necessary"""
        if self.NORMALIZED_RL_ACTION_SPACE:
            action = (action / self.HOVER_THRUST - 1) / self.norm_act_scale

        return action

    def denormalizeAction(self, action):
        """Converts a normalized action into a physical action if necessary"""
        if self.NORMALIZED_RL_ACTION_SPACE:
            action = (1 + self.norm_act_scale * action) * self.HOVER_THRUST

        return action

    def preprocessAction(self, action):
        """Converts the action passed to .step() into motor RPMs"""
        # action = self.denormalizeAction(action)

        # self.action_buffer.append(action)
        # self.current_physical_action = action

        # if np.isnan(action).any():
        #     print("[WARNING] NaN detected in action:", action)

        # thrust = np.clip(action, self.physical_action_bounds[0], self.physical_action_bounds[1])
        # self.current_clipped_action = thrust

        # # convert to quad motor rpm commands
        # pwm = self.cmd2pwm(thrust)
        # rpm = self.pwm2rpm(pwm)

        # return rpm

        # if self.args.enabled:
        clean = self.denormalizeAction(action).astype(np.float32)
        noisy = self.applyActionDisruptor(clean.copy())

        self.last_action_err = float(np.abs(noisy - clean).mean())
        self.action_buffer.append(clean)  # <- intended command
        self.current_cmd = clean  # for logging/obs
        self.current_physical_action = noisy  # what motors get

        thrust = np.clip(noisy, *self.physical_action_bounds)
        pwm = self.cmd2pwm(thrust)
        rpm = self.pwm2rpm(pwm)
        return rpm
        # else:
        #     action = self.denormalizeAction(action)

        #     self.action_buffer.append(action)
        #     self.current_physical_action = action

        #     if np.isnan(action).any():
        #         print("[WARNING] NaN detected in action:", action)

        #     thrust = np.clip(action, self.physical_action_bounds[0], self.physical_action_bounds[1])
        #     self.current_clipped_action = thrust

        #     # convert to quad motor rpm commands
        #     pwm = self.cmd2pwm(thrust)
        #     rpm = self.pwm2rpm(pwm)

        #     return rpm

    def physics(self, rpm):
        """Base PyBullet physics implementation."""
        forces = np.array(rpm**2) * self.KF
        torques = np.array(rpm**2) * self.KM

        # z_torque = -torques[0] + torques[1] - torques[2] + torques[3]
        z_torque = np.sum(self._yaw_dir * torques)

        for i in range(4):
            self._p.applyExternalForce(
                self.drone,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=self._p.LINK_FRAME,
            )

        self._p.applyExternalTorque(
            self.drone,
            4,
            torqueObj=[0, 0, z_torque],
            flags=self._p.LINK_FRAME,
        )

    def loadDrone(self):
        """Loads the drone at a 7 meter fixed distance around the launch pad"""
        # ---- 7 meter distance in 3D sphere around launch pad ---#
        # # ---- Tilt from vertical or horizontal ----#
        # phi = self.rng.uniform(0, np.pi / 2)
        # theta = self.rng.uniform(0, 2 * np.pi)

        # # ---- Fixed radius of 7 meters ----#
        # # ---- Convert spherical to cartesian coordinates ----#
        # radius = 7.0
        # x_off = radius * math.sin(phi) * math.cos(theta)
        # y_off = radius * math.sin(phi) * math.sin(theta)
        # z_off = radius * math.cos(phi)

        # # ---- Pad center ----#
        # pad_x, pad_y, pad_z = self.launch_pad_position

        # # ---- Shift drone spawn by offsets ----#
        # start_x = pad_x + x_off
        # start_y = pad_y + y_off
        # # start_z = pad_z + z_off
        # start_z = random.uniform(1, 3)

        # ---- 7 meter fixed horizontal distance, fixed 2 meter z spawn ----#
        fixed_distance = 7.0
        angle = self.rng.uniform(0, 2 * np.pi)
        pad_x, pad_y, _pad_z = self.launch_pad_position
        start_x = pad_x + fixed_distance * np.cos(angle)
        start_y = pad_y + fixed_distance * np.sin(angle)
        start_z = 2.0

        drone = self._p.loadURDF(
            str(files("deepRL_for_autonomous_drones") / "assets/cf2x.urdf"),
            # str(files("deepRL_for_autonomous_drones") / "assets/cf21x_bullet.urdf"),
            [start_x, start_y, start_z],
            flags=self._p.URDF_USE_INERTIA_FROM_FILE,
        )

        self.drone = drone

    def resetDrone(self):
        """(Re-)initializes the drone's kinematic information"""
        self.pos = np.zeros(3)
        self.quat = np.zeros(4)
        self.rpy = np.zeros(3)
        self.vel = np.zeros(3)
        self.ang_v = np.zeros(3)
        self.last_clipped_action = np.zeros(4)

    def updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.
        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).
        """

        self.pos, self.quat = self._p.getBasePositionAndOrientation(self.drone)
        self.rpy = self._p.getEulerFromQuaternion(self.quat)
        self.vel, self.ang_v = self._p.getBaseVelocity(self.drone)

        # self.quat = self.quat / np.linalg.norm(self.quat) if np.linalg.norm(self.quat) > 0 else np.array([0, 0, 0, 1])

        if np.isnan(self.pos).any():
            print("[WARNING] NaN detected in (pos) state!.")
        if np.isnan(self.quat).any():
            print("[WARNING] NaN detected in (quat) state!.")
        if np.isnan(self.rpy).any():
            print("[WARNING] NaN detected in (rpy) state!.")
        if np.isnan(self.vel).any():
            print("[WARNING] NaN detected in (vel) state!.")
        if np.isnan(self.ang_v).any():
            print("[WARNING] NaN detected in (ang_v) state!.")

    def getDroneStateVector(self):
        """Returns the state vector of the drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the state vector of the drone.
        """

        state = np.hstack(
            [
                self.pos,
                self.quat,
                self.rpy,
                self.vel,
                self.ang_v,
                self.last_clipped_action,
            ]
        )
        return state.reshape(
            20,
        )

    def _groundEffect(self, rpm):
        """
        Simulates ground effect, where the drone experiences increased lift
        when flying closer to the ground. It calculates additional thrust contributions
        for reach rotor. Allows for a more accurate representation of drone behavior during
        low-altitude flight.
        """

        # ---- Kin. info of all links (propellers and center of mass) ----#
        link_states = self._p.getLinkStates(
            self.getDroneID(),
            linkIndices=[0, 1, 2, 3, 4],
            computeLinkVelocity=1,
            computeForwardKinematics=1,
        )

        # ---- Simple, per-propeller ground effects ----#
        prop_heights = np.array(
            [
                link_states[0][0][2],
                link_states[1][0][2],
                link_states[2][0][2],
                link_states[3][0][2],
            ]
        )
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)
        gnd_effects = np.array(rpm**2) * self.KF * self.GND_EFF_COEFF * (self.PROP_RADIUS / (4 * prop_heights)) ** 2
        if np.abs(self.rpy[0]) < np.pi / 2 and np.abs(self.rpy[1]) < np.pi / 2:
            for i in range(4):
                self._p.applyExternalForce(
                    self.getDroneID(),
                    i,
                    forceObj=[0, 0, gnd_effects[i]],
                    posObj=[0, 0, 0],
                    flags=self._p.LINK_FRAME,
                )
