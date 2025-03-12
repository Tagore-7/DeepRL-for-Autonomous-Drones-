import argparse
import os 
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from gymnasium.wrappers import FlattenObservation
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
import xml.etree.ElementTree as ET
import math
import noise
import multiprocessing
import pkg_resources

from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from sb3_contrib import ARS, CrossQ, TQC, TRPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from deepRL_for_autonomous_drones.envs.Base_Drone_Controller import BaseDroneController
from deepRL_for_autonomous_drones.utils.Custom_Callbacks import EpisodeRewardCallback, ToggleWindCallback, ToggleStaticBlocksCallback, ToggleDonutObstaclesCallback, ToggleMovingBlocksCallback, SaveModelCallback

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmarking DeepRL algorithms for drone landing task')
    parser.add_argument('--algorithm_name', type=str, default='PPO',
                        help='The name of the algorithm to benchmark. Options: PPO, A2C, DDPG, TD3, SAC, ARS, CROSSQ, TQC, TRPO')
    parser.add_argument('--boundary_limits', type=int, default=20,
                        help='The boundary limits for the drone to fly in the environment')
    parser.add_argument('--launch_pad_position', type=lambda x: np.array([float(i) for i in x.split(",")]),
                        default=np.array([0, 0, 0]),
                        help='The position of the launch pad for the drone to land on in the environment')
    parser.add_argument('--gravity', type=float, default=-9.8,
                        help='The gravity value for the environment')
    parser.add_argument('--distance_reward_weight', type=float, default=2.0,
                        help='The weight for the distance reward (distance between drone and launch pad)')
    parser.add_argument('--leg_contact_reward', type=int, default=100,
                        help='The reward for the drone making contact with the launch pad')
    parser.add_argument('--tensorboard_log_dir', type=str, default='./logs_metrics_benchmark_tensorboard/',
                        help='The directory to store TensorBoard logs')
    parser.add_argument('--model_name_to_save', type=str, default='drone_landing_model_using_ppo',
                        help='Name of the model to save')
    parser.add_argument('--visual_mode', type=str, default="DIRECT",
                        help='Visual mode of the environment: GUI or DIRECT')
    parser.add_argument('--discount_factor', type=float, default=0.99,
                        help='Discount factor (gamma) for the RL algorithm')
    parser.add_argument('--reward_function', type = int, default= 1, 
                        help = 'Which reward function you want to use: 1 or 2 or 3')
    parser.add_argument('--debug_axes', type=bool, default=False,
                        help='Draws visual lines for drone axes for debugging')
    parser.add_argument('--add_obstacles', type=bool, default=False,
                        help='Determines if there will obstacles')
    parser.add_argument('--enable_wind', type=bool, default=False,
                        help='Determines if there will be wind effects applied to the drone')
    parser.add_argument('--enable_ground_effect', type=bool, default=False,
                        help='Determines if there will be ground effects applied to the drone')
    return parser.parse_args()

args = parse_args()

class DroneControllerRPM(BaseDroneController):
    def __init__(self):
        super(DroneControllerRPM, self).__init__(args=args)
        self.enable_wind: bool = self.args.enable_wind
        self.reward_function = self.args.reward_function

    def _actionSpace(self): 
        #---- RPMs of the four drone rotors ----#
        #---- [RPM_0, RPM_1, RPM_2, RPM_3]  ----#
        act_lower_bound = np.array([-1, -1, -1, -1], dtype=np.float32)
        act_upper_bound = np.array([ 1,  1,  1,  1], dtype=np.float32)

        #---- Add action buffer to action space ----#
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros(4))

        self.action_space = Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
        return self.action_space

    def _observationSpace(self):
        #---- The drone state dimensions                           ----#
        #---- [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]  ----#
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo])
        obs_upper_bound = np.array([hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi])

        #---- Add action buffer to observation space ----#
        #---- Adds the drones RPM actions to the observation space            ----#
        #---- So that the RL agent sees a recent history of actions performed ----#
        act_lo = -1
        act_hi = +1
        for _ in range(self.ACTION_BUFFER_SIZE):
            obs_lower_bound = np.hstack([obs_lower_bound, np.array([act_lo,act_lo,act_lo,act_lo])])
            obs_upper_bound = np.hstack([obs_upper_bound, np.array([act_hi,act_hi,act_hi,act_hi])])

        #---- LiDAR box space      ----#
        #---- Normalized in [0, 1] ----#
        lidar_dim = self.LIDAR_NUM_RAYS
        lidar_low = np.zeros((lidar_dim,), dtype=np.float32)
        lidar_high = np.ones((lidar_dim,), dtype=np.float32)

        if self.add_obstacles:
            self.observation_space = Dict({
                "drone_state": Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32),
                "lidar": Box(low=lidar_low, high=lidar_high, dtype=np.float32)
            })
        else:
            self.observation_space = Dict({
                "drone_state": Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            })

        return self.observation_space

    def _dragWind(self):
        # _, orientation = p.getBasePositionAndOrientation(self.drone)
        # linear_vel, _ = p.getBaseVelocity(self.drone)
        # base_rot = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        # relative_velocity = np.array(linear_vel) - self.wind_force

        base_rot = np.array(p.getMatrixFromQuaternion(self.quat)).reshape(3, 3)
        relative_velocity = np.array(self.vel) - self.wind_force

        drag = np.dot(base_rot.T, self.DRAG_COEFF * np.array(relative_velocity))
        p.applyExternalForce(
            self.drone,
            4,
            forceObj=drag,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
        )

    def _groundEffect(self, rpm):
        """
        Simulates ground effect, where the drone experiences increased lift
        when flying closer to the ground. It calculates additional thrust contributions
        for reach rotor. Allows for a more accurate representation of drone behavior during
        low-altitude flight.
        """

        #---- Kin. info of all links (propellers and center of mass) ----#
        link_states = p.getLinkStates(
            self.drone,
            linkIndices=[0, 1, 2, 3, 4],
            computeLinkVelocity=1,
            computeForwardKinematics=1
        )

        #---- Simple, per-propeller ground effects ----#
        prop_heights = np.array([link_states[0][0][2], link_states[1][0][2], link_states[2][0][2], link_states[3][0][2]])
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)
        gnd_effects = np.array(rpm**2) * self.KF * self.GND_EFF_COEFF * (self.PROP_RADIUS/(4 * prop_heights))**2
        if np.abs(self.rpy[0]) < np.pi/2 and np.abs(self.rpy[1]) < np.pi/2:
            for i in range(4):
                p.applyExternalForce(
                    self.drone,
                    i,
                    forceObj=[0, 0, gnd_effects[i]],
                    posObj=[0, 0, 0],
                    flags=p.LINK_FRAME
                )

    def _normalizedActionToRPM(self, action):
        if np.any(np.abs(action) > 1):
            print("Normalized action out of bounds")
        return np.where(action <= 0, (action+1)*self.HOVER_RPM, self.HOVER_RPM + (self.MAX_RPM - self.HOVER_RPM)*action)
        
    def _preprocessAction(self, action):
        self.action_buffer.append(action)
        rpm = np.array(self.HOVER_RPM * (1+0.05*action))

        return rpm
        
    def _physics(self, rpm):
        forces = np.array(rpm**2) * self.KF
        torques = np.array(rpm**2) * self.KM

        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])

        for i in range(4):
            p.applyExternalForce(
                self.drone,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
            )
        p.applyExternalTorque(
            self.drone,
            4,
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME,
        )
  
    def step(self, action):
        clipped_action = np.reshape(self._preprocessAction(action), 4)

        if self.PYB_STEPS_PER_CTRL > 1 and self.enable_ground_effect:
            self._updateAndStoreKinematicInformation()
        for _ in range(self.PYB_STEPS_PER_CTRL):
            self._physics(clipped_action)

            position, _ = p.getBasePositionAndOrientation(self.drone)
            p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=position)
            p.stepSimulation()

            if self.enable_ground_effect:
                self._groundEffect(clipped_action)

            p_s = self.rng.uniform(0, 1) # Probability for wind at each step
            if self._wind_effect_active and p_s < 0.3:
                self._dragWind()

            self.last_clipped_action = clipped_action

            if args.visual_mode.upper() == "GUI":
                time.sleep(self.time_step)

            # Update moving blocks on each simulation step.
            if self.add_obstacles and self._moving_blocks_active:
                self._updateMovingBlocks() 

        #---- Update and store the drones kinematic information ----#
        self._updateAndStoreKinematicInformation()

        observation = self._getObservation()
        reward = self._computeReward(observation, action, self.reward_function)
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()

        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        return observation, reward, terminated, truncated, {}

    def _computeTerminated(self):
        if self.landed:
            return True
        
        return False
        
    def _computeTruncated(self):
        state = self._getDroneStateVector()
        
        if self.crashed or self.step_counter >= self.max_steps:
              return True
        
        contact_with_ground = p.getContactPoints(self.drone, self.plane)
        if contact_with_ground:
            self.crashed = True 
            return True

        if self._static_blocks_active and self.add_obstacles:
            if any(p.getContactPoints(self.drone, block) for block in self.static_blocks):
                self.crashed = True
                return True

        if self._donut_obstacles_active and self.add_obstacles:
            if any(p.getContactPoints(self.drone, obs_obj) for obs_obj in self.obstacles):
                self.crashed = True 
                return True

        if self._moving_blocks_active and self.add_obstacles:
            if self.first_moving_block is not None and p.getContactPoints(self.drone, self.first_moving_block):
                self.crashed = True
                return True
            if self.second_moving_block is not None and p.getContactPoints(self.drone, self.second_moving_block):
                self.crashed = True
                return True
              
        if state[2] <= 0 or abs(state[0]) > self.boundary_limits or abs(state[1]) > self.boundary_limits or state[1] > self.boundary_limits:
              self.crashed = True 
              return True
        
        return False

def main():
    num_cpu = multiprocessing.cpu_count()
    print(f"Number of CPU cores available: {num_cpu}")
    if args.visual_mode.upper() == "GUI":
        num_cpu = 1
        raw_env = Monitor(DroneControllerRPM())
        env = FlattenObservation(raw_env)
    else: 
        num_cpu = multiprocessing.cpu_count()
        def make_env():
            env_ = DroneControllerRPM()
            env_ = FlattenObservation(env_)
            env_ = Monitor(env_)
            return env_
        env = make_vec_env(
            make_env,
            n_envs=num_cpu,
            vec_env_cls=SubprocVecEnv,
            seed=42
        )

    tensorboard_log_dir = args.tensorboard_log_dir
    gamma_value = args.discount_factor
    algorithm_name = args.algorithm_name.upper()
    
    #---- Custom callbacks ----#
    reward_callback = EpisodeRewardCallback()
    toggle_wind = ToggleWindCallback(threshold=int(10e6))
    toggle_static = ToggleStaticBlocksCallback(threshold=int(20e6))
    toggle_donuts = ToggleDonutObstaclesCallback(threshold=int(30e6))
    toggle_moving = ToggleMovingBlocksCallback(threshold=int(40e6))

    #---- Save models at specific timesteps ----#
    save_thresholds = [10e6, 20e6, 30e6, 40e6, 50e6]
    save_paths = [
        "landing_model_10M", 
        "landing_wind_model_wind_20M", 
        "landing_wind_static_model_30M", 
        "landing_wind_static_donuts_model_donuts_40M", 
        "landing_wind_static_donuts_moving_model_50M"
    ]
    save_callback = SaveModelCallback(thresholds=save_thresholds, save_paths=save_paths)

    #---- Choose the model based on the algorithm name ----#
    if algorithm_name == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, gamma=gamma_value, tensorboard_log=tensorboard_log_dir, device="cpu")
    elif algorithm_name == "A2C":
        model = A2C("MlpPolicy", env, verbose=1, gamma=gamma_value, tensorboard_log=tensorboard_log_dir, device="cpu")
    elif algorithm_name == "DDPG":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = DDPG("MlpPolicy", env, verbose=1, gamma=gamma_value, action_noise=action_noise, tensorboard_log=tensorboard_log_dir, device="cpu")
    elif algorithm_name == "TD3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3("MlpPolicy", env, verbose=1, gamma=gamma_value, action_noise=action_noise, tensorboard_log=tensorboard_log_dir, device="cpu")
    elif algorithm_name == "SAC":
        model = SAC("MlpPolicy", env, verbose=1, gamma=gamma_value, tensorboard_log=tensorboard_log_dir, device="cpu", ent_coef='auto', target_entropy='auto')
    elif algorithm_name == "ARS":
        model = ARS("MlpPolicy", env, verbose=1, learning_rate=0.02, delta_std=0.05, n_delta=8, n_top=8, tensorboard_log=tensorboard_log_dir, device="cpu")
    elif algorithm_name == "CROSSQ":
        model = CrossQ("MlpPolicy", env, verbose=1, gamma=gamma_value,
                       tensorboard_log=tensorboard_log_dir, device="cpu")
    elif algorithm_name == "TQC":
        policy_kwargs = dict(n_quantiles=25, n_critics=2)
        model = TQC("MlpPolicy", env, verbose=1, gamma=gamma_value,
                    tensorboard_log=tensorboard_log_dir, device="cpu",
                    top_quantiles_to_drop_per_net=2, policy_kwargs=policy_kwargs)
    elif algorithm_name == "TRPO":
        model = TRPO("MlpPolicy", env, verbose=1, gamma=gamma_value, tensorboard_log=tensorboard_log_dir, device="cpu")

    else:
        raise ValueError(f"Invalid algorithm name: {args.algorithm_name}")

    # For ARS, asynchronous evaluation is experimental and callbacks are not fully supported.
    if algorithm_name not in ["ARS"]:
        eval_callback = EvalCallback(
            env,
            best_model_save_path=tensorboard_log_dir,
            log_path=tensorboard_log_dir,
            eval_freq=10000,
            verbose=1
        )
        # model.learn(total_timesteps=15e6, callback=eval_callback, progress_bar=True)
        # model.learn(total_timesteps=5e6, callback=[eval_callback, reward_callback], progress_bar=True)
        model.learn(
            total_timesteps=50e6, 
            callback=[eval_callback, reward_callback, toggle_wind, toggle_static, toggle_donuts, toggle_moving, save_callback], 
            progress_bar=True
        )
    else:
        # model.learn(total_timesteps=15e6, progress_bar=True)
        model.learn(
            total_timesteps=50e6, 
            callback=[reward_callback, toggle_wind, toggle_static, toggle_donuts, toggle_moving, save_callback], 
            progress_bar=True
        )
    
    model.save(f"{args.model_name_to_save}")
    
    env.close()

if __name__ == "__main__":
    main()
