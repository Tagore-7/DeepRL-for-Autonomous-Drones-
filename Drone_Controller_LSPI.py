import numpy as np
import pybullet as p
import time
import random
import math
import xml.etree.ElementTree as ET
from gymnasium.spaces import Box
from Base_Drone_Controller import BaseDroneController  

# ---------------- LSPI Controller Implementation ---------------- #
class LSPIController:
    def __init__(self, gamma=0.95, epsilon=0.15, state_dim=6, action_dim=3, reg_lambda=1e-3):
        self.gamma = gamma         # discount factor
        self.epsilon = epsilon     # exploration probability 
        # quadratic polynomial basis for the state (with a bias term)
        # For s = [ex, ey, ez, evx, evy, evz] we use:
        # ψ(s) = [1, ex, ey, ez, evx, evy, evz, ex^2, ey^2, ez^2, evx^2, evy^2, evz^2]  --> dim=13
        self.state_basis_dim = 13

        # For action a = [ax, ay, az] we use:
        # θ(a) = [1, ax, ay, az, ax^2, ay^2, az^2]  --> dim=7
        self.action_basis_dim = 7

        # Total feature dimension using Kronecker product:
        self.feature_dim = self.state_basis_dim * self.action_basis_dim

        # LSPI weight vector: initialize to zeros
        self.W = np.zeros((self.feature_dim, 1))

        # Matrices for LSPI updates
        self.UPSILON = np.eye(self.feature_dim) * reg_lambda  # regularized initial value (symmetric positive definite)
        self.OMEGA = np.zeros((self.feature_dim, 1))

        # For collecting transitions (we can also update online)
        self.transitions = []

    def state_basis(self, s):
        # s is a numpy array of shape (6,) [ex, ey, ez, evx, evy, evz]
        # Build a basis: [1, s, s^2] where s^2 is element-wise square
        s = np.array(s).flatten()
        return np.concatenate(([1.0], s, s**2))

    def action_basis(self, a):
        # a is a numpy array of shape (3,)
        a = np.array(a).flatten()
        return np.concatenate(([1.0], a, a**2))

    def phi(self, s, a):
        # Return the Kronecker product of state and action basis
        psi = self.state_basis(s)     # dim: 13
        theta = self.action_basis(a)  # dim: 7
        return np.kron(psi, theta).reshape(-1, 1)  # shape: (91, 1)

    def update(self, transition):
        """
        Process one transition and update the LSPI matrices.
        transition: (s, a, r, s_next)
        where:
         - s: current state (6-dim: [ex, ey, ez, evx, evy, evz])
         - a: action taken (3-dim) chosen from our discrete set
         - r: reward received (scalar)
         - s_next: next state (6-dim); if terminal then s_next = None
        """
        s, a, r, s_next = transition
        phi_sa = self.phi(s, a)  # (feature_dim, 1)
        if s_next is None:
            phi_next = np.zeros((self.feature_dim, 1))
        else:
            # Greedy action for s_next using current weights:
            a_next = self.get_policy_action(s_next, DISCRETE_ACTIONS)
            phi_next = self.phi(s_next, a_next)
        # LSPI update: add contribution to matrices
        diff = phi_sa - self.gamma * phi_next  # (feature_dim, 1)
        self.UPSILON += np.dot(phi_sa, diff.T)   # (feature_dim, feature_dim)
        self.OMEGA += phi_sa * r                   # (feature_dim, 1)

    def finalize_update(self):
        # After processing transitions, update weights: W = UPSILON^{-1} * OMEGA
        self.W = np.linalg.solve(self.UPSILON, self.OMEGA)

    def get_Q(self, s, a):
        # Q(s,a) = phi(s,a)^T * W (scalar)
        phi_sa = self.phi(s, a)
        return float(np.dot(phi_sa.T, self.W))

    def get_policy_action(self, s, action_set=None):
        # Return the discrete action from action_set that maximizes Q(s,a).
        # If action_set is not provided, use a default set (to be defined externally)
        if action_set is None:
            raise ValueError("action_set must be provided")
        Q_values = [self.get_Q(s, a) for a in action_set]
        best_idx = np.argmax(Q_values)
        return action_set[best_idx]

    def choose_action(self, s, action_set):
        # With probability epsilon choose a random action; otherwise, choose greedy action.
        if random.random() < self.epsilon:
            return random.choice(action_set)
        else:
            return self.get_policy_action(s, action_set)

    def decay_epsilon(self, decay_rate=0.99, min_epsilon=0.01):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)


# ---------------- Discrete Action Set Definition ---------------- #
# We create a set of 18 non-uniform actions (each a 3D vector).
# These are chosen to give finer control near zero and coarser control further out.
DISCRETE_ACTIONS = [
    np.array([0.0,  0.0,  0.0]),  # hover
    np.array([0.1,  0.0,  0.0]),
    np.array([-0.1, 0.0,  0.0]),
    np.array([0.0,  0.1,  0.0]),
    np.array([0.0, -0.1,  0.0]),
    np.array([0.0,  0.0,  0.1]),
    np.array([0.0,  0.0, -0.1]),
    np.array([0.2,  0.0,  0.0]),
    np.array([-0.2, 0.0,  0.0]),
    np.array([0.0,  0.2,  0.0]),
    np.array([0.0, -0.2,  0.0]),
    np.array([0.0,  0.0,  0.2]),
    np.array([0.0,  0.0, -0.2]),
    np.array([0.1,  0.1,  0.0]),
    np.array([-0.1, -0.1, 0.0]),
    np.array([0.1, -0.1,  0.0]),
    np.array([-0.1, 0.1,  0.0]),
    np.array([0.1,  0.1,  0.1])
]


# ---------------- Drone Controller LSPI Implementation ---------------- #
class DroneControllerLSPI(BaseDroneController):
    def __init__(self, args):
        super(DroneControllerLSPI, self).__init__(args=args)
        # Initialize LSPI Controller
        self.lspi = LSPIController(gamma=0.95, epsilon=0.15)
        # LSPI will be trained on state = [pos_error (3), vel_error (3)]
        # For landing, we assume target velocity is zero.
        # For the landing task, we also use the exponential reward function from the paper.
        # Hyperparameters for reward function:
        self.alpha = np.array([1.0, 1.0, 1.0])  # these can be tuned as in the paper
        self.beta  = np.array([1.0, 1.0, 1.0])
        self.c1 = None  # will be computed per step
        self.c2 = None

        # For storing transitions (for LSPI update)
        self.transition_buffer = []
        # Set parameters for LSPI update frequency (e.g., update after each episode)
        self.episode_transitions = []

    def _actionSpace(self):
        # For LSPI, we use our discrete action set.
        # But we still define the action space for Gym as a discrete space.
        # However, our step() method will return continuous forces based on the selected discrete action.
        from gymnasium.spaces import Discrete
        self.action_space = Discrete(len(DISCRETE_ACTIONS))
        return self.action_space

    def _observationSpace(self):
        # Our observation will be similar to before; however, LSPI uses only position and velocity errors.
        # Here we keep the full observation but we compute state error internally.
        lo = -np.inf
        hi = np.inf
        # observation: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        self.observation_space = Box(low=lo, high=hi, shape=(12,), dtype=np.float32)
        return self.observation_space

    def reset(self, seed=None):
        obs, info = super().reset(seed)
        self.episode_transitions = []
        return obs, info

    def compute_state_error(self, obs):
        # obs: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        # For landing, target position is self.target_pos and target velocity is assumed to be zero.
        pos = np.array(obs[0:3])
        vel = np.array(obs[3:6])
        # Compute error
        pos_error = self.target_pos - pos   # target_pos is set from launch pad position
        vel_error = -vel                      # since target velocity is zero
        state_error = np.concatenate((pos_error, vel_error))
        return state_error

    def compute_reward(self, state_error):
        # Replicate the exponential reward function from the paper:
        # R(s, a) = -α^T (c1 * exp(2|e_pos|)) - β^T (c2 * exp(2|e_vel|))
        # For simplicity, we assume c1 = 1 - exp(-|e_pos|) and c2 = exp(-|e_pos|)
        pos_error = state_error[0:3]
        vel_error = state_error[3:6]
        c1 = 1 - np.exp(-np.abs(pos_error))
        c2 = np.exp(-np.abs(pos_error))
        reward = - np.dot(self.alpha, c1) - np.dot(self.beta, c2)
        return reward

    def step(self, action_index=None):
        """
        In this LSPI controller, at each step we:
         1. Get current observation and compute state error.
         2. Select action using LSPI (if not in training, use the learned policy).
         3. Apply a simple velocity tracking controller to drive the drone.
         4. Collect transition (s, a, r, s_next) for LSPI update.
         5. Compute reward using the exponential reward function.
        """
        # Get current observation and drone state:
        obs = self._get_observation()  # shape: (12,)
        state_error = self.compute_state_error(obs)  # shape: (6,)

        # LSPI action selection:
        # Choose an action from our discrete action set
        lspi_action = self.lspi.choose_action(state_error, DISCRETE_ACTIONS)

        # For our control, we treat the LSPI action as desired velocity command.
        # Compute a new target position by adding (desired velocity * dt) to current position.
        current_pos = np.array(obs[0:3])
        dt = self.time_step
        desired_vel = lspi_action  # LSPI action is a vector (3,) within a scale (e.g., 0.1 to 0.2)
        target_pos_lspi = current_pos + desired_vel * dt

        # Here, we use a simple PD-like control to drive the drone toward target_pos_lspi.
        # (This is a minimal controller; in a more detailed implementation, you may compute rotor commands.)
        position_error = target_pos_lspi - current_pos
        # Simple proportional controller for force:
        Kp = 100.0
        force_cmd = Kp * position_error

        # Apply force commands to the drone:
        # We apply the same force at each rotor (a rough approximation).
        # You may improve this by integrating with a full mixer.
        num_rotors = 4
        rotor_forces = (force_cmd[2] / num_rotors) if force_cmd[2] > 0 else 0  # focus on vertical force for landing
        for i in range(num_rotors):
            p.applyExternalForce(
                self.drone,
                i,
                forceObj=[0, 0, rotor_forces],
                posObj=self.rotor_positions_local[i],
                flags=p.LINK_FRAME,
            )

        # Step simulation
        p.stepSimulation()
        if self.args.visual_mode.upper() == "GUI":
            time.sleep(self.time_step)

        # Get next observation
        next_obs = self._get_observation()
        next_state_error = self.compute_state_error(next_obs)

        # Compute reward
        reward = self.compute_reward(state_error)

        # Check termination conditions
        terminated = self._is_done(next_obs)
        truncated = False  # no time truncation logic here

        # Save the transition for LSPI update:
        # If terminal, we use s_next = None
        transition = (state_error, lspi_action, reward, None if terminated else next_state_error)
        self.episode_transitions.append(transition)

        # Return observation, reward, termination flags, and info
        return next_obs, reward, terminated, truncated, {}

    def update_lspi(self):
        # Process all collected transitions in the episode and update LSPI weights
        for trans in self.episode_transitions:
            self.lspi.update(trans)
        self.lspi.finalize_update()
        # Decay exploration probability
        self.lspi.decay_epsilon()
        # Clear episode transitions
        self.episode_transitions = []

# ---------------- Main Training Loop ---------------- #
import argparse
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import multiprocessing
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

def parse_args():
    parser = argparse.ArgumentParser(description='LSPI for drone landing task')
    parser.add_argument('--visual_mode', type=str, default="DIRECT", help='Visual mode: GUI or DIRECT')
    parser.add_argument('--launch_pad_position', type=lambda x: np.array([float(i) for i in x.split(",")]),
                        default="0,0,0", help='Launch pad position')
    parser.add_argument('--boundary_limits', type=int, default=20, help='Boundary limits for the drone')
    parser.add_argument('--gravity', type=float, default=-9.8, help='Gravity value')
    parser.add_argument('--tensorboard_log_dir', type=str, default='./logs_metrics_benchmark_tensorboard/', help='TensorBoard log directory')
    parser.add_argument('--distance_reward_weight', type=float, default=2.0,
                        help='The weight for the distance reward (distance between drone and launch pad)')
    parser.add_argument('--leg_contact_reward', type=int, default=100,
                        help='The reward for the drone making contact with the launch pad')
    return parser.parse_args()

args = parse_args()

# We create an environment using our DroneControllerLSPI
class DroneLandingEnv(gym.Env):
    def __init__(self, args):
        super(DroneLandingEnv, self).__init__()
        self.controller = DroneControllerLSPI(args)
        self.action_space = self.controller._actionSpace()
        self.observation_space = self.controller._observationSpace()
        self.args = args

    def reset(self, seed=None, options=None):
        obs, info = self.controller.reset(seed)
        return obs, info

    def step(self, action=None):
        # We ignore external actions and let LSPI select actions internally.
        obs, reward, done, truncated, info = self.controller.step()
        return obs, reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        self.controller.close()

def main():
    # Use a single environment for LSPI training (LSPI is not a black-box deep RL algorithm)
    env = DroneLandingEnv(args)
    num_episodes = 100  # total training episodes
    max_steps_per_episode = 5000

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        for step in range(max_steps_per_episode):
            # LSPI controller ignores external action (we use None)
            obs, reward, done, truncated, info = env.step()
            total_reward += reward
            if done:
                break
        # After each episode, update LSPI using collected transitions:
        env.controller.update_lspi()
        print(f"Episode {ep+1} total reward: {total_reward:.2f}, epsilon: {env.controller.lspi.epsilon:.3f}")
    env.close()

if __name__ == "__main__":
    main()
