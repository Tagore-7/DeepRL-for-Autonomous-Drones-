import pybullet as p
import numpy as np


def safetyRewardFunction(env, observation, action, tilt_cost=0.0, spin_cost=0.0, lidar_cost=0.0):
    """
    Computes the reward function based on the author's paper: Towards end-to-end control for UAV autonomous landing via deep reinforcement learning,

    """
    obs = env.drone.getDroneStateVector()
    px, py, pz = obs[0:3]  # Position
    vx, vy, vz = obs[10:13]  # Linear velocity
    roll, pitch, _yaw = obs[7:10]
    wx, wy, wz = obs[13:16]
    ax, ay, az, aw = (
        action[0],
        action[1],
        action[2],
        action[3],
    )  # Actions from the agent

    rel_pos = np.array([px, py, pz]) - env.target_pos
    rel_vel = np.array([vx, vy, vz]) - np.array([0, 0, 0])
    distance_penalty = np.linalg.norm(rel_pos)  # Distance penalty
    velocity_penalty = np.linalg.norm(rel_vel)  # Velocity penalty
    action_penalty = np.linalg.norm([ax, ay, az, aw])  # Action penalty
    tilt_penalty = abs(roll) + abs(pitch)  # Weighted penalty for large tilt
    spin_penalty = abs(wx) + abs(wy) + abs(wz)  # Weighted penalty for high spin

    # Compute shaping reward
    shaping = -100 * distance_penalty - 10 * velocity_penalty - 1 * action_penalty - 5 * tilt_penalty - 5 * spin_penalty

    # Check if drone has landed safely
    contact_points = p.getContactPoints(env.drone.getDroneID(), env.launch_pad)

    if contact_points:
        if abs(vx) <= 0.1 and abs(vy) <= 0.1 and abs(vz) <= 0.1:
            print("Super soft landing")
        elif abs(vx) < 0.3 and abs(vy) < 0.3 and abs(vz) < 0.3:
            print("Landed")
        # elif abs(vx) < 0.5 and abs(vy) < 0.5 and abs(vz) < 0.5:
        #     print("Hard landing")
        # else:
        #     print("Crashed")
        env.c = 10 * (1 - abs(ax)) + 10 * (1 - abs(ay)) + 10 * (1 - abs(az)) + 10 * (1 - abs(aw))  # Bonus for throttle tending to zero
        shaping += env.c
        env.landed = True

    contact_points_plane = p.getContactPoints(env.drone.getDroneID(), env.plane)
    if contact_points_plane:
        shaping -= 100
        env.crashed = True

    # shaping += obstacleRewardShaping(env)

    # Reward difference (temporal difference shaping)
    reward = shaping if env.previous_shaping is None else shaping - env.previous_shaping
    env.previous_shaping = shaping

    return reward / 50


def firstRewardFunction(env, observation, action, tilt_cost=0.0, spin_cost=0.0, lidar_cost=0.0):
    """
    Computes the reward function based on the author's paper: Towards end-to-end control for UAV autonomous landing via deep reinforcement learning,

    """

    # ---- CONSTANTS ----#
    DISTANCE_WEIGHT = 100.0  # -100
    VELOCITY_WEIGHT = 20.0  # -10
    ACTION_WEIGHT = 1.0  # -1
    SOFT_LAND_REWARD = 50.0
    HARD_LAND_REWARD = 10.0
    ALIVE_BONUS = 1.0
    SCALE = 100.0

    obs = env.drone.getDroneStateVector()
    px, py, pz = obs[0:3]  # Position
    vx, vy, vz = obs[10:13]  # Linear velocity
    roll, pitch, _yaw = obs[7:10]
    ax, ay, az, aw = action

    rel_pos = np.array([px, py, pz]) - env.target_pos
    rel_vel = np.array([vx, vy, vz]) - np.array([0, 0, 0])

    distance_penalty = np.linalg.norm(rel_pos)  # Distance penalty
    velocity_penalty = np.linalg.norm(rel_vel)  # Velocity penalty
    action_penalty = np.linalg.norm([ax, ay, az, aw])  # Action penalty

    # Compute shaping reward
    shaping = -DISTANCE_WEIGHT * distance_penalty - VELOCITY_WEIGHT * velocity_penalty - ACTION_WEIGHT * action_penalty

    # --- lidar proximity shaping (soft buffer) ----------------------
    if env.observation_type in (2, 4) and env.add_obstacles:
        lidar_min = np.min(observation["lidar"]) * env.LIDAR_MAX_DISTANCE
        prox_thr = 0.24  # ↑ 50 cm buffer
        if lidar_min < prox_thr:
            lidar_pen = (prox_thr - lidar_min) / prox_thr  # 0→1
            shaping -= 20.0 * lidar_pen  # extra term

    # Check if drone has landed safely
    contact_points = p.getContactPoints(env.drone.getDroneID(), env.launch_pad)
    landing_bonus = 0.0
    if contact_points:
        safe_landing = abs(vx) <= 0.3 and abs(vy) <= 0.3 and abs(vz) <= 0.3
        landing_bonus = SOFT_LAND_REWARD if safe_landing else HARD_LAND_REWARD
        env.c = 10 * (1 - abs(ax)) + 10 * (1 - abs(ay)) + 10 * (1 - abs(az)) + 10 * (1 - abs(aw))  # Bonus for throttle tending to zero
        shaping += env.c
        env.landed = True

    # Reward difference (temporal difference shaping)
    # delta = shaping if env.previous_shaping is None else shaping - env.previous_shaping
    delta = shaping - (env.previous_shaping or 0.0)
    env.previous_shaping = shaping

    # New: Additional shaping reward
    # proximity_bonus = max(0, 5.0 - distance_penalty)  # Up to 5 pts as drone gets close
    # reward += proximity_bonus

    # d = 0.5 * np.pi
    # alive: bool = abs(pitch) <= d and abs(roll) <= d
    # alive_reward = 1.0 if alive else -10.0
    # reward += alive_reward

    # Add safe-hover bonus if clear from obstacles and not spinning
    alive = tilt_cost < 0.4 and spin_cost < 0.4 and lidar_cost < 0.2
    # if alive:
    #     reward += ALIVE_BONUS

    reward = delta + landing_bonus + (ALIVE_BONUS if alive else 0.0)
    reward = reward / SCALE

    return reward


def originalFirstRewardFunction(env, observation, action, tilt_cost=0.0, spin_cost=0.0, lidar_cost=0.0):
    """
    Computes the reward function based on the author's paper: Towards end-to-end control for UAV autonomous landing via deep reinforcement learning,

    """
    obs = env.drone.getDroneStateVector()
    px, py, pz = obs[0:3]  # Position
    vx, vy, vz = obs[10:13]  # Linear velocity
    roll, pitch, _yaw = obs[7:10]
    wx, wy, wz = obs[13:16]
    ax, ay, az, aw = (
        action[0],
        action[1],
        action[2],
        action[3],
    )  # Actions from the agent

    plane_penalty = 0

    rel_pos = np.array([px, py, pz]) - env.target_pos
    rel_vel = np.array([vx, vy, vz]) - np.array([0, 0, 0])
    distance_penalty = np.linalg.norm(rel_pos)  # Distance penalty
    velocity_penalty = np.linalg.norm(rel_vel)  # Velocity penalty
    action_penalty = np.linalg.norm([ax, ay, az, aw])  # Action penalty
    tilt_penalty = abs(roll) + abs(pitch)  # Weighted penalty for large tilt
    spin_penalty = abs(wx) + abs(wy) + abs(wz)  # Weighted penalty for high spin

    # ---- Lidar reward ----#
    lidar_obs_distances = observation["lidar"]
    min_distance = np.min(lidar_obs_distances) * env.LIDAR_MAX_DISTANCE
    lidar_reward = 1 - np.exp(-2 * min_distance)

    # Compute shaping reward
    shaping = -100 * distance_penalty - 10 * velocity_penalty - 1 * action_penalty + 10 * lidar_reward - 5 * tilt_penalty - 5 * spin_penalty

    # Check if drone has landed safely
    contact_points = p.getContactPoints(env.drone.getDroneID(), env.launch_pad)

    if contact_points:
        if abs(vx) <= 0.1 and abs(vy) <= 0.1 and abs(vz) <= 0.1:
            print("Super soft landing")
        elif abs(vx) < 0.3 and abs(vy) < 0.3 and abs(vz) < 0.3:
            print("Landed")
        elif abs(vx) < 0.5 and abs(vy) < 0.5 and abs(vz) < 0.5:
            print("Hard landing")
        else:
            print("Crashed")
        env.c = 10 * (1 - abs(ax)) + 10 * (1 - abs(ay)) + 10 * (1 - abs(az)) + 10 * (1 - abs(aw))  # Bonus for throttle tending to zero
        shaping += env.c
        env.landed = True

    contact_points_plane = p.getContactPoints(env.drone.getDroneID(), env.plane)
    if contact_points_plane:
        plane_penalty -= 100
        if abs(roll) > 1 or abs(pitch) > 1:
            shaping -= 50
        env.crashed = True

    shaping += obstacleRewardShaping(env)

    # Reward difference (temporal difference shaping)
    reward = shaping if env.previous_shaping is None else shaping - env.previous_shaping
    env.previous_shaping = shaping

    reward = reward + plane_penalty
    return reward


def secondRewardFunction(env, observation, _action, tilt_cost=0.0, spin_cost=0.0, lidar_cost=0.0):
    """
    Computes the reward function based on the author's paper: A reinforce-ment learning approach for autonomous control and landing of a quadrotor,

    reward using the exponential reward function:

    R(s, a) = -α^T [c1 · exp(2|e_pos|)] - β^T [c2 · exp(2|e_vel|)]

    where:
    e_pos = target_pos - current_pos,
    e_vel = target_vel - current_vel (assumed to be zero for landing),
    c1 = 1 - exp(-|e_pos|),
    c2 = exp(-|e_pos|).

    Parameters
    - observation[0:3] contains the current position.
    - observation[3:6] contains the current linear velocity.
    - env.target_pos is defined as the landing pad’s position.
    - env.alpha and env.beta are arrays (e.g., np.array([1.0, 1.0, 1.0])) tune the hyperparameters.
    """
    # Extract current state information:
    current_pos = np.array(observation[0:3])
    # velocity indices:
    current_vel = np.array(observation[6:9])

    # Define target state (for landing, target velocity is zero)
    target_pos = env.target_pos
    target_vel = np.array([0.0, 0.0, 0.0])

    # Compute errors:
    e_pos = target_pos - current_pos
    e_vel = target_vel - current_vel
    abs_e_pos = np.abs(e_pos)
    abs_e_vel = np.abs(e_vel)

    # Compute c1 and c2 as per the paper:
    c1 = 1 - np.exp(-abs_e_pos)
    c2 = np.exp(-abs_e_pos)

    # Compute the reward:
    # This implements: R(s, a) = -α^T [c1 * exp(2|e_pos|)] - β^T [c2 * exp(2|e_vel|)]
    reward = -np.dot(env.alpha, c1 * np.exp(2 * abs_e_pos)) - np.dot(env.beta, c2 * np.exp(2 * abs_e_vel))

    contact_points = p.getContactPoints(env.drone, env.launch_pad)
    if contact_points:
        if np.all(np.abs(current_vel) == 0.0):
            env.landed = True
            reward += 100
        else:
            env.crashed = True

    contact_points_plane = p.getContactPoints(env.drone, env.plane)
    if contact_points_plane:
        reward -= 50
        env.crashed = True

    reward += obstacleRewardShaping(env=env)

    return reward


def thirdRewardFunction(env, observation, action, tilt_cost=0.0, spin_cost=0.0, lidar_cost=0.0):
    """
    Computes the reward function based on the author's paper: "Inclined Quadrotor Landing using Deep Reinforcement Learning"

    r_k = -e_p - 0.2 * e_v - 0.1 * e_{phi, theta} - (0.1 * a_{phi, theta}^2) / max(e_p, 0.001)

    - e_p = || target_pos - current_pos ||  (Position error)
    - e_v = || current_vel ||              (Velocity error)
    - e_{phi, theta} = || roll_error, pitch_error || (Orientation error)
    - a_{phi, theta} = || roll_action, pitch_action || (Control effort penalty)

    This function is designed for **horizontal landing** on a **flat surface**.

    Parameters:
    - observation[0:3]: Current position [x, y, z]
    - observation[3:6]: Current velocity [vx, vy, vz]
    - observation[6:8]: Roll and Pitch angles [roll, pitch]
    - action[0:2]: Control inputs affecting roll and pitch [ax, ay]

    Returns:
    - reward (float): Computed reward value
    """
    # Extract current state information
    current_pos = np.array(observation[0:3])  # Position (x, y, z)
    current_vel = np.array(observation[6:9])  # Velocity (vx, vy, vz)
    roll_pitch = np.array(observation[3:5])  # Roll and pitch angles (phi, theta)

    # Target state (Landing pad)
    target_pos = env.target_pos  # (0,0,0) or defined landing pad

    # Compute errors
    e_p = np.linalg.norm(target_pos - current_pos)  # Position error (L2 norm)
    e_v = np.linalg.norm(current_vel)  # Velocity error (L2 norm)
    e_phi_theta = np.linalg.norm(roll_pitch)  # Orientation error (roll & pitch)

    # Control effort penalty (Action is for roll & pitch)
    a_phi_theta = np.linalg.norm(action[0:2])  # Only first two actions affect roll & pitch

    # Compute final reward
    reward = -e_p - 0.2 * e_v - 0.1 * e_phi_theta - (0.1 * (a_phi_theta**2)) / max(e_p, 0.001)

    # Check if the drone has landed safely
    contact_points = p.getContactPoints(env.drone, env.launch_pad)
    if contact_points:
        if np.all(np.abs(current_vel) == 0.0):  # If the velocity is low
            env.landed = True
            reward += 100  # Small bonus for successful landing
        else:
            env.crashed = True

    contact_points_plane = p.getContactPoints(env.drone, env.plane)
    if contact_points_plane:
        reward -= 50
        env.crashed = True

    reward += obstacleRewardShaping(env=env)

    return reward


def obstacleRewardShaping(env):
    reward = 0

    if env.add_obstacles:
        if any(env.getBulletClient().getContactPoints(env.drone.getDroneID(), tree) for tree in env.trees):
            # print("Hit a tree")
            reward -= 25
            env.crashed = True

    return reward


# ---- Add reward functions here ----#
reward_functions = {
    1: firstRewardFunction,
    2: secondRewardFunction,
    3: thirdRewardFunction,
    4: safetyRewardFunction,
}
