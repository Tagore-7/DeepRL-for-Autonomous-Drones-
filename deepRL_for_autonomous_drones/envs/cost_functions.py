import pybullet as p
import numpy as np


def _ramp(x, x0, x1):
    """
    Piece-wise linear ramp:
        0            if x <= x0
        (x-x0)/(x1-x0)  if x0 < x < x1
        1            if x >= x1
    """
    return np.clip((x - x0) / (x1 - x0), 0.0, 1.0)


def computeSimpleCost(env, observation):
    info = {}
    lidar_cost = 0.0
    collision_cost = 0.0
    plane_cost = 0.0
    tilt_cost = 0.0
    spin_cost = 0.0
    cost = 0.0

    if env.observation_type in [2, 4]:
        # ---- Lidar cost ----#
        lidar_obs_distances = observation["lidar"]
        min_distance = float(np.min(lidar_obs_distances) * env.LIDAR_MAX_DISTANCE)

        threshold = 0.32
        # threshold = 0.24
        if min_distance < threshold:
            lidar_cost = (threshold - min_distance) / threshold
            lidar_cost = np.clip(lidar_cost, 0, 1.0)

    if env.add_obstacles:
        if any(env.getBulletClient().getContactPoints(env.drone.getDroneID(), tree) for tree in env.trees):
            collision_cost = 5.0
            env.crashed = True

    contact_points_plane = env.getBulletClient().getContactPoints(env.drone.getDroneID(), env.plane)
    if contact_points_plane:
        plane_cost = 5.0
        env.crashed = True

    obs = env.drone.getDroneStateVector()
    roll, pitch, _yaw = obs[7:10]
    wx, wy, wz = obs[13:16]
    max_tilt = max(abs(roll), abs(pitch))
    tilt_soft = np.deg2rad(30)  # start charging here
    tilt_hard = np.deg2rad(60)
    tilt_cost = _ramp(max_tilt, tilt_soft, tilt_hard) * 1.0

    spin_mag = np.linalg.norm([wx, wy, wz])  # rad s-1
    spin_soft = 3.0
    spin_hard = 6.0
    spin_cost = _ramp(spin_mag, spin_soft, spin_hard) * 1.0

    cost = lidar_cost + collision_cost + plane_cost + tilt_cost + spin_cost
    # cost = lidar_cost + collision_cost + plane_cost
    info["cost"] = cost
    return info


def computeCost(env, observation):
    info = {}

    cost = 0.0
    lidar_cost = 0.0
    tilt_cost = 0.0
    spin_cost = 0.0

    if env.observation_type in [2, 4]:
        # ---- Lidar cost ----#
        lidar_obs_distances = observation["lidar"]
        min_distance = float(np.min(lidar_obs_distances) * env.LIDAR_MAX_DISTANCE)

        # The CF2X crazyflie is roughly 0.056m in width
        threshold = 0.1
        if min_distance < threshold:
            lidar_cost = (threshold - min_distance) / threshold
            lidar_cost = np.clip(lidar_cost, 0, 1.0)
            info["lidar_cost"] = lidar_cost
            cost += lidar_cost

    # ---- Plane crash cost ----#
    if env.getBulletClient().getContactPoints(env.drone.getDroneID(), env.plane):
        cost += 5.0
        # self.logger.info("Crashed into plane")
        env.crashed = True

    # ---- Obstacle crash cost ----#
    if env.add_obstacles:
        if any(env.getBulletClient().getContactPoints(env.drone.getDroneID(), tree) for tree in env.trees):
            cost += 5.0
            # self.logger.info("Crashed into a tree")
            env.crashed = True

    obs = env.drone.getDroneStateVector()
    roll, pitch, _yaw = obs[7:10]
    wx, wy, wz = obs[13:16]
    tilt_penalty = abs(roll) + abs(pitch)
    spin_penalty = abs(wx) + abs(wy) + abs(wz)
    # tilt_cost = 0.1 * tilt_penalty
    # spin_cost = 0.1 * spin_penalty

    tilt_cost = np.clip(tilt_penalty / np.pi, 0, 1.0) * 0.5  # max ~0.5
    spin_cost = np.clip(spin_penalty / 20.0, 0, 1.0) * 0.5  # max ~0.5
    info["tilt_cost"] = tilt_cost
    info["spin_cost"] = spin_cost
    cost += tilt_cost + spin_cost

    info["cost"] = cost

    # cost /= self.CTRL_STEPS

    # TO-DO: Change this so lidar_cost isn't always being returned, ie when no obstacles
    # return float(cost), float(tilt_cost), float(spin_cost), float(lidar_cost)
    return info


cost_functions = {
    1: computeCost,
    2: computeSimpleCost,
}
