import pybullet as p
import numpy as np


def computeSimpleCost(env, observation):
    info = {}
    lidar_cost = 0.0
    collision_cost = 0.0
    cost = 0.0

    if env.observation_type in [2, 4]:
        # ---- Lidar cost ----#
        lidar_obs_distances = observation["lidar"]
        min_distance = float(np.min(lidar_obs_distances) * env.LIDAR_MAX_DISTANCE)

        # The CF2X crazyflie is roughly 0.056m in width <- not the value we want

        # prop links positioned at + or - 0.028 in both x and y axes
        # so diagonal motor-to-motor length is ~0.056m
        # arm span (corner to corner) = sqrt(0.028^2 + 0.028^2) = 0.0396m ?
        # multiply by 2 to get a rough full body diagonal (motor-to-motor)
        # drone diagonal size roughly ~0.079m (0.0396 x 2)

        # arm="0.0397", so arm length is ~4 cm → full size (motor-to-motor) is ~8 cm, or 0.08m
        # testing threshold atm as drone size x 4, so 0.08m x 4 = 0.32m
        threshold = 0.32
        # threshold = 0.24
        if min_distance < threshold:
            lidar_cost = (threshold - min_distance) / threshold
            lidar_cost = np.clip(lidar_cost, 0, 1.0)

    if env.add_obstacles:
        if any(env.getBulletClient().getContactPoints(env.drone.getDroneID(), tree) for tree in env.trees):
            collision_cost = 5.0
            env.crashed = True

    cost = lidar_cost + collision_cost
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

    # TODO: Change this so lidar_cost isn't always being returned, ie when no obstacles
    # return float(cost), float(tilt_cost), float(spin_cost), float(lidar_cost)
    return info


cost_functions = {
    1: computeCost,
    2: computeSimpleCost,
}
