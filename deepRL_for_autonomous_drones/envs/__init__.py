from gymnasium.envs.registration import register

register(
    id="SafetyDroneLanding-v0",
    entry_point="deepRL_for_autonomous_drones.envs.Drone_Controller_RPM:DroneControllerRPM",
)

# register(
#     id="NavigationTask-v0",
#     entry_point="deepRL_for_autonomous_drones.envs.navigation_task:NavigationTask",
# )
# register(
#     id="ObstacleTask-v0",
#     entry_point="deepRL_for_autonomous_drones.envs.obstacle_task:ObstacleTask",
# )
# register(
#     id="LandingTask-v0",
#     entry_point="deepRL_for_autonomous_drones.envs.landing_task:LandingTask",
# )
