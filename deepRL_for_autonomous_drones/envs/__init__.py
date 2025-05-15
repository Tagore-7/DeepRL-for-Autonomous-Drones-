from gymnasium.envs.registration import register

register(id="SafetyDroneLanding-v0", entry_point="deepRL_for_autonomous_drones.envs.Drone_Controller_RPM:DroneControllerRPM")
