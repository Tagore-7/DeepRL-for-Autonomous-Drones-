from stable_baselines3.common.callbacks import BaseCallback

class EpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(EpisodeRewardCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        """
        This function is called at every step during training.
        """
        # Get monitor information from SB3 environment
        if "episode" in self.locals["infos"][0]:
            ep_reward = self.locals["infos"][0]["episode"]["r"]
            self.episode_rewards.append(ep_reward)
            print(f"Episode Reward: {ep_reward:.2f}")

        return True
    
class ToggleWindCallback(BaseCallback):
    def __init__(self, threshold: int, verbose=0):
        super(ToggleWindCallback, self).__init__(verbose)
        self.threshold = threshold
        self.has_toggled = False

    def _on_step(self) -> bool:
        if not self.has_toggled and self.model.num_timesteps >= self.threshold:
            print(f"Enabling wind effect at timestep: {self.model.num_timesteps}")
            self.training_env.env_method("setWindEffects", True)
            self.has_toggled = True
        return True
    
class ToggleStaticBlocksCallback(BaseCallback):
    def __init__(self, threshold: int, verbose=0):
        super(ToggleStaticBlocksCallback, self).__init__(verbose)
        self.threshold = threshold
        self.has_toggled = False

    def _on_step(self) -> bool:
        if not self.has_toggled and self.model.num_timesteps >= self.threshold:
            print(f"Enabling static blocks at timestep: {self.model.num_timesteps}")
            self.training_env.env_method("setStaticBlocks", True)
            self.has_toggled = True
        return True
    
class ToggleMovingBlocksCallback(BaseCallback):
    def __init__(self, threshold: int, verbose=0):
        super(ToggleMovingBlocksCallback, self).__init__(verbose)
        self.threshold = threshold
        self.has_toggled = False

    def _on_step(self) -> bool:
        if not self.has_toggled and self.model.num_timesteps >= self.threshold:
            print(f"Enabling moving blocks at timestep: {self.model.num_timesteps}")
            self.training_env.env_method("setMovingBlocks", True)
            self.has_toggled = True
        return True
    
class SaveModelCallback(BaseCallback):
    def __init__(self, thresholds, save_paths, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        self.thresholds = thresholds  # List of timesteps at which to save the model
        self.save_paths = save_paths  # List of corresponding file names
        self.saved = [False] * len(thresholds)

    def _on_step(self) -> bool:
        for i, threshold in enumerate(self.thresholds):
            if not self.saved[i] and self.model.num_timesteps >= threshold:
                self.model.save(self.save_paths[i])
                print(f"Saved model at {threshold} timesteps to {self.save_paths[i]}")
                self.saved[i] = True
        return True
    
class ToggleDonutObstaclesCallback(BaseCallback):
    def __init__(self, threshold: int, verbose=0):
        super(ToggleDonutObstaclesCallback, self).__init__(verbose)
        self.threshold = threshold
        self.has_toggled = False

    def _on_step(self) -> bool:
        if not self.has_toggled and self.model.num_timesteps >= self.threshold:
            print(f"Enabling donut obstacles at timestep: {self.model.num_timesteps}")
            self.training_env.env_method("setDonutObstacles", True)
            self.has_toggled = True
        return True
