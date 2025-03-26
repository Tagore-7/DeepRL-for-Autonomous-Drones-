import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
import gymnasium as gym


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        self.rgb_features_dim = 256
        self.state_features_dim = 64
        self.lidar_features_dim = 64

        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0

        for key, _ in observation_space.items():
            if key == "state":
                state_space = observation_space["state"]
                flattened_state_size = get_flattened_obs_dim(state_space)
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(flattened_state_size, self.state_features_dim),
                    nn.ReLU(),
                )
                total_concat_size += self.state_features_dim
            elif key == "lidar":
                state_space = observation_space["lidar"]
                flattened_state_size = get_flattened_obs_dim(state_space)
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(flattened_state_size, self.lidar_features_dim),
                    nn.ReLU(),
                )
                total_concat_size += self.lidar_features_dim
            elif key == "rgb":
                # Get the Box space for the RGB
                rgb_space = observation_space["rgb"]
                # CNN for RGB images
                extractors["rgb"] = NatureCNN(
                    rgb_space,
                    features_dim=self.rgb_features_dim,
                    normalized_image=False,
                )
                total_concat_size += self.rgb_features_dim

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        return torch.cat(encoded_tensor_list, dim=1)
