from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from gymnasium import spaces
import torch
import torch.nn as nn

from envs import possible_actions


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, **kwargs):
        super().__init__(observation_space, features_dim)
        
        # Check if observation space is Dict (contains both screen and automap)
        if isinstance(observation_space, spaces.Dict):
            print(f"{observation_space.keys()}")
            # Get dimensions from screen and automap spaces
            screen_channels = observation_space['screen'].shape[0]
            automap_channels = observation_space['automap'].shape[0]
            
            # Create separate CNNs for screen and automap
            # Screen CNN
            self.screen_cnn = nn.Sequential(
                # PyTorch expects [N, C, H, W] but we have [N, H, W, C], so we'll handle this in forward()
                nn.LayerNorm([screen_channels, 100, 156]),
                
                nn.Conv2d(screen_channels, 32, kernel_size=8, stride=4, padding=0, bias=False),
                nn.LayerNorm([32, 24, 38]),
                nn.LeakyReLU(**kwargs),
                
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, bias=False),
                nn.LayerNorm([64, 11, 18]),
                nn.LeakyReLU(**kwargs),
                
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
                nn.LayerNorm([64, 9, 16]),
                nn.LeakyReLU(**kwargs),
                
                nn.Flatten(),
            )
            
            self.automap_cnn = nn.Sequential(
                nn.LayerNorm([automap_channels, 100, 156]),
                
                nn.Conv2d(automap_channels, 16, kernel_size=8, stride=4, padding=0, bias=False),
                nn.LayerNorm([16, 24, 38]),
                nn.LeakyReLU(**kwargs),
                
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=False),
                nn.LayerNorm([32, 11, 18]),
                nn.LeakyReLU(**kwargs),
                
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False),
                nn.LayerNorm([32, 9, 16]),
                nn.LeakyReLU(**kwargs),
                
                nn.Flatten(),
            )

            self.action_history_embedding = nn.Embedding(32, 8)
            self.action_history_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 8, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(**kwargs),
                nn.Linear(128, 32),
                nn.LayerNorm(32),
                nn.LeakyReLU(**kwargs),
            )
            
            # Calculate output sizes for both CNNs
            screen_output_size = 9216  # 64 * 9 * 16
            automap_output_size = 4608  # 32 * 9 * 16
            action_history_output_size = 32
            
            # Create a combined linear layer
            self.linear = nn.Sequential(
                nn.Linear(screen_output_size + automap_output_size + action_history_output_size, features_dim, bias=False),
                nn.LayerNorm(features_dim),
                nn.LeakyReLU(**kwargs),
            )
        else:
            raise ValueError('Observation space must be a dict containing both screen and automap')

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Check if observations is a dict (contains both screen and automap)
        if isinstance(observations, dict):
            # Process screen and automap separately
            screen_features = self.screen_cnn(observations['screen'])
            automap_features = self.automap_cnn(observations['automap'])
            action_history_features = self.action_history_embedding(observations['action_history'].long())
            action_history_features = self.action_history_net(action_history_features)

            # print(f"{screen_features.shape=}, {automap_features.shape=}, {action_history_features.shape=}")
            
            # Concatenate features and pass through linear layer
            combined_features = torch.cat([screen_features, automap_features, action_history_features], dim=1)
            return self.linear(combined_features)
        else:
            print('Not using automap')
            return self.linear(self.cnn(observations))


# Unused for now, making sure screen and automap are processed correctly
def save_as_image(observations: torch.Tensor):
    # Convert automap to PIL image and save it
    from PIL import Image
    import numpy as np
    automap_image = Image.fromarray((observations['automap'].numpy()[0][:3, :, :].transpose(1, 2, 0) * 255.).astype(np.uint8))
    automap_image.save('automap.png')

    automap_image = Image.fromarray((observations['screen'].numpy()[0][:3, :, :].transpose(1, 2, 0) * 255.).astype(np.uint8))
    automap_image.save('screen.png')