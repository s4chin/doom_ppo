import torch
import torch.nn as nn
import random
from gymnasium import spaces
from typing import cast

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DQN_CNN(nn.Module):
    def __init__(self, input_shape=(3, 120, 160)):
        super(DQN_CNN, self).__init__()
        self.conv = nn.Sequential(
            # nn.LayerNorm(input_shape),
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self.conv(torch.zeros(1, *input_shape)).numel()
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
        )

    def forward(self, x):
        if random.random() < 0.1:
            print(f"{x.min()=}, {x.max()=}, {x.mean()=}, {x.shape=}")
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512, action_space_dim: int = 20, **kwargs):
        super().__init__(observation_space, features_dim)
        
        # Get dimensions from screen and automap spaces
        screen_channels, screen_h, screen_w = observation_space['screen'].shape
        automap_channels, automap_h, automap_w = observation_space['automap'].shape

        self.screen_shape = (screen_channels, screen_h, screen_w)
        self.automap_shape = (automap_channels, automap_h, automap_w)

        self.screen_cnn = DQN_CNN(input_shape=self.screen_shape)
        self.automap_cnn = DQN_CNN(input_shape=self.automap_shape)

        self.action_history_embedding = nn.Embedding(1 + action_space_dim, 8) # 1 for when there is no action (initial state)
        self.action_history_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(**kwargs),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(**kwargs),
        )

        screen_output_size = 512
        automap_output_size = 512
        action_history_output_size = 32
        
        # Create a combined linear layer
        self.linear = nn.Sequential(
            nn.Linear(screen_output_size + automap_output_size + action_history_output_size, features_dim, bias=False),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(**kwargs),

            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: spaces.Dict) -> torch.Tensor:
        # if random.random() < 0.1:
        #     save_as_image(observations)
        #     print(f"{observations['action_history']=}")
        # Process screen and automap separately

        assert observations['screen'].shape[1:] == self.screen_shape, f"Unexpected shape for screen: {observations['screen'].shape=}"
        assert observations['automap'].shape[1:] == self.automap_shape, f"Unexpected shape for automap: {observations['automap'].shape=}"

        screen_features = self.screen_cnn(observations['screen'])
        automap_features = self.automap_cnn(observations['automap'])
        action_history_features = self.action_history_embedding(observations['action_history'].long() + 1)
        action_history_features = self.action_history_net(action_history_features)
        
        if random.random() < 0.1:
            print(f"{screen_features.shape=}, {automap_features.shape=}, {action_history_features.shape=}")

        # Concatenate features and pass through linear layer
        combined_features = torch.cat([screen_features, automap_features, action_history_features], dim=1)
        output = self.linear(combined_features)
        return output


# Unused for now, making sure screen and automap are processed correctly
def save_as_image(observations: spaces.Dict):
    # Convert automap to PIL image and save it
    from PIL import Image
    import numpy as np
    for i in range(0, 12, 3):
        automap_image = Image.fromarray((observations['automap'].numpy()[0][i:i+3, :, :].transpose(1, 2, 0) * 255.).astype(np.uint8))
        automap_image.save(f'automap_{i}.png')

    for i in range(0, 12, 3):
        screen_image = Image.fromarray((observations['screen'].numpy()[0][i:i+3, :, :].transpose(1, 2, 0) * 255.).astype(np.uint8))
        screen_image.save(f'screen_{i}.png')
