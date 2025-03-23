from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gymnasium import spaces
import torch
import torch.nn as nn

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, **kwargs):
        super().__init__(observation_space, features_dim)
        
        # Calculate input channels based on observation space
        input_channels = observation_space.shape[0]  # For [H, W, C] format
        
        self.cnn = nn.Sequential(
            # PyTorch expects [N, C, H, W] but we have [N, H, W, C], so we'll handle this in forward()
            nn.LayerNorm([input_channels, 100, 156]),
            
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0, bias=False),
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

        self.linear = nn.Sequential(
            nn.Linear(9216, features_dim, bias=False),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(**kwargs),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))