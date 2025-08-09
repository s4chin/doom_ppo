from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from gymnasium import spaces
import torch
import torch.nn as nn

class DQN_CNN(nn.Module):
    def __init__(self, input_shape=(3, 120, 160)):
        super(DQN_CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.LayerNorm(input_shape),
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
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, action_space_dim: int = 20, **kwargs):
        super().__init__(observation_space, features_dim)
        
        # Get dimensions from screen and automap spaces
        screen_channels = observation_space['screen'].shape[0]
        automap_channels = observation_space['automap'].shape[0]
        
        # # Create separate CNNs for screen and automap
        # # Screen CNN
        # self.screen_cnn = nn.Sequential(
        #     # PyTorch expects [N, C, H, W] but we have [N, H, W, C], so we'll handle this in forward()
        #     nn.LayerNorm([screen_channels, 120, 160]),
        #     nn.Conv2d(screen_channels, 16, kernel_size=8, stride=4, padding=2, bias=False),
        #     nn.LayerNorm([16, 30, 40]),
        #     nn.LeakyReLU(**kwargs),
            
        #     # nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
        #     # nn.LayerNorm([32, 30, 40]),
        #     # nn.LeakyReLU(**kwargs),
            
        #     nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.LayerNorm([32, 15, 20]),
        #     nn.LeakyReLU(**kwargs),
            
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.LayerNorm([64, 8, 10]),
        #     nn.LeakyReLU(**kwargs),
            
        #     nn.Flatten(),
        # )
        
        # self.automap_cnn = nn.Sequential(
        #     nn.LayerNorm([automap_channels, 120, 160]),
        #     nn.Conv2d(automap_channels, 16, kernel_size=8, stride=4, padding=2, bias=False),
        #     nn.LayerNorm([16, 30, 40]),
        #     nn.LeakyReLU(**kwargs),
            
        #     # nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
        #     # nn.LayerNorm([32, 30, 40]),
        #     # nn.LeakyReLU(**kwargs),
            
        #     nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.LayerNorm([32, 15, 20]),
        #     nn.LeakyReLU(**kwargs),
            
        #     nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.LayerNorm([32, 8, 10]),
        #     nn.LeakyReLU(**kwargs),
            
        #     nn.Flatten(),
        # )

        self.screen_cnn = DQN_CNN()
        self.automap_cnn = DQN_CNN()

        self.action_history_embedding = nn.Embedding(1 + action_space_dim, 8) # 1 for no action (initial state)
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
        # screen_output_size = 64 * 8 * 10
        # automap_output_size = 32 * 8 * 10

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
            nn.LeakyReLU(**kwargs),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # if random.random() < 0.1:
        #     save_as_image(observations)
        #     print(f"{observations['action_history']=}")
        # Process screen and automap separately

        assert observations['screen'].shape[1] == 3, f"Unexpected shape for screen: {observations['screen'].shape=}"
        assert observations['automap'].shape[1] == 3, f"Unexpected shape for automap: {observations['automap'].shape=}"

        screen_features = self.screen_cnn(observations['screen'])
        automap_features = self.automap_cnn(observations['automap'])
        action_history_features = self.action_history_embedding(observations['action_history'].long() + 1)
        action_history_features = self.action_history_net(action_history_features)
        
        # Concatenate features and pass through linear layer
        combined_features = torch.cat([screen_features, automap_features, action_history_features], dim=1)
        output = self.linear(combined_features)
        return output


# Unused for now, making sure screen and automap are processed correctly
def save_as_image(observations: torch.Tensor):
    # Convert automap to PIL image and save it
    from PIL import Image
    import numpy as np
    for i in range(0, 12, 3):
        automap_image = Image.fromarray((observations['automap'].numpy()[0][i:i+3, :, :].transpose(1, 2, 0) * 255.).astype(np.uint8))
        automap_image.save(f'automap_{i}.png')

    for i in range(0, 12, 3):
        screen_image = Image.fromarray((observations['screen'].numpy()[0][i:i+3, :, :].transpose(1, 2, 0) * 255.).astype(np.uint8))
        screen_image.save(f'screen_{i}.png')