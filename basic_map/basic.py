import numpy as np
import vizdoom
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.monitor import Monitor
import typing as t
import cv2
# Type alias for clarity
Frame = np.ndarray

class DoomEnv(gym.Env):
    """Wrapper environment following Gymnasium interface for a VizDoom game instance."""
    
    def __init__(self, game: vizdoom.DoomGame, frame_processor: t.Callable, frame_skip: int = 4):
        """Initialize the Doom environment with a game instance, frame processor, and frame skip."""
        super().__init__()
        
        # Define action space: Discrete space based on number of available buttons
        self.action_space = spaces.Discrete(game.get_available_buttons_size())
        
        # Define observation space: Processed screen buffer dimensions
        h, w, c = game.get_screen_height(), game.get_screen_width(), game.get_screen_channels()
        processed_shape = frame_processor(np.zeros((h, w, c))).shape
        self.observation_space = spaces.Box(low=0, high=255, shape=processed_shape, dtype=np.uint8)
        
        # Store instance variables
        self.game = game
        self.possible_actions = np.eye(self.action_space.n).tolist()  # One-hot encoded actions for VizDoom
        self.frame_skip = frame_skip
        self.frame_processor = frame_processor
        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.state = self.empty_frame

    def step(self, action: int) -> t.Tuple[Frame, float, bool, bool, dict]:
        """Apply an action to the environment and return the resulting state."""
        reward = self.game.make_action(self.possible_actions[action], self.frame_skip)
        terminated = self.game.is_episode_finished()
        truncated = False  # VizDoom handles termination; no truncation needed here
        self.state = self._get_frame(terminated)
        return self.state, reward, terminated, truncated, {}

    def reset(self, *, seed: t.Optional[int] = None, options: t.Optional[dict] = None) -> t.Tuple[Frame, dict]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        self.game.new_episode()
        self.state = self._get_frame()
        return self.state, {}

    def close(self) -> None:
        """Close the VizDoom game instance."""
        self.game.close()

    def render(self, mode: str = "human") -> None:
        """Render method (not implemented as VizDoom handles its own rendering)."""
        pass

    def _get_frame(self, done: bool = False) -> Frame:
        """Get the current frame, or an empty frame if the episode is done."""
        return self.frame_processor(self.game.get_state().screen_buffer) if not done else self.empty_frame

    def seed(self, seed: t.Optional[int] = None) -> t.List[t.Optional[int]]:
        """Set the seed for VizDoom's random number generator."""
        if seed is not None:
            self.game.set_seed(seed)
        return [seed]

def create_env(**kwargs) -> DoomEnv:
    """Create and initialize a VizDoom environment."""
    game = vizdoom.DoomGame()
    game.load_config("scenarios/basic.cfg")
    game.init()
    return DoomEnv(game, **kwargs)

def create_vec_env(n_envs=1, is_eval=False, **kwargs) -> DummyVecEnv:
    """Create a vectorized environment with multiple DoomEnv instances."""
    if is_eval:
        return DummyVecEnv([lambda: Monitor(create_env(**kwargs), f"logs/monitor/eval")] * n_envs)
    return DummyVecEnv([lambda: create_env(**kwargs) for _ in range(n_envs)])

def frame_processor(frame: Frame) -> Frame:
    # Removed print statement to avoid clutter during training
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return frame


def create_agent(env, **kwargs) -> PPO:
    """Create a PPO agent with a CNN policy."""
    return PPO(
        policy=ActorCriticCnnPolicy,
        env=env,
        n_steps=2048,
        batch_size=32,
        learning_rate=2.5e-4,  # Standard learning rate for PPO
        gamma=0.99,            # Discount factor
        gae_lambda=0.95,       # GAE lambda parameter
        clip_range=0.2,        # PPO clipping parameter
        ent_coef=0.01,         # Entropy coefficient to encourage exploration
        vf_coef=0.5,           # Value function coefficient
        max_grad_norm=0.5,     # Gradient clipping for numerical stability
        tensorboard_log="logs/tensorboard",
        verbose=1,             # Set to 1 to see training progress
        seed=42,               # Fixed seed for reproducibility
        **kwargs
    )

if __name__ == "__main__":
    # Environment configuration
    env_args = {
        "frame_skip": 4,          # Number of frames to repeat the last action
        "frame_processor": frame_processor  # Frame processing function
    }

    # Create training and evaluation environments
    # Using 4 parallel environments for better training efficiency
    n_envs = 4
    training_env = create_vec_env(n_envs=n_envs, **env_args)
    eval_env = create_vec_env(n_envs=1, is_eval=True, **env_args)
    # Initialize the agent
    agent = create_agent(training_env)

    # Set up evaluation callback to monitor progress and save the best model
    evaluation_callback = EvalCallback(
        eval_env,
        n_eval_episodes=10,
        eval_freq=5000, # Evaluate every 5000 steps for more frequent feedback
        log_path="logs/evaluations/basic",
        best_model_save_path="logs/models/basic",
        deterministic=True,
        render=False
    )

    # Train the agent with more timesteps for better learning
    agent.learn(
        total_timesteps=200000,    # Increased from 40000 for more thorough learning
        tb_log_name="ppo_basic",
        callback=evaluation_callback
    )

    # Save the final model
    agent.save("logs/models/basic/final_model")

    # Clean up
    training_env.close()
    eval_env.close()
