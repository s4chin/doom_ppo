import cv2
import numpy as np
import vizdoom
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from envs import DoomEnvSP
from policy import CustomCNN

# Type alias for clarity
Frame = np.ndarray


def create_env(config_path="config/test.cfg", map_name=None, render=False, n_frames=4, **kwargs) -> DoomEnvSP:
    """Create and initialize a VizDoom environment."""
    game = vizdoom.DoomGame()
    game.load_config(config_path)
    
    # Set the map if provided
    if map_name:
        game.set_doom_map(map_name)
    
    # Set rendering mode
    if not render:
        game.set_window_visible(False)
    else:
        game.set_window_visible(True)
    
    # Enable automap
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vizdoom.AutomapMode.OBJECTS)
    game.set_automap_rotate(True)
    game.set_automap_render_textures(True)
        
    game.init()
    return DoomEnvSP(game, n_frames=n_frames, **kwargs)

def create_vec_env(n_envs=1, is_eval=False, maps=None, **kwargs) -> DummyVecEnv:
    """Create a vectorized environment with multiple DoomEnv instances, potentially using different maps."""
    if is_eval:
        # For evaluation, always use E1M1 and render it
        eval_kwargs = kwargs.copy()
        eval_kwargs["map_name"] = "E1M1"
        eval_kwargs["render"] = True  # Always render the eval environment
        return DummyVecEnv([lambda: Monitor(create_env(**eval_kwargs), f"logs/monitor/eval")] * n_envs)
    
    if maps and len(maps) > 0:
        # Create environments with different maps for training
        env_creators = []
        for i in range(n_envs):
            # Select map in a round-robin fashion
            map_name = maps[i % len(maps)]
            env_kwargs = kwargs.copy()
            env_kwargs["map_name"] = map_name
            
            # Only render the first environment
            env_kwargs["render"] = (i == 0)
            
            env_creators.append(lambda env_kwargs=env_kwargs: create_env(**env_kwargs))
        return DummyVecEnv(env_creators)
    else:
        # Default behavior if no maps are specified
        return DummyVecEnv([lambda i=i: create_env(render=(i==0), **kwargs) for i in range(n_envs)])

def frame_processor(frame: Frame) -> Frame:
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    frame = frame[10:-10, 2:-2, :]
    return frame

def automap_processor(automap: Frame) -> Frame:
    automap = cv2.resize(automap, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    automap = automap[10:-10, 2:-2, :]
    return automap

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

class RewardAverageCallback(BaseCallback):
    """
    Custom callback for logging the running average of episode rewards to tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.window_size = 10  # Number of episodes to average over

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                if len(self.episode_rewards) > self.window_size:
                    self.episode_rewards.pop(0)
                
                if len(self.episode_rewards) > 0:
                    avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                    self.logger.record('rollout/reward_running_avg', avg_reward)
        
        return True

def solve_env(env_args, n_envs, agent_args, maps=None):
    """Train agent on multiple maps in parallel."""
    
    # Create training environment with multiple maps
    training_env = create_vec_env(n_envs=n_envs, maps=maps, **env_args)
    
    # Create evaluation environment (always using E1M1)
    eval_env = create_vec_env(n_envs=1, is_eval=True, **env_args)
    
    agent = create_agent(training_env, **agent_args)
    
    print(f"Training on {len(maps) if maps else 1} maps across {n_envs} environments")
    print(f"Rendering only the first training environment")
    print(f"Rendering the evaluation environment")
    if maps:
        print(f"Maps: {', '.join(maps)}")
    print(f"Evaluating on: E1M1")
    
    evaluation_callback = EvalCallback(
        eval_env,
        n_eval_episodes=3,
        eval_freq=5000,
        log_path="logs/evaluations/multi_map",
        best_model_save_path="logs/models/multi_map",
        deterministic=True,
        render=False
    )
    
    reward_callback = RewardAverageCallback()
    
    callbacks = [evaluation_callback, reward_callback]
    
    agent.learn(
        total_timesteps=1000000,
        tb_log_name="ppo_multi_map",
        callback=callbacks
    )
    
    agent.save("logs/models/multi_map/final_model")
    
    training_env.close()
    eval_env.close()

if __name__ == "__main__":
    # Environment configuration
    env_args = {
        "frame_skip": 4,
        "frame_processor": frame_processor,
        "automap_processor": automap_processor,
        "config_path": "config/test.cfg",
        "n_frames": 4  # Add n_frames parameter
    }

    doom_maps = [f"E1M{i}" for i in range(1, 10)]  # E1M1 through E1M9
    
    # Using parallel environments with different maps
    n_envs = 1
    agent_args = {}
    agent_args['policy_kwargs'] = {'features_extractor_class': CustomCNN}

    solve_env(env_args, n_envs, agent_args=agent_args, maps=doom_maps)
