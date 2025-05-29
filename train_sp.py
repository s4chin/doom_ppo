import cv2
import numpy as np
import vizdoom

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from callbacks import RewardAverageCallback
from envs import DoomEnvSP
from policy import (
    CustomCNN
)
from stable_baselines3.common.policies import ActorCriticCnnPolicy

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

def create_vec_env(map, n_envs=1, is_eval=False, **kwargs) -> DummyVecEnv:
    """Create a vectorized environment with multiple DoomEnv instances, potentially using different maps."""
    if is_eval:
        eval_kwargs = kwargs.copy()
        eval_kwargs["map_name"] = map
        eval_kwargs["render"] = True  # Always render the eval environment
        return DummyVecEnv([lambda: Monitor(create_env(**eval_kwargs), f"logs/monitor/eval")] * n_envs)
    
    env_creators = []
    for i in range(n_envs):
        env_kwargs = kwargs.copy()
        env_kwargs["map_name"] = map
        
        # Only render the first environment for viewing
        env_kwargs["render"] = (i == 0)
        
        env_creators.append(lambda env_kwargs=env_kwargs: create_env(**env_kwargs))
    return DummyVecEnv(env_creators)

def frame_processor(frame: Frame) -> Frame:
    # frame.shape is (240, 320, 3) which we resize to (120, 160)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return frame

def create_agent(env, **kwargs) -> PPO:
    """Create a PPO agent with a CNN policy."""
    return PPO(
        policy=ActorCriticCnnPolicy,  # Use our custom policy with action history
        env=env,
        n_steps=512,
        batch_size=64,
        learning_rate=1e-4,  # From GameNGen paper
        gamma=0.99,          # From GameNGen paper
        gae_lambda=0.95,     # GAE lambda parameter
        clip_range=0.2,      # PPO clipping parameter
        ent_coef=0.1,        # From GameNGen paper - higher for more exploration
        vf_coef=0.5,         # Value function coefficient
        max_grad_norm=0.5,   # Gradient clipping for numerical stability
        tensorboard_log="logs/tensorboard",
        verbose=1,
        seed=666,
        **kwargs
    )


def solve_env(env_args, n_envs, agent_args, map):
    """Train agent on multiple maps in parallel."""
    
    training_env = create_vec_env(map=map, n_envs=n_envs, **env_args)
    
    # Use only one eval env
    eval_env = create_vec_env(map=map, n_envs=1, is_eval=True, **env_args)
    
    agent = create_agent(training_env, **agent_args)
    
    print(f"Training on {map} across {n_envs} environments")
    print(f"Rendering only the first training environment")
    print(f"Rendering the evaluation environment")
    
    evaluation_callback = EvalCallback(
        eval_env,
        n_eval_episodes=3,
        eval_freq=5000,
        log_path="logs/evaluations/multi_map",
        best_model_save_path="logs/models/multi_map",
        deterministic=True,
        render=False,
    )
    
    # reward_callback = RewardAverageCallback()
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="logs/models/checkpoints/",
        name_prefix="ppo_doom_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    callbacks = [evaluation_callback, checkpoint_callback]
    
    agent.learn(
        total_timesteps=10_000_000,
        tb_log_name="ppo_multi_map",
        callback=callbacks
    )
    
    agent.save("logs/models/multi_map/final_model")
    
    training_env.close()
    eval_env.close()

if __name__ == "__main__":
    env_args = {
        "frame_skip": 4,
        "frame_processor": frame_processor,
        "automap_processor": frame_processor,
        "config_path": "config/test.cfg",
        "n_frames": 1, # stacking frames for CNN input (may not be useful)
        "n_actions_history": 32  # track last 32 actions for actor-critic models, doesn't go through CNN
    }

    doom_map = "E1M2"
    
    n_envs = 8 # From GameNGen paper
    agent_args = {}
    
    # Add policy_kwargs to specify our CustomCNN as the features extractor
    agent_args['policy_kwargs'] = {
        'features_extractor_class': CustomCNN,
        'features_extractor_kwargs': {'features_dim': 512}
    }

    solve_env(env_args, n_envs, agent_args=agent_args, map=doom_map)
