import cv2
import os
import argparse
import numpy as np
import vizdoom  # type: ignore
import json

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from typing import Callable, List, Any, Dict, TypeAlias
import gymnasium as gym

from envs import DoomEnvSP
from policy import (
    CustomCNN
)
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from callbacks import RewardAverageCallback, VideoEvalCallback

# Type alias for clarity
Frame: TypeAlias = np.ndarray


def save_run_metadata(output_dir: str, map_name: str, difficulty: int | None, env_params: dict) -> None:
    """Save training run metadata next to model artifacts.

    Writes a generic metadata file as well as common model-named variants so
    test scripts can discover metadata from only a model path.
    """
    metadata = {
        "map": map_name,
        "difficulty": difficulty,
        "env": {
            "config_path": env_params.get("config_path"),
            "n_frames": env_params.get("n_frames"),
            "frame_skip": env_params.get("frame_skip"),
            "n_actions_history": env_params.get("n_actions_history"),
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    # Generic metadata file for the directory
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

def create_env(
    config_path: str = "config/test.cfg",
    map_name: str | None = None,
    render: bool = False,
    n_frames: int = 4,
    difficulty: int | None = None,
    **kwargs,
) -> DoomEnvSP:
    """Create and initialize a VizDoom environment."""
    game = vizdoom.DoomGame()
    game.load_config(config_path)
    
    if map_name:
        game.set_doom_map(map_name)
    
    if difficulty is not None:
        game.set_doom_skill(int(difficulty))
    
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

def create_vec_env(map_name, n_envs: int = 1, is_eval: bool = False, log_dir: str = "logs", **kwargs) -> VecEnv:
    """Create a vectorized environment with multiple DoomEnv instances, potentially using different maps."""
    if is_eval:
        eval_kwargs = kwargs.copy()
        eval_kwargs["map_name"] = map_name
        eval_kwargs["render"] = True  # Always render the eval environment
        monitor_path = os.path.join(log_dir, "monitor", "eval")
        os.makedirs(monitor_path, exist_ok=True)
        return DummyVecEnv([lambda: Monitor(create_env(**eval_kwargs), monitor_path)] * n_envs)
    
    def make_env(env_kwargs, rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env = create_env(**env_kwargs)
            monitor_path = os.path.join(log_dir, "monitor", "train", f"{rank}")
            os.makedirs(monitor_path, exist_ok=True)
            return Monitor(env, monitor_path)
        return _init

    env_creators: List[Callable[[], gym.Env]] = []
    for i in range(n_envs):
        env_kwargs = kwargs.copy()
        env_kwargs["map_name"] = map_name
        
        # Only render the first environment for viewing
        env_kwargs["render"] = (i == 0)
        
        env_creators.append(make_env(env_kwargs, i))
    return SubprocVecEnv(env_creators)  # type: ignore[arg-type]

def frame_processor(frame: Frame) -> Frame:
    # frame.shape is (240, 320, 3) which we resize to (120, 160)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return frame

def create_agent(env, tensorboard_log: str = "logs/tensorboard", **kwargs) -> PPO:
    """Create a PPO agent with a CNN policy."""

    # SubprocVecEnv/DummyVecEnv expose a gymnasium Discrete action space
    action_space_dim = env.action_space.n
    kwargs['policy_kwargs']['features_extractor_kwargs']['action_space_dim'] = action_space_dim

    return PPO(
        policy=MultiInputActorCriticPolicy,  # Use our custom policy with action history
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
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=666,
        **kwargs
    )


def solve_env(env_args, n_envs, agent_args, map_name, log_dir: str, ckpt_dir: str):
    """Train agent on multiple maps in parallel."""
    
    training_env = create_vec_env(map_name=map_name, n_envs=n_envs, log_dir=log_dir, **env_args)
    eval_env = create_vec_env(map_name=map_name, n_envs=1, is_eval=True, log_dir=log_dir, **env_args)
    
    tb_log_dir = os.path.join(log_dir, "tensorboard")
    eval_log_path = os.path.join(log_dir, "evaluations", "multi_map")
    best_model_dir = os.path.join(log_dir, "models", "multi_map")
    os.makedirs(eval_log_path, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # Persist metadata alongside checkpoints and best/final models to avoid
    # train/test mismatch in evaluation scripts.
    metadata_env = {
        "config_path": env_args.get("config_path"),
        "n_frames": env_args.get("n_frames"),
        "frame_skip": env_args.get("frame_skip"),
        "n_actions_history": env_args.get("n_actions_history"),
    }
    difficulty_value = env_args.get("difficulty")
    save_run_metadata(best_model_dir, map_name=map_name, difficulty=difficulty_value, env_params=metadata_env)
    save_run_metadata(ckpt_dir, map_name=map_name, difficulty=difficulty_value, env_params=metadata_env)
    
    agent = create_agent(training_env, tensorboard_log=tb_log_dir, **agent_args)
    
    eval_env = VecTransposeImage(eval_env)
    
    print(f"Training on {map_name} across {n_envs} environments")
    print("Rendering only the first training environment")
    print("Rendering the evaluation environment")

    evaluation_callback = EvalCallback(
        eval_env,
        n_eval_episodes=3,
        eval_freq=5000,
        log_path=eval_log_path,
        best_model_save_path=best_model_dir,
        deterministic=False,
        render=False,
    )

    video_callback = VideoEvalCallback(
        eval_env=eval_env,
        gif_dir=os.path.join(log_dir, "figures", "eval"),
        n_eval_episodes=3,
        eval_freq=5000,
        deterministic=False,
        fps=9,
        verbose=1,
        video_format="mp4",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=ckpt_dir,
        name_prefix="ppo_doom_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    avg_reward_callback = RewardAverageCallback(verbose=1)
    callbacks = [evaluation_callback, video_callback, checkpoint_callback, avg_reward_callback]
    
    agent.learn(
        total_timesteps=10_000_000, # GameNGen paper uses 50M total steps, which means 50M // 8 = 6.25M steps per env
        tb_log_name="ppo_multi_map",
        callback=callbacks
    )
    
    final_model_path = os.path.join(best_model_dir, "final_model")
    agent.save(final_model_path)
    
    training_env.close()
    eval_env.close()

if __name__ == "__main__":
    # CLI arguments
    parser = argparse.ArgumentParser(description="Train PPO agent for VizDoom (single-process maps)")
    parser.add_argument("--difficulty", type=int, choices=[1, 2, 3, 4, 5], help="Doom skill level (1-5)")
    parser.add_argument("--log-dir", type=str, default="logs", help="Base directory for logs and models")
    parser.add_argument("--map", type=str, default="E1M1", help="Doom map name, e.g., E1M2")
    parser.add_argument("--config", type=str, default="config/test.cfg", help="VizDoom config path")
    parser.add_argument("--n-frames", type=int, default=1, help="Number of frames to stack for observations")
    parser.add_argument("--n-training-envs", type=int, default=8, help="Number of training environments to run in parallel")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights and Biases for logging") # Not implemented yet
    args = parser.parse_args()


    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        print("Failed to set OpenCV threads, continuing...")

    log_dir = args.log_dir
    ckpt_dir = os.path.join(log_dir, "models", "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    doom_map = args.map

    env_args = {
        "frame_skip": 4,
        "frame_processor": frame_processor,
        "automap_processor": frame_processor,
        "config_path": args.config,
        "n_frames": args.n_frames,
        "n_actions_history": 32,  # track last 32 actions for actor-critic models, doesn't go through CNN
        "difficulty": args.difficulty,
    }
    
    n_envs = args.n_training_envs
    agent_args: Dict[str, Any] = {"device": "auto"}
    
    # Add policy_kwargs to specify our CustomCNN as the features extractor
    agent_args['policy_kwargs'] = {
        'features_extractor_class': CustomCNN,
        # action_space_dim is set in create_agent
        'features_extractor_kwargs': {'features_dim': 512, 'action_space_dim': None}
    }

    solve_env(env_args, n_envs, agent_args=agent_args, map_name=doom_map, log_dir=log_dir, ckpt_dir=ckpt_dir)
