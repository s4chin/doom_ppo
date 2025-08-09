import imageio  # type: ignore[import]
import os
import json
from pathlib import Path
from train_sp import create_vec_env, frame_processor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage

def _discover_metadata_path(model_path: str) -> str | None:
    """Given a model file path, locate the accompanying metadata file.
        model_metadata.json in the model's directory
    """
    p = Path(model_path)
    metadata_path = os.path.join(p.parent.as_posix(), "model_metadata.json")
    if os.path.exists(metadata_path):
        print(f"Found metadata at {metadata_path}")
        return metadata_path
    return None


def make_gif(model_path, file_path):
    agent = PPO.load(model_path)
    print(f"{agent.policy.observation_space=}")

    metadata_path = _discover_metadata_path(model_path)
    if metadata_path is None:
        raise FileNotFoundError(
            f"Could not find metadata for model at {model_path}. "
            "Expected '<model>.meta.json' or 'model_metadata.json' in the same directory."
        )

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    doom_map = meta.get("map", "E1M1")
    difficulty = meta.get("difficulty")
    env_meta = meta.get("env", {})

    env_args = {
        "frame_skip": env_meta.get("frame_skip", 4),
        "frame_processor": frame_processor,
        "automap_processor": frame_processor,
        "config_path": env_meta.get("config_path", "config/test.cfg"),
        "n_frames": env_meta.get("n_frames", 1),
        "n_actions_history": env_meta.get("n_actions_history", 32),
        # Keep inference with the same frame_skip for policy decisions, but capture all frames for GIF
        "capture_intermediate_frames": True,
        "difficulty": difficulty,
    }

    env = create_vec_env(n_envs=1, map_name=doom_map, is_eval=True, **env_args)
    # Wrap with VecTransposeImage to match the training setup
    env = VecTransposeImage(env)
    env.seed(0)
    
    print(f"Environment observation space: {env.observation_space}")
    print(f"Evaluating on map={doom_map}, difficulty={difficulty}")
    
    images = []
    
    for i in range(1):
        obs = env.reset()
        
        done = False
        step_count = 0
        
        while not done:
            action, _ = agent.predict(obs, deterministic=False)
            print(f"Action: {action}")
            obs, reward, dones, info = env.step(action)
            done = dones[0]
            # Prefer captured no-skip frames if available
            try:
                frame_list = info[0].get("captured_frames") if isinstance(info, (list, tuple)) else None
            except Exception:
                frame_list = None
            if frame_list:
                images.extend(frame_list)
            else:
                # Fallback to current state frame if not capturing
                state = env.envs[0].env.game.get_state()
                if state is not None:
                    images.append(state.screen_buffer)
            
            step_count += 1
            if step_count > 5000:
                print("Episode too long, breaking...")
                break
    
    print(f"Saving GIF with {len(images)} frames")
    imageio.mimsave(file_path, images, fps=35)
    
    env.close()
    return agent

os.makedirs('figures', exist_ok=True)

agent = make_gif('logs/models/checkpoints/ppo_doom_model_8000_steps.zip', 'figures/sp_agent.gif')
print("GIF created successfully!")
