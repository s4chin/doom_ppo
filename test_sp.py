import imageio
import os
import numpy as np
from train_sp import create_vec_env, frame_processor, create_agent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage

def make_gif(model_path, file_path):
    agent = PPO.load(model_path)
    print(f"{agent.policy.observation_space=}")
    
    env_args = {
        "frame_skip": 4,
        "frame_processor": frame_processor,
        "automap_processor": frame_processor,
        "config_path": "config/test.cfg",
        "n_frames": 1, # stacking frames for CNN input
        "n_actions_history": 32  # track last 32 actions for actor-critic models, doesn't go through CNN
    }
    
    env = create_vec_env(n_envs=1, map="E1M2", **env_args)
    # Wrap with VecTransposeImage to match the training setup
    env = VecTransposeImage(env)
    env.seed(0)
    
    print(f"Environment observation space: {env.observation_space}")
    
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
            
            screen_buffer = env.envs[0].game.get_state().screen_buffer
            images.append(screen_buffer)
            
            step_count += 1
            if step_count > 1000:
                print("Episode too long, breaking...")
                break
    
    print(f"Saving GIF with {len(images)} frames")
    imageio.mimsave(file_path, images, fps=35)
    
    env.close()
    return agent

os.makedirs('figures', exist_ok=True)

agent = make_gif('logs/models/multi_map/best_model.zip', 'figures/sp_agent.gif')
print("GIF created successfully!")
