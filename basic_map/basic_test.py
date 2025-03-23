import imageio
import os
from train_sp import create_vec_env, frame_processor
from stable_baselines3 import PPO

def make_gif(agent, file_path):
    env = create_vec_env(frame_skip=1, frame_processor=frame_processor)
    env.set_attr('game.set_seed', 0)
    
    images = []

    for i in range(5):
        obs = env.reset()

        done = False
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            images.append(env.envs[0].game.get_state().screen_buffer)

    imageio.mimsave(file_path, images, fps=35)

    env.close()

# Uncomment this if you want to pick the best model from a previous training session
agent = PPO.load('logs/models/basic/best_model.zip')
os.makedirs('figures', exist_ok=True)
make_gif(agent, 'figures/basic_agent.gif')

# env = create_vec_env(frame_skip=1, frame_processor=frame_processor)

# agent.set_env(env)
# agent.learn(total_timesteps=20000)

# env.close()
# os.makedirs('figures', exist_ok=True)
# make_gif(agent, 'figures/basic_agent_adjusted.gif')