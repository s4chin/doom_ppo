from stable_baselines3.common.callbacks import BaseCallback


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
