from stable_baselines3.common.callbacks import BaseCallback
import os
import imageio


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


class VideoEvalCallback(BaseCallback):
    """
    Run evaluation episodes on a provided eval env and save a GIF per episode.

    This callback is independent from SB3's EvalCallback and is meant to run
    at the same frequency so users get videos for every eval cycle.
    """

    def __init__(
        self,
        eval_env,
        gif_dir: str = "figures/eval",
        n_eval_episodes: int = 3,
        eval_freq: int = 5000,
        deterministic: bool = False,
        fps: int = 35,
        max_steps_per_episode: int = 5000,
        video_format: str = "gif",
        codec: str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.gif_dir = gif_dir
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.fps = fps
        self.max_steps_per_episode = max_steps_per_episode
        self.video_format = video_format.lower()
        self.codec = codec
        self._last_eval_step = 0

    def _on_training_start(self) -> None:
        os.makedirs(self.gif_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            try:
                self._run_and_save_gifs()
            except Exception as exc:
                if self.verbose:
                    print(f"[GifEvalCallback] Failed to create GIFs: {exc}")
        return True

    def _run_and_save_gifs(self) -> None:
        for ep_idx in range(self.n_eval_episodes):
            images = []

            obs = self.eval_env.reset()
            done = False
            step_count = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, dones, infos = self.eval_env.step(action)

                try:
                    done = bool(dones[0])  # type: ignore[index]
                except Exception:
                    done = bool(dones)

                frame_list = None
                try:
                    if isinstance(infos, (list, tuple)) and len(infos) > 0:
                        frame_list = infos[0].get("captured_frames")
                except Exception:
                    frame_list = None

                if frame_list:
                    images.extend(frame_list)
                else:
                    try:
                        state = self.eval_env.venv.envs[0].env.game.get_state()  # type: ignore[attr-defined]
                        if state is not None:
                            images.append(state.screen_buffer)
                    except Exception:
                        pass

                step_count += 1
                if step_count > self.max_steps_per_episode:
                    if self.verbose:
                        print("[GifEvalCallback] Episode too long, breaking...")
                    break

            if images:
                ext = "mp4" if self.video_format == "mp4" else "gif"
                filename = f"eval_step_{self.num_timesteps}_ep_{ep_idx + 1}.{ext}"
                file_path = os.path.join(self.gif_dir, filename)
                self._save_video(file_path, images)
                if self.verbose:
                    print(f"[GifEvalCallback] Saved {ext.upper()}: {file_path} ({len(images)} frames)")
            else:
                if self.verbose:
                    print("[GifEvalCallback] No frames captured; skipping save")

    def _save_video(self, file_path: str, images: list) -> None:
        try:
            if file_path.lower().endswith(".mp4"):
                writer_kwargs = {"fps": self.fps}
                if self.codec is not None:
                    writer_kwargs["codec"] = self.codec
                with imageio.get_writer(file_path, **writer_kwargs) as writer:
                    for frame in images:
                        writer.append_data(frame)
            else:
                imageio.mimsave(file_path, images, fps=self.fps)
        except Exception as e:
            if not file_path.lower().endswith(".gif"):
                fallback_path = os.path.splitext(file_path)[0] + ".gif"
                imageio.mimsave(fallback_path, images, fps=self.fps)
