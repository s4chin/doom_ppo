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
import itertools
from vizdoom import Button, GameVariable

# Type alias for clarity
Frame = np.ndarray

AMMO_VARIABLES = [GameVariable.AMMO0, GameVariable.AMMO1, GameVariable.AMMO2, GameVariable.AMMO3, GameVariable.AMMO4, GameVariable.AMMO5, GameVariable.AMMO6, GameVariable.AMMO7, GameVariable.AMMO8, GameVariable.AMMO9]


# class DoomEnv(gym.Env):
#     """Wrapper environment following Gymnasium interface for a VizDoom game instance."""
    
#     def __init__(self, game: vizdoom.DoomGame, frame_processor: t.Callable, frame_skip: int = 4):
#         """Initialize the Doom environment with a game instance, frame processor, and frame skip."""
#         super().__init__()
        
#         # Define action space: Discrete space based on number of available buttons
#         self.action_space = spaces.Discrete(game.get_available_buttons_size())
        
#         # Define observation space: Processed screen buffer dimensions
#         h, w, c = game.get_screen_height(), game.get_screen_width(), game.get_screen_channels()
#         processed_shape = frame_processor(np.zeros((h, w, c))).shape
#         self.observation_space = spaces.Box(low=0, high=255, shape=processed_shape, dtype=np.uint8)
        
#         # Store instance variables
#         self.game = game
#         self.possible_actions = np.eye(self.action_space.n).tolist()  # One-hot encoded actions for VizDoom
#         self.frame_skip = frame_skip
#         self.frame_processor = frame_processor
#         self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
#         self.state = self.empty_frame

#     def step(self, action: int) -> t.Tuple[Frame, float, bool, bool, dict]:
#         """Apply an action to the environment and return the resulting state."""
#         reward = self.game.make_action(self.possible_actions[action], self.frame_skip)
#         terminated = self.game.is_episode_finished()
#         truncated = False  # VizDoom handles termination; no truncation needed here
#         self.state = self._get_frame(terminated)
#         return self.state, reward, terminated, truncated, {}

#     def reset(self, *, seed: t.Optional[int] = None, options: t.Optional[dict] = None) -> t.Tuple[Frame, dict]:
#         """Reset the environment to start a new episode."""
#         super().reset(seed=seed)
#         self.game.new_episode()
#         self.state = self._get_frame()
#         return self.state, {}

#     def close(self) -> None:
#         """Close the VizDoom game instance."""
#         self.game.close()

#     def render(self, mode: str = "human") -> None:
#         """Render method (not implemented as VizDoom handles its own rendering)."""
#         pass

#     def _get_frame(self, done: bool = False) -> Frame:
#         """Get the current frame, or an empty frame if the episode is done."""
#         return self.frame_processor(self.game.get_state().screen_buffer) if not done else self.empty_frame

#     def seed(self, seed: t.Optional[int] = None) -> t.List[t.Optional[int]]:
#         """Set the seed for VizDoom's random number generator."""
#         if seed is not None:
#             self.game.set_seed(seed)
#         return [seed]


# Buttons that cannot be used together
MUTUALLY_EXCLUSIVE_GROUPS = [
    [Button.MOVE_RIGHT, Button.MOVE_LEFT],
    [Button.TURN_RIGHT, Button.TURN_LEFT],
    [Button.MOVE_FORWARD, Button.MOVE_BACKWARD],
    [Button.ATTACK, Button.USE],
]

# Buttons that can only be used alone.
EXCLUSIVE_BUTTONS = [Button.ATTACK, Button.USE]


def has_exclusive_button(actions: np.ndarray, buttons: np.array) -> np.array:
    exclusion_mask = np.isin(buttons, EXCLUSIVE_BUTTONS)
    
    # Flag actions that have more than 1 active button among exclusive list.
    return (np.any(actions.astype(bool) & exclusion_mask, axis=-1)) & (np.sum(actions, axis=-1) > 1)


def has_excluded_pair(actions: np.ndarray, buttons: np.array) -> np.array:
    # Create mask of shape (n_mutual_exclusion_groups, n_available_buttons), marking location of excluded pairs.
    mutual_exclusion_mask = np.array([np.isin(buttons, excluded_group) 
                                      for excluded_group in MUTUALLY_EXCLUSIVE_GROUPS])

    # Flag actions that have more than 1 button active in any of the mutual exclusion groups.
    return np.any(np.sum(
        # Resulting shape (n_actions, n_mutual_exclusion_groups, n_available_buttons)
        (actions[:, np.newaxis, :] * mutual_exclusion_mask.astype(int)),
        axis=-1) > 1, axis=-1)


def get_available_actions(buttons: np.array) -> t.List[t.List[float]]:
    # Create list of all possible actions of size (2^n_available_buttons x n_available_buttons)
    action_combinations = np.array([list(seq) for seq in itertools.product([0., 1.], repeat=len(buttons))])

    # Build action mask from action combinations and exclusion mask
    illegal_mask = (has_excluded_pair(action_combinations, buttons)
                    | has_exclusive_button(action_combinations, buttons))

    possible_actions = action_combinations[~illegal_mask]
    possible_actions = possible_actions[np.sum(possible_actions, axis=1) > 0]  # Remove no-op

    print('Built action space of size {} from buttons {}'.format(len(possible_actions), buttons))
    return possible_actions.tolist()

possible_actions = get_available_actions(np.array([
    Button.ATTACK, Button.MOVE_FORWARD, Button.MOVE_LEFT, 
    Button.MOVE_RIGHT, Button.TURN_LEFT, Button.TURN_RIGHT, Button.USE]))

class DoomEnvSP(gym.Env):
    """Wrapper environment following Gymnasium interface for a VizDoom game instance."""
    
    def __init__(self, game: vizdoom.DoomGame, frame_processor: t.Callable, frame_skip: int = 4, n_frames: int = 4):
        """Initialize the Doom environment with a game instance, frame processor, and frame skip."""
        super().__init__()
        
        # Store number of frames to stack
        self.n_frames = n_frames
        
        # Define action space: Discrete space based on number of available buttons
        self.action_space = spaces.Discrete(game.get_available_buttons_size())
        
        # Define observation space: Processed screen buffer dimensions
        h, w, c = game.get_screen_height(), game.get_screen_width(), game.get_screen_channels()
        processed_shape = frame_processor(np.zeros((h, w, c))).shape
        
        # Update observation space to account for stacked frames (channels are multiplied by n_frames)
        self.single_frame_shape = processed_shape
        stacked_channels = processed_shape[2] * n_frames  # Multiply channels by n_frames
        
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(processed_shape[0], processed_shape[1], stacked_channels),  # [H, W, C*n_frames]
            dtype=np.uint8
        )
        
        # Store instance variables
        self.game = game
        self.possible_actions = possible_actions
        self.frame_skip = frame_skip
        self.frame_processor = frame_processor
        self.empty_frame = np.zeros(self.single_frame_shape, dtype=np.uint8)
        
        # Initialize frame stack
        self.frame_stack = [self.empty_frame] * n_frames
        self.state = self._get_stacked_frames()

        self.prev_health = 100  # Typical starting health in Doom
        self.prev_armor = 0
        self.prev_ammo = 0
        self.prev_killcount = 0
        self.prev_itemcount = 0
        self.prev_secretcount = 0
        self.visited_cells = set()
        self.start_x, self.start_y = None, None
        self.grid_size = 50  # Grid size for exploration; adjust based on map scale

    def step(self, action: int) -> t.Tuple[Frame, float, bool, bool, dict]:
        """Apply an action to the environment and return the resulting state."""
        reward = self.game.make_action(self.possible_actions[action], self.frame_skip)
        terminated = self.game.is_episode_finished()
        truncated = False  # VizDoom handles termination; no truncation needed here
        
        new_frame = self._get_frame(terminated)
        # Update frame stack with the new frame
        self.frame_stack.pop(0)
        self.frame_stack.append(new_frame)
        self.state = self._get_stacked_frames()
        
        reward = self.get_reward(self.state, reward, terminated, truncated, {})
        return self.state, reward, terminated, truncated, {}

    def get_reward(self, obs, _, done, truncated, info):
        state = self.game.get_state()
        reward = 0

        if state:
            # Current state variables
            curr_health = self.game.get_game_variable(GameVariable.HEALTH)
            curr_armor = self.game.get_game_variable(GameVariable.ARMOR)
            curr_ammo = sum(self.game.get_game_variable(variable) for variable in AMMO_VARIABLES)
            curr_hitcount = self.game.get_game_variable(GameVariable.HITCOUNT)
            curr_killcount = self.game.get_game_variable(GameVariable.KILLCOUNT)
            curr_itemcount = self.game.get_game_variable(GameVariable.ITEMCOUNT)
            curr_secretcount = self.game.get_game_variable(GameVariable.SECRETCOUNT)
            curr_x = self.game.get_game_variable(GameVariable.POSITION_X)
            curr_y = self.game.get_game_variable(GameVariable.POSITION_Y)

            # Reward calculations
            # 1. Player hit: -100 if health decreases
            if curr_health < self.prev_health:
                reward += -100

            # 2. Player death: -5,000 if health is 0 and episode ends
            if done and curr_health <= 0:
                reward += -5000
            
            # 3. Enemy hit: 300 per hit
            if curr_hitcount > self.prev_hitcount:
                reward += 300 * (curr_hitcount - self.prev_hitcount)
            
            # 4. Enemy kill: 1000 per kill
            if curr_killcount > self.prev_killcount:
                reward += 1000 * (curr_killcount - self.prev_killcount)

            # 5. Item pickup: 100 per item
            if curr_itemcount > self.prev_itemcount:
                reward += 100 * (curr_itemcount - self.prev_itemcount)

            # 6. Secret found: 500 per secret
            if curr_secretcount > self.prev_secretcount:
                reward += 500 * (curr_secretcount - self.prev_secretcount)

            # 7. Exploration: Reward for new areas based on grid
            grid_x = int(curr_x / self.grid_size)
            grid_y = int(curr_y / self.grid_size)
            cell = (grid_x, grid_y)
            if cell not in self.visited_cells:
                self.visited_cells.add(cell)
                l1_distance = abs(grid_x - int(self.start_x / self.grid_size)) + \
                             abs(grid_y - int(self.start_y / self.grid_size))
                reward += 20 * (1 + 0.5 * l1_distance)

            # 8. Health delta: 10 * change in health
            health_delta = curr_health - self.prev_health
            reward += 10 * health_delta

            # 9. Armor delta: 10 * change in armor
            armor_delta = curr_armor - self.prev_armor
            reward += 10 * armor_delta

            # 10. Ammo delta: 10 * positive change, 1 * negative change
            ammo_delta = curr_ammo - self.prev_ammo
            reward += 10 * max(0, ammo_delta) + min(0, ammo_delta)

            # 11. Penalty for each step
            reward -= 1

            # Update previous state variables
            self.prev_health = curr_health
            self.prev_armor = curr_armor
            self.prev_ammo = curr_ammo
            self.prev_killcount = curr_killcount
            self.prev_itemcount = curr_itemcount
            self.prev_secretcount = curr_secretcount

        return reward

    def reset(self, *, seed: t.Optional[int] = None, options: t.Optional[dict] = None) -> t.Tuple[Frame, dict]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        self.game.new_episode()
        
        # Reset frame stack with initial frame
        initial_frame = self._get_frame()
        self.frame_stack = [initial_frame] * self.n_frames
        self.state = self._get_stacked_frames()
        
        state = self.game.get_state()
        if state:
            self.prev_health = self.game.get_game_variable(GameVariable.HEALTH)
            self.prev_armor = self.game.get_game_variable(GameVariable.ARMOR)
            self.prev_ammo = sum(self.game.get_game_variable(variable) for variable in AMMO_VARIABLES)
            self.prev_killcount = self.game.get_game_variable(GameVariable.KILLCOUNT)
            self.prev_itemcount = self.game.get_game_variable(GameVariable.ITEMCOUNT)
            self.prev_secretcount = self.game.get_game_variable(GameVariable.SECRETCOUNT)
            self.start_x = self.game.get_game_variable(GameVariable.POSITION_X)
            self.start_y = self.game.get_game_variable(GameVariable.POSITION_Y)
            self.visited_cells = {(int(self.start_x / self.grid_size), int(self.start_y / self.grid_size))}
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

    def _get_stacked_frames(self) -> Frame:
        """Stack frames along the channel dimension (last axis for numpy arrays)."""
        # Concatenate along the channel dimension (axis 2 for [H, W, C] format)
        return np.concatenate(self.frame_stack, axis=2)
