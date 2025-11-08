import itertools
import math
import typing as t
import random

import cv2
import gymnasium as gym
import numpy as np
import vizdoom
from gymnasium import spaces
from vizdoom import Button, GameVariable

# Type alias for clarity
Frame = np.ndarray

AMMO_VARIABLES = [GameVariable.AMMO0, GameVariable.AMMO1, GameVariable.AMMO2, GameVariable.AMMO3, GameVariable.AMMO4, GameVariable.AMMO5, GameVariable.AMMO6, GameVariable.AMMO7, GameVariable.AMMO8, GameVariable.AMMO9]

# Buttons that cannot be used together
MUTUALLY_EXCLUSIVE_GROUPS = [
    [Button.MOVE_RIGHT, Button.MOVE_LEFT],
    [Button.TURN_RIGHT, Button.TURN_LEFT],
    [Button.MOVE_FORWARD, Button.MOVE_BACKWARD],
]

# Buttons that can only be used alone.
EXCLUSIVE_BUTTONS = []


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
    """Return a reduced, hand-picked discrete action set.

    We prefer a curated list of common actions (move, turn, strafe, shoot, use),
    filtered to only include buttons present in the current scenario. If none of
    the curated actions are applicable, we fall back to the combinatorial
    enumeration (with mutual exclusivity constraints) used previously.
    """
    # Map available buttons to indices for vector construction
    button_to_index: dict[vizdoom.Button, int] = {btn: i for i, btn in enumerate(buttons)}

    def construct_action_vector(pressed: list[vizdoom.Button]) -> list[float]:
        vec = [0.0] * len(buttons)
        for b in pressed:
            vec[button_to_index[b]] = 1.0
        return vec

    def all_present(pressed: list[vizdoom.Button]) -> bool:
        return all(b in button_to_index for b in pressed)

    # Curated actions (no no-op). Adjust this list to taste.
    curated_actions_raw: list[list[vizdoom.Button]] = [
        [Button.MOVE_FORWARD],
        [Button.MOVE_BACKWARD],
        [Button.MOVE_LEFT],
        [Button.MOVE_RIGHT],
        [Button.TURN_LEFT],
        [Button.TURN_RIGHT],
        [Button.ATTACK],
        [Button.MOVE_FORWARD, Button.ATTACK],
        [Button.TURN_LEFT, Button.ATTACK],
        [Button.TURN_RIGHT, Button.ATTACK],
        [Button.MOVE_LEFT, Button.ATTACK],
        [Button.MOVE_RIGHT, Button.ATTACK],
        [Button.USE],
    ]

    curated_vectors: list[list[float]] = [
        construct_action_vector(combo)
        for combo in curated_actions_raw
        if all_present(combo)
    ]

    if len(curated_vectors) > 0:
        print('Built curated action space of size {} from buttons {}'.format(len(curated_vectors), buttons))
        return curated_vectors

    # Fallback: original combinatorial enumeration with constraints
    action_combinations = np.array([list(seq) for seq in itertools.product([0., 1.], repeat=len(buttons))])

    # Build action mask from action combinations and exclusion mask
    illegal_mask = (has_excluded_pair(action_combinations, buttons)
                    | has_exclusive_button(action_combinations, buttons))

    possible_actions = action_combinations[~illegal_mask]
    possible_actions = possible_actions[np.sum(possible_actions, axis=1) > 0]  # Remove no-op

    print('Built action space of size {} from buttons {}'.format(len(possible_actions), buttons))
    return possible_actions.tolist()


class DoomEnvSP(gym.Env):
    """Wrapper environment following Gymnasium interface for a VizDoom game instance."""
    
    def __init__(self, game: vizdoom.DoomGame, frame_processor: t.Callable, automap_processor: t.Callable, frame_skip: int = 4, n_frames: int = 4, n_actions_history: int = 32, capture_intermediate_frames: bool = False):
        """Initialize the Doom environment with a game instance, frame processor, and frame skip."""
        super().__init__()

        self.sticky_prob = 0.05
        self._last_action = -1
        
        self.n_frames = n_frames
        self.n_actions_history = n_actions_history

        h, w, c = game.get_screen_height(), game.get_screen_width(), game.get_screen_channels()
        frame_shape = frame_processor(np.zeros((h, w, c))).shape
        self.single_frame_shape = frame_shape

        self.game = game
        self.possible_actions = get_available_actions(game.get_available_buttons())
        self.action_space = spaces.Discrete(len(self.possible_actions))
        self.available_buttons = game.get_available_buttons()
        self.frame_skip = frame_skip
        # Should be true only at inference
        self.capture_intermediate_frames = capture_intermediate_frames

        self.frame_processor = frame_processor
        self.automap_processor = automap_processor

        self.empty_frame = np.zeros(self.single_frame_shape, dtype=np.uint8)
        self.action_history = np.full((self.n_actions_history,), -1, dtype=np.int64)
        
        if game.is_automap_buffer_enabled():
            automap_shape = frame_shape
            self.empty_automap = np.zeros(automap_shape, dtype=np.uint8)
            
            stacked_screen_channels = frame_shape[2] * n_frames
            stacked_automap_channels = automap_shape[2] * n_frames
            
            self.observation_space = spaces.Dict({
                'screen': spaces.Box(
                    low=0, high=255,
                    shape=(frame_shape[0], frame_shape[1], stacked_screen_channels),
                    dtype=np.uint8
                ),
                'automap': spaces.Box(
                    low=0, high=255,
                    shape=(automap_shape[0], automap_shape[1], stacked_automap_channels),
                    dtype=np.uint8
                ),
                'action_history': spaces.Box(
                    low=-1, high=len(self.possible_actions)-1,
                    shape=(self.n_actions_history,),
                    dtype=np.int64
                )
            })
            
            self.screen_stack = [self.empty_frame] * n_frames
            self.automap_stack = [self.empty_automap] * n_frames
        else:
            raise ValueError('Game automap is not enabled')
        
        self.state = self._get_stacked_frames()

        self.prev_health = self.game.get_game_variable(GameVariable.HEALTH)
        self.prev_armor = self.game.get_game_variable(GameVariable.ARMOR)
        self.prev_ammo = sum(self.game.get_game_variable(variable) for variable in AMMO_VARIABLES)
        self.prev_hitcount = self.game.get_game_variable(GameVariable.HITCOUNT)
        self.prev_killcount = self.game.get_game_variable(GameVariable.KILLCOUNT)
        self.prev_itemcount = self.game.get_game_variable(GameVariable.ITEMCOUNT)
        self.prev_secretcount = self.game.get_game_variable(GameVariable.SECRETCOUNT)
        self.grid_size = 128 # TODO: TBD
        self.start_x, self.start_y = self.game.get_game_variable(GameVariable.POSITION_X), self.game.get_game_variable(GameVariable.POSITION_Y)

        self.visited_cells = {(math.floor(self.start_x / self.grid_size), math.floor(self.start_y / self.grid_size))} # lets see

    def step(self, action: int) -> t.Tuple[Frame, float, bool, bool, dict]:
        """Apply an action to the environment and return the resulting state."""
        if self._last_action != -1 and random.random() < self.sticky_prob:
            action = self._last_action
        self._last_action = action
        print(f"{action=}")

        info: dict = {"executed_action": action}
        if self.capture_intermediate_frames:
            total_reward = 0.0
            captured_frames: list[np.ndarray] = []
            for _ in range(self.frame_skip):
                step_reward = self.game.make_action(self.possible_actions[action], 1)
                total_reward += step_reward
                state = self.game.get_state()
                if state is not None:
                    captured_frames.append(state.screen_buffer)
                if self.game.is_episode_finished():
                    break
            reward = float(total_reward)
            terminated = self.game.is_episode_finished()
            info["captured_frames"] = captured_frames
        else:
            reward = self.game.make_action(self.possible_actions[action], self.frame_skip)
            terminated = self.game.is_episode_finished()
        truncated = False  # VizDoom handles termination; no truncation needed here
        
        # Update action history
        self.action_history = np.roll(self.action_history, shift=-1)
        self.action_history[-1] = action
        
        new_screen, new_automap = self._get_frame(terminated)
        self.screen_stack.pop(0)
        self.screen_stack.append(new_screen)
        self.automap_stack.pop(0)
        self.automap_stack.append(new_automap)
        
        self.state = self._get_stacked_frames()
        reward = self.get_reward(self.state, reward, terminated, truncated, {})
        return self.state, reward, terminated, truncated, info

    def get_reward(self, obs, _, done, truncated, info):
        state = self.game.get_state()
        reward = 0

        # Detect player death independently of state availability (state is None on terminal step)
        player_dead = self.game.is_player_dead()

        if done and player_dead:
            reward += -5000

        if state:
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

            # 2. Death penalty handled before this block
            
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
            grid_x = math.floor(curr_x / self.grid_size)
            grid_y = math.floor(curr_y / self.grid_size)
            cell = (grid_x, grid_y)
            if cell not in self.visited_cells:
                self.visited_cells.add(cell)
                l1_distance = abs(grid_x - math.floor(self.start_x / self.grid_size)) + \
                             abs(grid_y - math.floor(self.start_y / self.grid_size))
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

            self.prev_health = curr_health
            self.prev_armor = curr_armor
            self.prev_ammo = curr_ammo
            self.prev_hitcount = curr_hitcount
            self.prev_killcount = curr_killcount
            self.prev_itemcount = curr_itemcount
            self.prev_secretcount = curr_secretcount


        return reward

    def reset(self, *, seed: t.Optional[int] = None, options: t.Optional[dict] = None) -> t.Tuple[Frame, dict]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        self.game.new_episode()
        
        # Reset action history
        self._last_action = -1
        self.action_history = np.full((self.n_actions_history,), -1, dtype=np.int64)
        
        initial_screen, initial_automap = self._get_frame()
        self.screen_stack = [initial_screen] * self.n_frames
        self.automap_stack = [initial_automap] * self.n_frames
        
        self.state = self._get_stacked_frames()
        
        state = self.game.get_state()
        if state:
            self.prev_health = self.game.get_game_variable(GameVariable.HEALTH)
            self.prev_armor = self.game.get_game_variable(GameVariable.ARMOR)
            self.prev_ammo = sum(self.game.get_game_variable(variable) for variable in AMMO_VARIABLES)
            self.prev_hitcount = self.game.get_game_variable(GameVariable.HITCOUNT)
            self.prev_killcount = self.game.get_game_variable(GameVariable.KILLCOUNT)
            self.prev_itemcount = self.game.get_game_variable(GameVariable.ITEMCOUNT)
            self.prev_secretcount = self.game.get_game_variable(GameVariable.SECRETCOUNT)
            
            self.start_x, self.start_y = self.game.get_game_variable(GameVariable.POSITION_X), self.game.get_game_variable(GameVariable.POSITION_Y)
            self.visited_cells = {(math.floor(self.start_x / self.grid_size), math.floor(self.start_y / self.grid_size))}
        return self.state, {}

    def close(self) -> None:
        """Close the VizDoom game instance."""
        self.game.close()

    def render(self, mode: str = "human") -> None:
        """Render method (not implemented as VizDoom handles its own rendering)."""
        pass

    def _get_frame(self, done: bool = False) -> Frame:
        """Get the current frame, or an empty frame if the episode is done."""
        if done:
            return self.empty_frame, self.empty_automap
        
        state = self.game.get_state()
        screen = self.frame_processor(state.screen_buffer)
        automap = self.automap_processor(state.automap_buffer)
        return screen, automap

    def seed(self, seed: t.Optional[int] = None) -> t.List[t.Optional[int]]:
        """Set the seed for VizDoom's random number generator."""
        if seed is not None:
            self.game.set_seed(seed)
        return [seed]

    def _get_stacked_frames(self) -> dict[str, object]:
        """Stack frames along the channel dimension (last axis for numpy arrays)."""
        return {
            'screen': np.concatenate(self.screen_stack, axis=2),
            'automap': np.concatenate(self.automap_stack, axis=2),
            'action_history': self.action_history
        }
    
    def get_action_labels(self) -> list[str]:
        """Generate human-readable labels for each action based on button combinations.
        
        Returns:
            List of action labels, one per discrete action in the action space.
        """
        # Map button enum to readable name
        button_names = {
            Button.MOVE_FORWARD: "Forward",
            Button.MOVE_BACKWARD: "Backward",
            Button.MOVE_LEFT: "Left",
            Button.MOVE_RIGHT: "Right",
            Button.TURN_LEFT: "Turn Left",
            Button.TURN_RIGHT: "Turn Right",
            Button.ATTACK: "Attack",
            Button.USE: "Use",
            Button.SPEED: "Run",
            Button.STRAFE: "Strafe",
            Button.CROUCH: "Crouch",
            Button.JUMP: "Jump",
            Button.RELOAD: "Reload",
            Button.ZOOM: "Zoom",
        }
        
        # Define priority order for button display (movement before actions)
        button_priority = {
            Button.MOVE_FORWARD: 0,
            Button.MOVE_BACKWARD: 1,
            Button.MOVE_LEFT: 2,
            Button.MOVE_RIGHT: 3,
            Button.TURN_LEFT: 4,
            Button.TURN_RIGHT: 5,
            Button.STRAFE: 6,
            Button.SPEED: 7,
            Button.CROUCH: 8,
            Button.JUMP: 9,
            Button.ATTACK: 10,
            Button.USE: 11,
            Button.RELOAD: 12,
            Button.ZOOM: 13,
        }
        
        labels = []
        for action_vector in self.possible_actions:
            # Find which buttons are pressed in this action
            pressed_buttons = []
            for i, val in enumerate(action_vector):
                if val == 1.0:
                    button = self.available_buttons[i]
                    pressed_buttons.append((button, button_names.get(button, str(button))))
            
            # Sort by priority to ensure consistent ordering (movement before actions)
            pressed_buttons.sort(key=lambda x: button_priority.get(x[0], 100))
            button_labels = [name for _, name in pressed_buttons]
            
            # Combine button names with "+"
            label = "+".join(button_labels) if button_labels else "No-op"
            labels.append(label)
        
        return labels
