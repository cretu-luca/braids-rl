import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces

from .braid import Braid
from .config import Configuration
from .braid_generator import BraidGenerator

class BraidEnv(gym.Env):
    def __init__(self, dataset_path: str, n_strands: int = Configuration.N_STRANDS, max_len: int = Configuration.MAX_LEN):
        super(BraidEnv, self).__init__()

        self.n_strands = n_strands
        self.max_len = max_len

        self.current_braid = None
        self.last_action = None
        self.stuck_steps = 0

        self.dataset = BraidGenerator.load_dataset(dataset_path)

        self.action_space = spaces.MultiDiscrete([4, max_len])

        self.observation_space = spaces.Box(
            low=-(n_strands-1), 
            high=max(n_strands-1, 100), 
            shape=(max_len + 1,), 
            dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self.dataset:
            self.current_braid = Braid([], self.n_strands)
        else:
            self.current_braid = random.choice(self.dataset).copy()
            if len(self.current_braid) > self.max_len:
                self.current_braid.word = self.current_braid.word[:self.max_len]

        self.last_action = None
        self.stuck_steps = 0

        return self._get_obs(), {}

    def _get_obs(self):
        word_pad = self.current_braid.get_padded_word(self.max_len)
        return np.array(word_pad + [self.stuck_steps], dtype=np.int32)

    def step(self, action):
        move_type, index = action
        prev_len = len(self.current_braid)
        success = False
        
        is_reversal = False
        if self.last_action is not None:
            last_move, last_idx = self.last_action
            if last_idx == index:
                if (last_move == 3 and move_type == 2) or (last_move == 2 and move_type == 3):
                    is_reversal = True

        if is_reversal:
            self.stuck_steps += 1
            return self._get_obs(), Configuration.REWARD_LOOP, False, False, {"success": False}

        if move_type == 0:
            success = self.current_braid.apply_commutation(index)
        elif move_type == 1:
            success = self.current_braid.apply_braid_relation(index)
        elif move_type == 2:
            success = self.current_braid.remove_pair_at_index(index)
        elif move_type == 3:
            if len(self.current_braid) < self.max_len - 2:
                gen = random.randint(1, self.n_strands - 1)
                success = self.current_braid.insert_canceling_pair(index, gen)

        reward = Configuration.REWARD_STEP

        if not success:
            self.stuck_steps += 1
            reward += Configuration.REWARD_INVALID - (0.1 * self.stuck_steps)
        else:
            self.stuck_steps = 0
            self.last_action = action 
            new_len = len(self.current_braid)
            
            if new_len < prev_len:
                reward += Configuration.REWARD_SHRINK
            elif new_len > prev_len:
                reward += Configuration.REWARD_GROW
            else:
                reward += 0.1

            if new_len == 0:
                return self._get_obs(), Configuration.REWARD_SOLVED, True, False, {"success": True}

        truncated = len(self.current_braid) >= self.max_len
        if truncated:
            reward += Configuration.REWARD_INVALID * 5 

        return self._get_obs(), reward, False, truncated, {"success": success}