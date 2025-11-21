import numpy as np
import random

from gymnasium import gym, spaces

from braid import Braid
from config import Configuration

class BraidEnv(gym.Env):
    def __init__(self, n_strands: int, max_len: int):
        super().__init__()

        self.n_strands = n_strands
        self.max_len = max_len

        self.current_braid = None
        self.dataset = None

        self.action_space = spaces.MultiDiscrete([4, max_len])

        self.observation_space = spaces.Box(
            low=-(n_strands-1), 
            high=max(n_strands-1, 100), 
            shape=(max_len + 1,), 
            dtype=np.int32
        )

    def reset(self):
        super().reset()

        if not self.dataset:
            self.current_braid = Braid([], self.n_strands)
        else:
            self.current_braid = random.choice(self.dataset).copy()

        self.last_action = None

        return self.get_obs(), {}

    def get_state(self):
        state = np.zeros(self.max_len)
        word = self.current_braid.word

        if len(word) > 0:
            state[:min(len(word), self.max_len)] = word[:self.max_len]

        return state

    def step(self, action):
        move_type, index = action
        prev_len = len(self.current_braid)
        success = False

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
            reward += Configuration.REWARD_INVALID
        else:
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
            reward += Configuration.REWARD_INVALID * 2

        return self._get_obs(), reward, False, truncated, {"success": success}
