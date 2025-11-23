import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces

from .braid import Braid
from .config import Configuration
from .braid_generator import BraidGenerator

class BraidEnv(gym.Env):
    def __init__(self, dataset_path: str, n_strands: int, max_len: int, config: Configuration):
        super().__init__()

        self.n_strands = n_strands
        self.max_len = max_len
        self.config = config
        
        self.dataset = BraidGenerator.load_dataset(dataset_path)
        if not self.dataset:
            print(f"Warning: No data found at {dataset_path}")
            
        self.current_braid = None
        
        # Flattened Action Space: 4 * max_len
        # 0..max_len-1          : Commute
        # max_len..2*max_len-1  : R3
        # 2*max_len..3*max_len-1: Remove
        # 3*max_len..4*max_len-1: Insert
        self.action_space = spaces.Discrete(4 * max_len)

        self.observation_space = spaces.Box(
            low=-(n_strands-1), 
            high=max(n_strands-1, 100), 
            shape=(max_len,),
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
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros(self.max_len, dtype=np.int32)
        word = self.current_braid.word
        length = min(len(word), self.max_len)
        if length > 0:
            obs[:length] = word[:length]
        return obs

    def action_masks(self):
        mask = np.zeros(4 * self.max_len, dtype=bool)
        
        curr_len = len(self.current_braid)

        for i in range(curr_len - 1):
            if self.current_braid.check_commutation(i):
                mask[i] = True

        offset_r3 = self.max_len
        for i in range(curr_len - 2):
            if self.current_braid.check_braid_relation(i):
                mask[offset_r3 + i] = True
                
        offset_rem = 2 * self.max_len
        for i in range(curr_len - 1):
            if self.current_braid.check_remove_pair(i):
                mask[offset_rem + i] = True
                
        offset_ins = 3 * self.max_len
        if curr_len < self.max_len - 2:
            for i in range(curr_len + 1):
                mask[offset_ins + i] = True
             
        return mask

    def step(self, action):
        move_type = action // self.max_len
        index = action % self.max_len
        
        prev_len = len(self.current_braid)
        success = False

        if move_type == 0: 
            success = self.current_braid.apply_commutation(index)
        elif move_type == 1: 
            success = self.current_braid.apply_braid_relation(index)
        elif move_type == 2: 
            success = self.current_braid.remove_pair_at_index(index)
        elif move_type == 3:
            gen = random.randint(1, self.n_strands - 1)
            success = self.current_braid.insert_canceling_pair(index, gen)

        reward = self.config.REWARD_STEP
        
        if not success:
            reward += self.config.REWARD_INVALID 
        else:
            new_len = len(self.current_braid)
            if new_len < prev_len: reward += self.config.REWARD_SHRINK
            elif new_len > prev_len: reward += self.config.REWARD_GROW
            else: reward += 0.0

            if new_len == 0:
                return self._get_obs(), self.config.REWARD_SOLVED, True, False, {"success": True}

        truncated = len(self.current_braid) >= self.max_len
        if truncated:
            reward += self.config.REWARD_INVALID * 2

        return self._get_obs(), reward, False, truncated, {"success": success}