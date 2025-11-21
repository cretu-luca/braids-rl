from gymnasium import gym, spaces
import numpy as np

class BraidEnv(gym.Env):
    def __init__(self, n_strands: int, max_len: int):
        super().__init__()

        self.n_strands = n_strands
        self.max_len = max_len

        self.current_braid = None

        self.action_space = spaces.MultiDiscrete([4, max_len])

        self.observation_space = spaces.Box(
            low=-(n_strands-1), 
            high=max(n_strands-1, 100), 
            shape=(max_len + 1,), 
            dtype=np.int32
        )

    def reset(self):
        pass

    def get_obs(self):
        pass

    def step(self):
        pass
