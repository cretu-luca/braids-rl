import numpy as np
import os
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from .agent_metrics import AgentMetrics

def mask_fn(env):
    return env.action_masks()

class BraidAgent:
    def __init__(self, config, hyperparameters, model_path=None, name="untrained"):
        self.config = config
        self.hyperparameters = hyperparameters
        self.model = None
        self.name = name
        self.metrics = AgentMetrics(config)

        if model_path:
            self.load(model_path)

    def train(self, env, total_timesteps, save_path=None):
        env = ActionMasker(env, mask_fn)

        if self.model is None:
            self.model = MaskablePPO("MlpPolicy", env, **self.hyperparameters)
        else:
            self.model.set_env(env)

        print(f"[{self.name}] Training for {total_timesteps} steps...")
        self.model.learn(total_timesteps=total_timesteps)
        
        if save_path:
            self.save(save_path)

    def predict(self, obs, env):
        if not self.model: return 0

        action_masks = env.action_masks()
        action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
        return action

    def solve(self, env, max_steps=None, reset_metrics=False):
        if max_steps is None: max_steps = self.config.MAX_INFERENCE_STEPS
        if reset_metrics: self.metrics.reset()

        obs, _ = env.reset()

        for step in range(max_steps):
            action = self.predict(obs, env)
            
            move_type = action // self.config.MAX_LEN
            self.metrics.record_step(move_type)

            next_obs, reward, terminated, truncated, info = env.step(action)
            
            obs = next_obs
            current_braid = [x for x in obs if x != 0]

            if terminated or len(current_braid) == 0:
                self.metrics.record_episode_end(success=True)
                return True, step + 1
            
            if truncated:
                break

        self.metrics.record_episode_end(success=False)
        return False, max_steps

    def save(self, path):
        if self.model: self.model.save(path)

    def load(self, path):
        self.model = MaskablePPO.load(path)
        self.name = os.path.basename(path).replace('.zip', '')