import numpy as np
import os
from sb3_contrib import MaskablePPO
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

    def train(self, env, total_timesteps, save_path=None, callback=None, log_name=None):
        env = ActionMasker(env, mask_fn)

        if log_name is None:
            log_name = self.name

        if self.model is None:
            self.model = MaskablePPO("MlpPolicy", env, **self.hyperparameters)
        else:
            self.model.set_env(env)

        print(f"[{self.name}] Training for {total_timesteps} steps...")
        
        self.model.learn(
            total_timesteps=total_timesteps, 
            callback=callback, 
            tb_log_name=log_name, 
            reset_num_timesteps=False
        )
        
        if save_path:
            self.save(save_path)

    def predict(self, obs, env):
        if not self.model: return 0

        action_masks = env.action_masks()
        action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
        return action

    def solve(self, env, max_steps=None, reset_metrics=False, verbose=False):
        if max_steps is None: max_steps = self.config.MAX_INFERENCE_STEPS
        if reset_metrics: self.metrics.reset()

        obs, _ = env.reset()
        
        initial_braid = [x for x in obs if x != 0]
        if verbose: 
            print(f"Solving: {initial_braid}")

        for step in range(max_steps):
            action = self.predict(obs, env)
            
            move_type = action // self.config.MAX_LEN
            self.metrics.record_step(move_type)

            next_obs, reward, terminated, truncated, info = env.step(action)
            
            obs = next_obs
            current_braid = [x for x in obs if x != 0]

            if verbose and step % 10 == 0:
                 print(f"  Step {step}: Len {len(current_braid)} - Last Move: {info.get('move_name', move_type)}")

            if terminated or len(current_braid) == 0:
                self.metrics.record_episode_end(success=True)
                if verbose: 
                    print(f"  SOLVED in {step+1} steps!")
                return True, step + 1
            
            if truncated:
                break

        self.metrics.record_episode_end(success=False)
        if verbose: 
            print(f"  FAILED (Max steps {max_steps})")
        return False, max_steps

    def save(self, path):
        if self.model: 
            self.model.save(path)
            self.name = os.path.basename(path).replace('.zip', '')

    def load(self, path):
        self.model = MaskablePPO.load(path)
        self.name = os.path.basename(path).replace('.zip', '')