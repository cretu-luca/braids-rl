import numpy as np
from .agent_metrics import AgentMetrics

class BraidAgent:
    def __init__(self, config, algorithm_class, hyperparameters, model_path=None):
        self.config = config
        self.algorithm_class = algorithm_class
        self.hyperparameters = hyperparameters
        self.model = None
        
        self.metrics = AgentMetrics(config)

        if model_path:
            self.load(model_path)

    def train(self, env, total_timesteps, save_path=None):
        if self.model is None:
            self.model = self.algorithm_class("MlpPolicy", env, **self.hyperparameters)
        else:
            self.model.set_env(env)

        self.model.learn(total_timesteps=total_timesteps)

        if save_path:
            self.save(save_path)

    def predict(self, obs, deterministic=True):
        if not self.model:
            return [0, 0]
        action, _ = self.model.predict(obs, deterministic=False)
        return action

    def solve(self, env, max_steps: int = None, reset_metrics: bool = False):
        if max_steps is None:
            max_steps = self.config.MAX_INFERENCE_STEPS

        if reset_metrics:
            self.metrics.reset()

        obs, _ = env.reset()

        for step in range(max_steps):
            action = self.predict(obs)
            self.metrics.record_step(action[0])

            next_obs, _, terminated, truncated, info = env.step(action)

            if not info['success']:
                valid_len = np.count_nonzero(obs[:-1])
                limit = max(1, valid_len - 1)

                rescue_type = np.random.choice([0, 1])
                rescue_idx = np.random.randint(0, limit) 
                rescue_action = np.array([rescue_type, rescue_idx])

                self.metrics.record_rescue(rescue_type)

                next_obs, reward, terminated, truncated, info = env.step(rescue_action)

            obs = next_obs
            current_braid = [x for x in obs[:-1] if x != 0]

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
        self.model = self.algorithm_class.load(path)