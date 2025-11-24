from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from collections import defaultdict

class BraidCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(BraidCallback, self).__init__(verbose)
        self.episode_successes = []
        self.action_counts = defaultdict(int)
        self.total_actions = 0

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'move_type' in info:
                move = info['move_type']
                self.action_counts[move] += 1
                self.total_actions += 1

        for i, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][i]
                is_success = info.get("is_success", False)
                self.episode_successes.append(is_success)

        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_successes) > 0:
            success_rate = np.mean(self.episode_successes)
            self.logger.record("custom/success_rate", success_rate)
            self.episode_successes = []

        if self.total_actions > 0:
            self.logger.record("actions/commute_ratio", self.action_counts[0] / self.total_actions)
            self.logger.record("actions/r3_ratio", self.action_counts[1] / self.total_actions)
            self.logger.record("actions/remove_ratio", self.action_counts[2] / self.total_actions)
            self.logger.record("actions/insert_ratio", self.action_counts[3] / self.total_actions)
            
            self.action_counts = defaultdict(int)
            self.total_actions = 0