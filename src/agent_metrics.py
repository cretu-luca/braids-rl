import json
import os
from collections import defaultdict
from .config import Configuration

class AgentMetrics:
    ACTION_NAMES = {
        0: "Commute",
        1: "R3 (Relation)",
        2: "Remove Pair",
        3: "Insert Pair"
    }

    def __init__(self, config: Configuration):
        self.config = config
        self.reset()

    def reset(self):
        self.agent_actions = defaultdict(int)
        self.rescue_actions = defaultdict(int)
        self.rescue_counts = 0
        self.episodes = 0
        self.solved = 0
        self.failed = 0
        self.total_steps = 0

    def record_step(self, action_type: int):
        """Records an action chosen by the Agent."""
        self.agent_actions[int(action_type)] += 1
        self.total_steps += 1

    def record_rescue(self, action_type: int):
        """Records a forced rescue action."""
        self.rescue_counts += 1
        self.rescue_actions[int(action_type)] += 1

    def record_episode_end(self, success: bool):
        self.episodes += 1
        if success:
            self.solved += 1
        else:
            self.failed += 1

    def print_summary(self):
        print(f"\n--- Agent Performance Summary ---")
        print(f"Episodes: {self.episodes} | Solved: {self.solved} ({self.solved/self.episodes*100 if self.episodes else 0:.1f}%)")
        print(f"Total Steps: {self.total_steps} | Rescues Triggered: {self.rescue_counts}")
        
        print("\nAgent Action Distribution:")
        total_agent = sum(self.agent_actions.values()) or 1
        for type_id, count in sorted(self.agent_actions.items()):
            name = self.ACTION_NAMES.get(type_id, "Unknown")
            pct = (count / total_agent) * 100
            print(f"  {name}: {count} ({pct:.1f}%)")

        if self.rescue_counts > 0:
            print("\nRescue Action Distribution:")
            total_rescue = sum(self.rescue_actions.values()) or 1
            for type_id, count in sorted(self.rescue_actions.items()):
                name = self.ACTION_NAMES.get(type_id, "Unknown")
                pct = (count / total_rescue) * 100
                print(f"  {name}: {count} ({pct:.1f}%)")

    def save(self, model_name: str):
        os.makedirs(self.config.METRICS_DIR, exist_ok=True)
        filepath = os.path.join(self.config.METRICS_DIR, f"{model_name}_metrics.json")
        
        data = {
            "model_name": model_name,
            "stats": {
                "episodes": self.episodes,
                "solved": self.solved,
                "total_steps": self.total_steps,
                "rescues": self.rescue_counts,
                "agent_actions": dict(self.agent_actions),
                "rescue_actions": dict(self.rescue_actions)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Metrics saved to {filepath}")