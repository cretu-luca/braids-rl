import os

class Configuration:
    def __init__(self, n_strands = 3, max_len = 20, learning_rate = 0.0003, entropy_coef = 0.01, total_timesteps = 300_000, 
                 reward_step = -0.05, reward_invalid = -1.0, reward_loop = -2.0, reward_solved = 20.0, reward_shrink = 1.0,
                 reward_grow = -1.0, max_inference_steps = 50, DATA_DIR = "./data/", MODEL_DIR = "./model/", LOG_DIR = "./log/"):
        
        self.DATA_DIR = DATA_DIR
        self.MODEL_DIR = MODEL_DIR
        self.LOG_DIR = LOG_DIR

        self.N_STRANDS = n_strands
        self.MAX_LEN = max_len
        self.LEARNING_RATE = learning_rate
        self.ENTROPY_COEF = entropy_coef

        self.TOTAL_TIMESTEPS = total_timesteps
        
        self.REWARD_STEP = reward_step
        self.REWARD_INVALID = reward_invalid
        self.REWARD_LOOP = reward_loop
        self.REWARD_SOLVED = reward_solved
        self.REWARD_SHRINK = reward_shrink
        self.REWARD_GROW = reward_grow

        self.MAX_INFERENCE_STEPS = max_inference_steps

    def get_dataset_path(self, level_name):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        return os.path.join(self.DATA_DIR, f"{level_name}.txt")

    def get_model_path(self, name):
        os.makedirs(Configuration.MODEL_DIR, exist_ok=True)
        return os.path.join(Configuration.MODEL_DIR, name)