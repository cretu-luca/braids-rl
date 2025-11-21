import os

class Configuration:
    N_STRANDS = 3
    MAX_LEN = 20
    
    DATA_DIR = "./data/"
    MODEL_DIR = "./model/"
    LOG_DIR = "./log/"

    LEARNING_RATE = 0.0003
    ENTROPY_COEF = 0.01
    TOTAL_TIMESTEPS = 300_000
    
    REWARD_STEP = -0.05
    REWARD_INVALID = -1.0
    REWARD_LOOP = -2.0
    REWARD_SOLVED = 20.0
    REWARD_SHRINK = 1.0
    REWARD_GROW = -1.0

    MAX_INFERENCE_STEPS = 50

    @staticmethod
    def get_dataset_path(level_name):
        os.makedirs(Configuration.DATA_DIR, exist_ok=True)
        return os.path.join(Configuration.DATA_DIR, f"{level_name}.txt")

    @staticmethod
    def get_model_path(name):
        os.makedirs(Configuration.MODEL_DIR, exist_ok=True)
        return os.path.join(Configuration.MODEL_DIR, name)