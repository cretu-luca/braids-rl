import os

class Configuration:
    N_STRANDS = 3
    
    DATA_DIR = "../data/"
    MODEL_DIR = "../model/"
    LOG_DIR = "../log/"

    @staticmethod
    def get_dataset_path(level_name):
        os.makedirs(Configuration.DATA_DIR, exist_ok=True)
        return os.path.join(Configuration.DATA_DIR, f"{level_name}.txt")

    @staticmethod
    def get_model_path(name):
        os.makedirs(Configuration.MODEL_DIR, exist_ok=True)
        return os.path.join(Configuration.MODEL_DIR, name)