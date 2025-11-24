import os
import sys
import concurrent.futures
import time

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import Configuration
from src.braid_generator import BraidGenerator

def generate_task(args):
    n_strands, crossings, difficulty, count, filepath, compute_optimal, seed = args
    
    try:
        config = Configuration()
        gen = BraidGenerator(n_strands=n_strands, config=config, seed=seed)
        
        if os.path.exists(filepath):
            return f"[SKIP] {os.path.basename(filepath)} exists"

        gen.generate_dataset(
            count=count, 
            crossings=crossings, 
            difficulty=difficulty, 
            filepath=filepath,
            compute_optimal=compute_optimal
        )
        return f"[DONE] {os.path.basename(filepath)}"
    except Exception as e:
        return f"[FAIL] {os.path.basename(filepath)}: {str(e)}"

def generate_all_datasets_parallel():
    config = Configuration()
    
    crossings_list = [8, 16, 24]
    moves_list = [10, 50, 100]
    
    tasks = []
    
    print(f"--- Preparing Training Tasks ---")
    TRAIN_COUNT = 5000
    train_strands = [3, 5, 7]
    
    for i, st in enumerate(train_strands):
        for j, cr in enumerate(crossings_list):
            for k, mv in enumerate(moves_list):
                filename = f"train_n{st}_c{cr}_m{mv}"
                path = os.path.join(config.DATA_DIR, "train", f"{filename}.txt")
                
                seed = 42 + (i * 100) + (j * 10) + k
                tasks.append((st, cr, mv, TRAIN_COUNT, path, False, seed))

    print(f"--- Preparing Fine-Tuning Tasks ---")
    FINETUNE_COUNT = 50
    
    for i, st in enumerate(train_strands):
        for j, cr in enumerate(crossings_list):
            for k, mv in enumerate(moves_list):
                filename = f"ft_n{st}_c{cr}_m{mv}"
                path = os.path.join(config.DATA_DIR, "finetune", f"{filename}.txt")
                
                seed = 9999 + (i * 100) + (j * 10) + k
                tasks.append((st, cr, mv, FINETUNE_COUNT, path, True, seed))

    print(f"--- Preparing Test Tasks ---")
    TEST_COUNT = 100
    test_strands = [3, 5, 7, 9]
    
    for i, st in enumerate(test_strands):
        for j, cr in enumerate(crossings_list):
            for k, mv in enumerate(moves_list):
                filename = f"test_n{st}_c{cr}_m{mv}"
                path = os.path.join(config.DATA_DIR, "test", f"{filename}.txt")
                
                seed = 1337 + (i * 100) + (j * 10) + k
                tasks.append((st, cr, mv, TEST_COUNT, path, True, seed))

    os.makedirs(os.path.join(config.DATA_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(config.DATA_DIR, "finetune"), exist_ok=True)
    os.makedirs(os.path.join(config.DATA_DIR, "test"), exist_ok=True)

    print(f"\nStarting Parallel Generation of {len(tasks)} datasets...")
    print(f"Using all available CPU cores.")
    
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(generate_task, task) for task in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    duration = time.time() - start_time
    print(f"\n--- COMPLETE in {duration:.2f} seconds ---")

if __name__ == "__main__":
    generate_all_datasets_parallel()