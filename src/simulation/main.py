# src/data_generation/main.py

import numpy as np
import time
import pickle
import multiprocessing
from tqdm import tqdm
import warnings
import os

# Import our refactored components
import src.config as cfg
from src.simulation.rbergomi_simulator import PathSimulator
from src.benchmark.malliavin_delta import calculate_optimal_delta_mlqmc

# Ignore potential overflow warnings from numpy, as we handle them with clipping.
warnings.filterwarnings('ignore', category=RuntimeWarning)

def generate_one_sample(sample_id):
    """
    Generates a single training sample: a feature vector and its corresponding optimal delta label.
    """
    # Ensure each process has a unique random state
    np.random.seed(int.from_bytes(np.random.bytes(4), byteorder='little') + sample_id)

    simulator = PathSimulator(cfg.H, cfg.rho, cfg.v, cfg.sigma0_sq, cfg.T)
    
    # 1. Randomly sample a starting time t < T to create a conditional state F_t
    t_start = np.random.uniform(0.05, 0.95) * cfg.T
    M_hist = int(t_start / cfg.T * 100)
    M_hist = max(M_hist, 10) # Ensure a minimum number of steps

    # 2. Simulate the historical path up to t_start
    history = simulator.simulate_history(t_start, M_hist, cfg.S0)
    
    # 3. Calculate the optimal delta (this is our ground truth label)
    optimal_delta = calculate_optimal_delta_mlqmc(
        history, simulator, cfg.K, cfg.rho,
        cfg.L, cfg.M0, cfg.B, cfg.N_SAMPLES_PER_LEVEL
    )

    # 4. Construct the feature vector for the neural network
    # Feature 1: Moneyness (S_t / K)
    # Feature 2: Time to maturity (T - t)
    # Feature 3: Current volatility (sqrt(sigma_sq_t))
    # Feature 4: Summary of the historical Volterra path
    V_hist_path = history["V_hist"]
    hist_indices = np.linspace(0, len(V_hist_path)-1, cfg.NUM_HIST_FEATURES, dtype=int)
    V_hist_features = V_hist_path[hist_indices]
    
    feature_vector = np.concatenate([
        [history["S_t"] / cfg.K],
        [cfg.T - t_start],
        [np.sqrt(max(history["sigma_sq_t"], 0))],
        V_hist_features
    ])
    
    return {'features': feature_vector, 'label': optimal_delta}

if __name__ == '__main__':
    print("="*60)
    print("Starting Training Data Generation for Rough Volatility Hedging")
    print(f"Target samples: {cfg.TOTAL_SAMPLES:,}")
    print(f"Using {cfg.NUM_CORES} parallel processes.")
    print("="*60)
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    start_time = time.time()
    
    dataset = []
    
    # Use multiprocessing.Pool for parallel computation
    with multiprocessing.Pool(processes=cfg.NUM_CORES) as pool:
        with tqdm(total=cfg.TOTAL_SAMPLES, desc="Generating samples") as pbar:
            for i, result in enumerate(pool.imap_unordered(generate_one_sample, range(cfg.TOTAL_SAMPLES))):
                if result is not None:
                    dataset.append(result)
                pbar.update(1)
                
                # Periodically save chunks to disk
                if len(dataset) >= cfg.SAVE_CHUNK_SIZE:
                    chunk_num = (i + 1) // cfg.SAVE_CHUNK_SIZE
                    filename = os.path.join(cfg.OUTPUT_DIR, f'delta_hedging_dataset_chunk_{chunk_num}.pkl')
                    with open(filename, 'wb') as f:
                        pickle.dump(dataset, f)
                    print(f"\nSaved {len(dataset)} samples to {filename}")
                    dataset = [] # Clear list to free memory

    # Save any remaining data
    if dataset:
        filename = os.path.join(cfg.OUTPUT_DIR, 'delta_hedging_dataset_chunk_final.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"\nSaved final {len(dataset)} samples to {filename}")

    end_time = time.time()
    total_duration = end_time - start_time
    
    print("\n" + "="*60)
    print("Data generation complete!")
    print(f"Total time: {total_duration / 3600:.2f} hours ({total_duration:.2f} seconds)")
    print(f"Dataset saved in chunks to '{cfg.OUTPUT_DIR}' directory.")
    print("="*60)