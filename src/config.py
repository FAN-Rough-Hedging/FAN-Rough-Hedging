# src/config.py

"""
Global configuration for the FAN-Hedge project.
Centralizes all parameters for simulation, option pricing, and model training.
"""

# ==============================================================================
# 1. Rough Bergomi Model Parameters
# ==============================================================================
# Corresponds to model definitions in the paper (e.g., Definition 2)
H = 0.07          # Hurst index (H < 0.5 for roughness)
rho = -0.7        # Correlation between price and volatility noise sources
v = 1.8           # Volatility of volatility (eta in some literature)
sigma0_sq = 0.04  # Initial instantaneous variance (xi_0)

# ==============================================================================
# 2. Option & Market Parameters
# ==============================================================================
T = 1.0           # Time to maturity (in years)
K = 1.0           # Strike price
S0 = 1.0          # Initial stock price, used for normalization

# ==============================================================================
# 3. Benchmark Delta Calculation Parameters (MLQMC Engine)
# ==============================================================================
L = 5             # Max level for MLQMC (levels 0, 1, ..., L)
M0 = 16           # Number of steps on the coarsest level
B = 2             # Refinement factor for steps between levels (B=2 -> doubles steps)
# Optimized sample allocation per level for variance reduction
N_SAMPLES_PER_LEVEL = [131072, 32768, 8192, 2048, 512, 128] 

# ==============================================================================
# 4. Data Generation Parameters
# ==============================================================================
TOTAL_SAMPLES = 1_000_000      # Total number of (feature, label) pairs to generate
NUM_HIST_FEATURES = 16         # Number of historical points of the Volterra process to use as features
SAVE_CHUNK_SIZE = 5000         # Save data to disk every N samples to manage memory
OUTPUT_DIR = "data/"           # Directory to save generated datasets
NUM_CORES = 12                 # Number of CPU cores for parallel generation. Set to multiprocessing.cpu_count() for max.