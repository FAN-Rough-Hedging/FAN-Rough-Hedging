# FAN: Fractional Attention Networks for Optimal Hedging in Rough Volatility Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-24XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/24XX.XXXXX) <!-- 论文上线后替换 -->

This repository contains the official PyTorch implementation for the paper: **"FAN: Optimal Delta Hedging in Rough Volatility Models with Fractional Attention Neural Networks"** (AAAI/NeurIPS/ICML 2025).

Our work introduces **FAN**, a novel neural network architecture with a hard-coded inductive bias that captures the power-law memory inherent in rough volatility models. FAN achieves the theoretically optimal convergence rate for hedging error, demonstrating superior performance and sample efficiency compared to standard sequential models like LSTMs and Transformers.

<p align="center">
  <img src="assets/fan_architecture.png" width="700"> <!-- 在 assets 文件夹放一张模型结构图 -->
  <br>
  <em>Figure 1: The architecture of our Fractional Attention Network (FAN).</em>
</p>

## Key Contributions

1.  **Theoretical Analysis**: We derive the precise power-law kernel `(t-s)^(H-3/2)` that governs the dynamics of the optimal hedging strategy in rough volatility markets (Proposition 2).
2.  **Novel Architecture (FAN)**: We propose a Fractional Attention Network (FAN) that embeds this kernel as an inductive bias, enabling efficient learning of long-range, power-law dependencies.
3.  **State-of-the-Art Performance**: We demonstrate that FAN achieves the theoretically tight Mean Square Hedge Error (MSHE) bound of `Θ(N^(2H-1))`, significantly outperforming existing benchmarks.

## Repository Structure

```
.
├── environment.yml      # Conda environment for reproducibility
├── notebooks/           # Notebooks for visualization and analysis
├── paper/               # Paper LaTeX source
├── scripts/             # End-to-end training and evaluation scripts
└── src/                 # Main source code
    ├── simulation/      # Rough volatility model simulator
    ├── models/          # FAN, baselines (LSTM, Transformer)
    ├── benchmark/       # Malliavin weight-based "ground truth" delta
    ├── training/        # Training loops and utilities
    ├── evaluation/      # Evaluation metrics and plotting
    └── utils/           # Shared utilities (config, logging)
```

## Getting Started

### 1. Installation

Clone the repository and set up the Conda environment:

```bash
git clone https://github.com/your-username/FAN-Hedge.git
cd FAN-Hedge
conda env create -f environment.yml
conda activate fan-hedge
```

### 2. Data Simulation

First, simulate the financial market paths based on the Rough Volatility model. Configurations can be found in `src/simulation/config.py`.

```bash
python src/simulation/main.py --hurst 0.1 --rho -0.7 --num-paths 100000
```

### 3. Running Experiments

We provide simple scripts to replicate our main results.

**Train FAN Model:**
```bash
bash scripts/train_fan.sh
```

**Evaluate all models and generate plots:**
```bash
bash scripts/evaluate_all.sh
```

This will train the models, calculate the Mean Square Hedge Error (MSHE), and save the final plots (like the convergence rate plot) in the `results/` directory.

## Citation

If you find our work useful, please cite our paper:

```bibtex
@inproceedings{yourname2025fan,
  title={FAN: Optimal Delta Hedging in Rough Volatility Models with Fractional Attention Neural Networks},
  author={Your Name and Co-authors},
  booktitle={Conference Name},
  year={2025}
}
```

## Contact

For any questions, please open an issue or contact [your-email@example.com].
