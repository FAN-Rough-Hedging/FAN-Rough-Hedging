# FAN: Fractional Attention Networks for Optimal Hedging in Rough Volatility Models

<p align="center">
  <strong><a href="#english-version-readme">English</a></strong> |
  <strong><a href="#中文版-readme">中文</a></strong>
</p>

---

## English Version README

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-24XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/24XX.XXXXX) <!-- TODO: Replace with your paper's arXiv link -->

This repository contains the official PyTorch implementation for the paper: **"FAN: Optimal Delta Hedging in Rough Volatility Models with Fractional Attention Neural Networks"** (AAAI/NeurIPS/ICML 2025).

Our work introduces **FAN (Fractional Attention Network)**, a novel neural network architecture with a hard-coded inductive bias that captures the power-law memory inherent in rough volatility models. FAN achieves the theoretically optimal convergence rate for hedging error, demonstrating superior performance and sample efficiency compared to standard sequential models like LSTMs and Transformers.

<p align="center">
  <img src="assets/fan_architecture.png" width="700" alt="FAN Architecture Diagram"> 
  <br>
  <em>Figure 1: The architecture of our Fractional Attention Network (FAN), which incorporates a fixed power-law kernel.</em>
</p>

### Key Contributions

1.  **Dominant Power-Law Kernel**: We prove through Malliavin calculus that the optimal hedging strategy in rough volatility markets is dominated by a deterministic integral kernel with a power-law singularity `(t-s)^(H-3/2)` (Proposition 2).
2.  **Novel Architecture (FAN)**: We propose a Fractional Attention Network (FAN) that hard-codes this kernel as an inductive bias, enabling efficient learning of long-range, power-law dependencies that standard architectures struggle to capture.
3.  **State-of-the-Art Performance**: We demonstrate that FAN achieves the theoretically tight Mean Square Hedge Error (MSHE) bound of `Θ(N^(2H-1))`, significantly outperforming existing benchmarks and closing the gap between theory and practice.

## Easy Access

(training & validation) run main.py 
(parameters modify) modify setting.py


### How to Replicate Our Results

Follow these steps to set up the environment, generate the data, and train the models.

#### Step 1: Installation

Clone the repository and set up the Conda environment.

```bash
git clone https://github.com/your-username/FAN-Hedge.git
cd FAN-Hedge
# It's recommended to create a Conda or venv environment
# conda create -n fan-hedge python=3.9
# conda activate fan-hedge
pip install -r requirements.txt # Make sure to create a requirements.txt file
```

#### Step 2: Data Generation

This is the most time-consuming step. We use multiprocessing to generate the training dataset, which consists of market states (features) and their corresponding theoretically optimal deltas (labels) computed via MLQMC.

```bash
# This command will use the parameters from src/config.py
# It may take several hours depending on your machine and TOTAL_SAMPLES
python -m src.data_generation.main
```
After completion, the `data/` directory will be populated with `delta_hedging_dataset_chunk_*.pkl` files.

#### Step 3: Model Training

Now, you can train the FAN model and the baseline models.

**Train our FAN Model:**
```bash
python train.py --model FAN --lr 0.0001 --batch_size 256 --epochs 50
```

**Train the LSTM Baseline:**
```bash
python train.py --model LSTM --lr 0.0001 --batch_size 256 --epochs 50
```
Trained models will be saved in the `trained_models/` directory.

#### Step 4: Evaluation (Coming Soon)
The `evaluate.py` script will load the trained models and generate the final plots and metrics presented in the paper, such as the MSHE convergence plot.

### Citation

If you find our work useful, please cite our paper:

```bibtex
@inproceedings{yourname2025fan,
  title={FAN: Optimal Delta Hedging in Rough Volatility Models with Fractional Attention Neural Networks},
  author={Your Name and Co-authors},
  booktitle={Conference Name},
  year={2025}
}
```

---

## <a name="中文版-readme"></a>中文版 README

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-24XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/24XX.XXXXX) <!-- TODO: 替换为您的论文 arXiv 链接 -->

本代码库是论文 **《FAN: 基于分数注意力网络的粗糙波动率模型最优Delta对冲策略》** (AAAI/NeurIPS/ICML 2025) 的官方 PyTorch 实现。

我们的工作引入了 **FAN (分数注意力网络)**，这是一种新颖的神经网络架构。它通过硬编码的归纳偏置 (inductive bias) 来捕捉粗糙波动率模型中固有的幂律记忆特性。实验证明，FAN 能够达到理论上最优的对冲误差收敛速率，其性能和样本效率均显著优于 LSTM 和 Transformer 等标准序列模型。

<p align="center">
  <img src="assets/fan_architecture.png" width="700" alt="FAN 架构图"> 
  <br>
  <em>图 1: 我们的分数注意力网络 (FAN) 架构，其核心是集成了一个固定的幂律核。</em>
</p>

### 核心贡献

1.  **确定主导幂律核**: 我们通过 Malliavin 导数分析，证明了在粗糙波动率市场中，最优对冲策略由一个具有幂律奇异性 `(t-s)^(H-3/2)` 的确定性积分核主导 (论文命题2)。
2.  **提出 FAN 架构**: 我们设计了一种分数注意力网络 (FAN)，将上述幂律核作为归纳偏置硬编码到模型中，使其能高效学习标准架构难以捕捉的长程幂律依赖关系。
3.  **达到理论最优性能**: 我们证明了 FAN 能够达到理论上最紧的均方对冲误差 (MSHE) 界 `Θ(N^(2H-1))`，显著超越了现有基准模型，弥合了金融理论与深度学习实践之间的鸿沟。

### 代码库结构

```
.
├── data/                      # 用于存放生成的数据集
├── trained_models/            # 用于存放训练好的模型权重
├── train.py                   # 训练模型的主脚本
└── src/                       # 所有源代码
    ├── config.py              # 全局配置文件
    ├── simulation/            # 粗糙波动率模型模拟器
    ├── data_generation/       # 生成训练数据集的脚本
    ├── benchmark/             # 基于 Malliavin 权重的“真实” Delta 计算
    ├── models/                # FAN 及基准模型 (LSTM, Transformer)
    └── training/              # 数据集、损失函数等
```

### 如何复现我们的结果

请遵循以下步骤来配置环境、生成数据并训练模型。

#### 第 1 步：安装依赖

克隆本代码库，并配置 Conda 环境。

```bash
git clone https://github.com/your-username/FAN-Hedge.git
cd FAN-Hedge
# 建议创建一个 Conda 或 venv 虚拟环境
# conda create -n fan-hedge python=3.9
# conda activate fan-hedge
pip install -r requirements.txt # 请确保创建 requirements.txt 文件
```

#### 第 2 步：生成数据

这是最耗时的一步。我们利用多进程并行生成训练数据集，其中包含市场状态（特征）和通过 MLQMC 方法计算出的理论最优 Delta（标签）。

```bash
# 此命令将使用 src/config.py 中的参数
# 根据您的机器性能和 TOTAL_SAMPLES 设置，可能需要数小时
python -m src.data_generation.main
```
运行结束后，`data/` 目录下将生成 `delta_hedging_dataset_chunk_*.pkl` 文件。

#### 第 3 步：训练模型

现在，您可以开始训练 FAN 模型和基准模型。

**训练我们的 FAN 模型：**
```bash
python train.py --model FAN --lr 0.0001 --batch_size 256 --epochs 50
```

**训练 LSTM 基准模型：**
```bash
python train.py --model LSTM --lr 0.0001 --batch_size 256 --epochs 50
```
训练好的模型权重将保存在 `trained_models/` 目录下。

#### 第 4 步：评估 (即将推出)
`evaluate.py` 脚本将加载训练好的模型，并生成论文中展示的关键图表和指标，例如 MSHE 收敛速率图。

### 引用

如果我们的工作对您有帮助，请引用我们的论文：

```bibtex
@inproceedings{yourname2025fan,
  title={FAN: Optimal Delta Hedging in Rough Volatility Models with Fractional Attention Neural Networks},
  author={Your Name and Co-authors},
  booktitle={Conference Name},
  year={2025}
}
```
