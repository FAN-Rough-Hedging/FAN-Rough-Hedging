
# FAN: Fractional Attention Networks for Optimal Hedging in Rough Volatility Models

<p align="center">
  <strong><a href="#english-version-readme">English</a></strong> |
  <strong><a href="#中文版-readme">中文</a></strong>
</p>

---

## <a name="english-version-readme"></a>English Version README

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

### Project Structure

```
.
├── Loss_Plot/            # Loss curve plots (.pdf)
├── Model_FAN/            # FAN model (fractal attention + feed-forward network)
├── Trained_Model/        # Trained model weights (.pth)
├── TrainingReport/       # Training reports (.json)
├── compare_models_file/  # Benchmark models (LSTM / Transformer)
├── utils/                # Data processing and training controller
│   └── dataset/          # Prepared datasets (.pt files)
├── main.py               # Entry script for training & validation
├── setting.py            # Model & training configurations
├── requirements.txt      # Project dependencies
└── README.md             # Documentation
```

### Quick Start

Follow these steps to set up the environment, generate the data, and train the models.

#### Step 1: Installation

Clone the repository and set up a virtual environment (Conda is recommended).

```bash
# Clone the repository
git clone https://github.com/your-username/FAN-Hedge.git
cd FAN-Hedge

# Create and activate a Conda environment
# conda create -n fan-hedge python=3.11
# conda activate fan-hedge

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Data Generation

This step generates the training dataset, which consists of market states (features) and their corresponding theoretically optimal deltas (labels). This can be time-consuming.

```bash
# This command will use parameters from setting.py to generate dataset
python utils/DataProcessor.py
```
After completion, the `utils/dataset/` directory will be populated with `.pt` files.

#### Step 3: Model Training

Run the main script to start training. All configurations, such as model selection (FAN, LSTM, Transformer), learning rate, and epochs, are managed in `setting.py`.

```bash
# Modify settings in setting.py as needed, then run:
python main.py
```
The training process will:
- Load configurations from `setting.py`.
- Load a pre-trained model from `Trained_Model/` if it exists.
- Run training and validation loops.
- Save the trained model weights to `Trained_Model/`.
- Generate a training report in `TrainingReport/`.
- Save loss curve plots to `Loss_Plot/`.

### Configuration

All training and model parameters can be customized in **`setting.py`**:
- **Model Selection**: `FAN_SETTING`, `LSTM_SETTING`, `VANILLAT_TRANSFORMER_SETTING`.
- **Optimizer**: `Adam`, `AdamW`, `SGD`, etc., along with hyperparameters like `LR` and `WEIGHT_DECAY`.
- **LR Scheduler**: `StepLR`, `CosineAnnealingLR`, etc.
- **Training Hyperparameters**: `EPOCHS`, `BATCH_SIZE`.

### Key Features

- **Training Reports (JSON)**: After each run, a detailed JSON report is saved in `TrainingReport/`, containing hyperparameters, training time, and validation loss statistics.
- **Loss Visualization**: Loss curves are automatically saved as PDF files in `Loss_Plot/` for easy analysis.
- **Model Checkpointing**: Models are saved to `Trained_Model/` and can be loaded automatically to resume training or for evaluation.

### FAQ

- **Weight file not found**: Ensure the model filename in `Trained_Model/` matches the name specified in `setting.py`.
- **GPU Out of Memory (OOM)**: Reduce `BATCH_SIZE` in `setting.py` or use a smaller dataset.
- **"Trying to backward through the graph a second time"**: This error typically occurs with incorrect gradient handling. If you are implementing gradient accumulation, ensure you call `(loss / accumulation_steps).backward()` correctly and manage `optimizer.zero_grad()` and `optimizer.step()` timing.

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

### 项目结构

```
.
├── Loss_Plot/            # 损失曲线图 (.pdf)
├── Model_FAN/            # FAN 模型 (分数注意力网络 + 前馈网络)
├── Trained_Model/        # 训练好的模型权重 (.pth)
├── TrainingReport/       # 训练报告 (.json)
├── compare_models_file/  # 基准模型 (LSTM / Transformer)
├── utils/                # 数据处理和训练控制器
│   └── dataset/          # 准备好的数据集 (.pt 文件)
├── main.py               # 训练与验证的入口脚本
├── setting.py            # 模型与训练的配置文件
├── requirements.txt      # 项目依赖
└── README.md             # 项目文档
```

### 快速开始

请遵循以下步骤来配置环境、生成数据并训练模型。

#### 第 1 步：安装依赖

克隆本代码库，并配置虚拟环境（推荐使用 Conda）。

```bash
# 克隆仓库
git clone https://github.com/your-username/FAN-Hedge.git
cd FAN-Hedge

# 创建并激活 Conda 环境
# conda create -n fan-hedge python=3.11
# conda activate fan-hedge

# 安装依赖
pip install -r requirements.txt
```

#### 第 2 步：生成数据

此步骤会生成训练所需的数据集，包含市场状态（特征）和对应的理论最优 Delta（标签），可能比较耗时。

```bash
# 此命令将使用 setting.py 中的参数生成数据集
python utils/DataProcessor.py
```
运行结束后，`utils/dataset/` 目录下将生成 `.pt` 数据文件。

#### 第 3 步：训练模型

运行主脚本即可开始训练。所有配置项，如模型选择（FAN、LSTM、Transformer）、学习率、训练轮数等，均在 `setting.py` 文件中统一管理。

```bash
# 根据需要修改 setting.py 中的配置，然后运行：
python main.py
```
训练流程将自动完成以下任务：
- 从 `setting.py` 加载配置。
- 如果 `Trained_Model/` 目录下存在预训练模型，则加载权重。
- 执行训练和验证循环。
- 将训练好的模型权重保存到 `Trained_Model/`。
- 在 `TrainingReport/` 目录中生成训练报告。
- 在 `Loss_Plot/` 目录中保存损失曲线图。

### 参数配置

所有的训练和模型参数都可以在 **`setting.py`** 文件中修改：
- **模型选择**: `FAN_SETTING`, `LSTM_SETTING`, `VANILLAT_TRANSFORMER_SETTING`。
- **优化器**: `Adam`, `AdamW`, `SGD` 等，以及 `LR` (学习率) 和 `WEIGHT_DECAY` (权重衰减) 等超参数。
- **学习率调度器**: `StepLR`, `CosineAnnealingLR` 等。
- **训练超参数**: `EPOCHS` (训练轮数), `BATCH_SIZE` (批大小)。

### 主要功能

- **训练报告 (JSON)**: 每次训练后，详细的 JSON 报告会保存在 `TrainingReport/` 中，记录了超参数、训练耗时和验证集损失统计信息。
- **损失可视化**: 训练过程中的损失曲线会自动保存为 PDF 文件于 `Loss_Plot/` 目录，方便分析。
- **模型断点续训**: 模型权重保存在 `Trained_Model/`，程序会自动加载已有权重，方便继续训练或进行评估。

### 常见问题 (FAQ)

- **找不到权重文件**: 请确保 `Trained_Model/` 中的模型文件名与 `setting.py` 中指定的名称一致。
- **GPU 显存不足 (OOM)**: 在 `setting.py` 中减小 `BATCH_SIZE` 或使用规模更小的数据集。
- **"Trying to backward through the graph a second time"**: 这个错误通常是由于不正确的梯度计算图操作导致。如果你在实现梯度累积，请确保正确调用 `(loss / accumulation_steps).backward()` 并控制好 `optimizer.zero_grad()` 和 `optimizer.step()` 的调用时机。

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
```
