
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

This project provides simulation and model training for high-frequency option hedging. It includes our primary **Fractal Attention Network (FAN)**, along with **LSTM** and a **vanilla Transformer** for benchmarking. The project supports GPU acceleration, dataset generation, training reports in JSON, and loss visualization.

<p align="center">
  <img src="assets/fan_architecture.png" width="700" alt="FAN Architecture Diagram"> 
  <br>
  <em>Figure 1: The architecture of our Fractional Attention Network (FAN), which incorporates a fixed power-law kernel.</em>
</p>

### Key Contributions

1.  **Dominant Power-Law Kernel**: We prove through Malliavin calculus that the optimal hedging strategy in rough volatility markets is dominated by a deterministic integral kernel with a power-law singularity `(t-s)^(H-3/2)` (Proposition 2).
2.  **Novel Architecture (FAN)**: We propose a Fractional Attention Network (FAN) that hard-codes this kernel as an inductive bias, enabling efficient learning of long-range, power-law dependencies that standard architectures struggle to capture.
3.  **State-of-the-Art Performance**: We demonstrate that FAN achieves the theoretically tight Mean Square Hedge Error (MSHE) bound of `Θ(N^(2H-1))`, significantly outperforming existing benchmarks and closing the gap between theory and practice.

### Project Structure

```
.
├── Loss_Plot/              # Loss curve plots (.pdf)
├── Model_FAN/              # FAN model (fractal attention + feed-forward network)
├── Trained_Model/          # Trained model weights (.pth)
├── TrainingReport/         # Training reports (JSON)
├── compare_models_file/    # Benchmark models (LSTM / Transformer)
├── utils/                  # Data processing and training controller
│   ├── dataset/            # Prepared datasets (.pt files)
│   └── DataProcessor.py    # Script to generate datasets
├── main.py                 # Entry script for training & validation
├── setting.py              # Model & training configurations
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```

### Requirements & Installation

-   **Environment**: A Conda virtual environment is recommended.
-   **Python**: Python 3.11+
-   **Hardware**: CUDA-enabled GPU for acceleration (optional but highly recommended).

**Installation Steps:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/FAN-Hedge.git
    cd FAN-Hedge
    ```
2.  Create and activate a Conda environment:
    ```bash
    conda create -n fan-hedge python=3.11
    conda activate fan-hedge
    ```
3.  Install dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, install major libraries manually via Conda:
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install numpy pandas matplotlib tqdm
    ```

### Quick Start

Follow these steps to run the training and validation pipeline.

#### Step 1: Generate Dataset

If the dataset is not already present under `utils/dataset/`, run the data processor script. This will generate the necessary `.pt` files.

```bash
python utils/DataProcessor.py
```

#### Step 2: Run Training & Validation

Execute the main script. This will handle the entire training and validation process based on the configurations in `setting.py`.

```bash
python main.py
```

By default, the script will:
- Initialize the training-validation process.
- Load pre-trained model weights from `Trained_Model/` if specified and available.
- Run the training and validation loops.
- Save the best model weights to `Trained_Model/`.
- Generate a JSON training report in `TrainingReport/`.
- Plot and save the loss curve in `Loss_Plot/`.


### Configuration & Customization

All model, training, and data parameters can be modified in **`setting.py`**.

-   **Model Selection**: Change the active model via `FAN_SETTING`, `LSTM_SETTING`, etc.
-   **Hyperparameters**: Adjust optimizer (`Adam`, `AdamW`), learning rate (`LR`), weight decay (`WEIGHT_DECAY`), etc.
-   **LR Scheduler**: Configure learning rate schedulers like `StepLR` or `CosineAnnealingLR`.
-   **Training Settings**: Set the number of `EPOCHS` and `BATCH_SIZE`.

### Model Saving & Loading

-   Trained model weights are automatically saved to the `Trained_Model/` directory.
-   The `main.py` script can load existing weights to resume training or for evaluation. When loading, it's safer to use `weights_only=True`:
    ```python
    # Example from main.py
    state_dict = torch.load(path, weights_only=True)
    model.load_state_dict(state_dict)
    ```

### Training Report & Visualization

-   **JSON Report**: After each run, a detailed report is saved in `TrainingReport/`, containing timestamps, hyperparameters, and validation results (mean/std loss).
-   **Loss Curves**: Training and validation loss curves are plotted and saved as PDF files in the `Loss_Plot/` directory.

### Citation

If you find our work useful for your research, please cite our paper:

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

本项目提供了高频期权对冲的模拟与模型训练。代码库包含了我们的核心模型 **分数注意力网络 (FAN)**，以及用于性能比较的 **LSTM** 和 **标准 Transformer**。项目支持 GPU 加速、数据集生成、JSON 格式的训练报告和损失可视化。

<p align="center">
  <img src="assets/fan_architecture.png" width="700" alt="FAN 架构图"> 
  <br>
  <em>图 1: 我们的分数注意力网络 (FAN) 架构，其核心是集成了一个固定的幂律核。</em>
</p>

### 核心贡献

1.  **确定主导幂律核**: 我们通过 Malliavin 导数分析，证明了在粗糙波动率市场中，最优对冲策略由一个具有幂律奇异性 `(t-s)^(H-3/2)` 的确定性积分核主导 (论文命题2)。
2.  **提出 FAN 架构**: 我们设计了一种分数注意力网络 (FAN)，将上述幂律核作为归纳偏置硬编码到模型中，使其能高效学习标准架构难以捕捉的长程幂律依赖关系。
3.  **达到理论最优性能**: 我们证明了 FAN 能够达到理论上最紧的均方对冲误差 (MSHE) 界 `Θ(N^(2H-1))`，显著超越了现有基准模型，弥合了金融理论与深度学习实践之间的鸿沟。

### 项目结构

```
.
├── Loss_Plot/              # 损失曲线图 (.pdf)
├── Model_FAN/              # FAN 模型 (分数注意力网络 + 前馈网络)
├── Trained_Model/          # 训练好的模型权重 (.pth)
├── TrainingReport/         # 训练报告 (JSON)
├── compare_models_file/    # 基准模型 (LSTM / Transformer)
├── utils/                  # 数据处理与训练控制器
│   ├── dataset/            # 预处理好的数据集 (.pt)
│   └── DataProcessor.py    # 用于生成数据集的脚本
├── main.py                 # 训练与验证的入口脚本
├── setting.py              # 模型与训练参数配置
├── requirements.txt        # 项目依赖
└── README.md               # 项目文档
```

### 环境要求与安装

-   **环境**: 建议使用 Conda 虚拟环境。
-   **Python**: Python 3.11+
-   **硬件**: 支持 CUDA 的 GPU (可选，但强烈推荐以加速训练)。

**安装步骤:**

1.  克隆本仓库：
    ```bash
    git clone https://github.com/your-username/FAN-Hedge.git
    cd FAN-Hedge
    ```
2.  创建并激活 Conda 环境：
    ```bash
    conda create -n fan-hedge python=3.11
    conda activate fan-hedge
    ```
3.  通过 `requirements.txt` 安装依赖：
    ```bash
    pip install -r requirements.txt
    ```
    或者，通过 Conda 手动安装主要库：
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install numpy pandas matplotlib tqdm
    ```

### 快速开始

请遵循以下步骤来运行完整的训练和验证流程。

#### 第 1 步：生成数据集

如果 `utils/dataset/` 目录下不存在数据集文件，请先运行数据处理脚本以生成所需的 `.pt` 文件。

```bash
python utils/DataProcessor.py
```

#### 第 2 步：运行训练与验证

执行主脚本。该脚本将根据 `setting.py` 文件中的配置，自动完成整个训练和验证流程。

```bash
python main.py
```
默认情况下，该脚本会：
- 初始化训练和验证流程。
- 如果指定且存在，从 `Trained_Model/` 加载预训练模型。
- 执行训练和验证循环。
- 将最优模型权重保存到 `Trained_Model/`。
- 在 `TrainingReport/` 目录中生成 JSON 格式的训练报告。
- 在 `Loss_Plot/` 目录中绘制并保存损失曲线图。

### 参数配置

所有模型、训练和数据相关的参数都可以在 **`setting.py`** 文件中进行修改。

-   **模型选择**: 通过 `FAN_SETTING`, `LSTM_SETTING` 等配置来切换使用的模型。
-   **超参数**: 调整优化器 (`Adam`, `AdamW`)、学习率 (`LR`)、权重衰减 (`WEIGHT_DECAY`) 等。
-   **学习率调度器**: 配置 `StepLR` 或 `CosineAnnealingLR` 等学习率调度策略。
-   **训练设置**: 设置训练轮数 `EPOCHS` 和批量大小 `BATCH_SIZE`。

### 模型保存与加载

-   训练好的模型权重会自动保存在 `Trained_Model/` 目录下。
-   `main.py` 脚本能够加载已有的权重文件以继续训练或进行评估。为安全起见，加载时推荐使用 `weights_only=True` 参数：
    ```python
    # main.py 中的示例
    state_dict = torch.load(path, weights_only=True)
    model.load_state_dict(state_dict)
    ```

### 训练报告与可视化

-   **JSON 报告**: 每次运行结束后，详细的报告将保存在 `TrainingReport/` 目录中，内容包括时间戳、超参数配置和验证集损失（均值/标准差）。
-   **损失曲线**: 训练和验证过程中的损失曲线会被绘制并保存为 PDF 文件，存放于 `Loss_Plot/` 目录。

### 引用

如果我们的工作对您的研究有帮助，请引用我们的论文：

```bibtex
@inproceedings{yourname2025fan,
  title={FAN: Optimal Delta Hedging in Rough Volatility Models with Fractional Attention Neural Networks},
  author={Your Name and Co-authors},
  booktitle={Conference Name},
  year={2025}
}
```
```
