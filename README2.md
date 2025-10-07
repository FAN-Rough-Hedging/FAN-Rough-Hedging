// English Version

## High-Frequency Option Hedging Simulation

This project provides simulation and model training for high-frequency option hedging. It includes a Fractal Attention Network (FAN), LSTM, and a vanilla Transformer for benchmarking. The project supports GPU acceleration, dataset generation, training reports in JSON, and loss visualization.

## Project Structure

```
├── Loss_Plot/                # Loss curve plots
├── Model_FAN/                # FAN model (fractal attention + feed-forward network)
├── Trained_Model/            # Trained model weights (.pth)
├── TrainingReport/           # Training reports (JSON)
├── compare_models_file/      # Benchmark models (LSTM / Transformer)
├── utils/                    # Data processing and training controller
│   └── dataset/              # Prepared datasets (.pt files)
├── main.py                   # Entry script for training & validation
├── setting.py                # Model & training configurations
├── requirements.txt          # Project dependencies
└── README.md                 # Documentation
```

## Requirements

- Use a Conda virtual environment (recommended): `torchGPU`
- Python 3.11 (recommended)
- Optional: CUDA for GPU acceleration

Note: Run commands in a Conda terminal. Avoid using `&&` in PowerShell as it may fail.

## Install Dependencies

From the project root directory:

```bash
pip install -r requirements.txt
```

Alternatively, using conda (recommended for scientific libraries):

```bash
conda install pytorch torchvision numpy pandas matplotlib tqdm psutil sympy typing-extensions -c pytorch -c conda-forge
```

## Quick Start

1. Ensure datasets exist under `utils/dataset/` (files with `.pt`). If not, run follow file to get the dataset:

```bash
python utils/DataProcessor.py
```
   
2. Run the entry script:

```bash
python main.py
```

The default pipeline will:
- Initialize the training-validation process
- Load model weights from `Trained_Model/FAN_test.pth` (if present)
- Run training and validation
- Generate a JSON training report in `TrainingReport/`

## Training & Validation

- The pipeline logic is implemented in `main.py`, with training and validation coordinated via `utils/ModelController.py` and configurations in `setting.py`.
- Key configurations:
  - `FAN_SETTING`, `LSTM_SETTING`, `VANILLAT_TRANSFORMER_SETTING`
  - Dataset config, optimizer type and hyperparameters (e.g., learning rate, weight decay), LR schedulers, etc.

### Customize Training Parameters
Modify values in `setting.py`, such as:
- Optimizer (`Adam`, `AdamW`, `SGD`, etc.) and hyperparameters: `LR`, `WEIGHT_DECAY`
- LR scheduler: `StepLR`, `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`
- Training settings: `EPOCHS`, `BATCH_SIZE`

## Saving & Loading Models

- Trained weights are saved under `Trained_Model/`, e.g., `FAN_test.pth`.
- Loading weights in `main.py`:

```python
state_dict = torch.load(path, weights_only=True)
model.load_state_dict(state_dict)
```

Tip: Prefer `weights_only=True` when loading to avoid executing arbitrary objects and to improve security.

## Training Report (JSON)

After training, a JSON report is saved under `TrainingReport/`, with filenames containing timestamps and model names, e.g.:

```
TrainingReport/
├── FAN_test_YYYYMMDD_hhmmss_report.json
```

It includes:
- Training time (`YYYY-MM-DD HH:MM:SS`)
- Model name and key hyperparameters
- Dataset configuration, optimizer configuration, LR scheduler configuration
- Number of epochs, batch size
- Validation loss statistics (mean and std)

## Visualization

- Loss curves during training/validation are saved to `Loss_Plot/` (e.g., `FAN_test.pdf`).
- When using matplotlib with LaTeX mode and multilingual text:
  - Chinese font: SimSun (Windows built-in)
  - English font: Times New Roman
  - Ensure Chinese characters can be displayed correctly.

## FAQ

- Weight file not found:
  - Ensure the filename in `Trained_Model/` matches the code (e.g., `FAN_test.pth`).
- GPU OOM or memory issues:
  - Reduce `BATCH_SIZE` or dataset scale.
- "Trying to backward through the graph a second time":
  - Avoid calling `backward()` multiple times on the same graph. For gradient accumulation, use `(loss / accumulation_steps).backward()` per batch, and control the timing of `zero_grad()` and `step()`.

## Development Guidelines

- Modular, high cohesion, low coupling, reusable, maintainable
- Prefer using library-provided utilities; avoid reinventing the wheel
- Vectorize numerical operations; leverage `torch` and GPU when available
- Configure matplotlib carefully for LaTeX and multilingual display

## License

This project is intended for academic or personal research use. For commercial use or redistribution, please comply with relevant open-source licenses and copyright requirements.
