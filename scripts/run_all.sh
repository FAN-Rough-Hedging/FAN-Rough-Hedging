#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "===== STARTING FULL REPLICATION PIPELINE ====="

# --- STAGE 1: DATA GENERATION ---
echo "\n[STAGE 1/4] Generating training data..."
python -m src.data_generation.main

# --- STAGE 2: MODEL TRAINING ---
echo "\n[STAGE 2/4] Training models..."
echo "Training FAN..."
python train.py --model FAN --epochs 50 
echo "Training LSTM..."
python train.py --model LSTM --epochs 50

# --- STAGE 3: EVALUATION ---
echo "\n[STAGE 3/4] Evaluating models..."
bash scripts/run_evaluation.sh

# --- STAGE 4: PLOTTING ---
echo "\n[STAGE 4/4] Generating final plots..."
bash scripts/run_plotting.sh

echo "\n===== FULL REPLICATION PIPELINE COMPLETE! ====="
echo "Check the 'figures' and 'results' directories for outputs."