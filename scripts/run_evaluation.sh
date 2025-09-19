#!/bin/bash
echo "--- RUNNING EVALUATION ---"

# Create results directory if it doesn't exist
mkdir -p results

# Evaluate FAN Model
echo "Evaluating FAN..."
python evaluate.py \
    --model FAN \
    --model-path trained_models/FAN_best.pt \
    --output-file results/fan.json

# Evaluate LSTM Baseline
echo "Evaluating LSTM..."
python evaluate.py \
    --model LSTM \
    --model-path trained_models/LSTM_best.pt \
    --output-file results/lstm.json

echo "Evaluation finished. Check the 'results' directory."