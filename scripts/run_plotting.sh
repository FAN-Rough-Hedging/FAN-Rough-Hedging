#!/bin/bash
echo "--- GENERATING PLOTS ---"

# Create figures directory if it doesn't exist
mkdir -p figures

python src/visualization/plotter.py \
    --results-dir results \
    --output-dir figures

echo "Plotting finished. Check the 'figures' directory."