# src/visualization/plotter.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import argparse
import glob

# Set plot style for publication quality
sns.set_theme(style="whitegrid")

def plot_delta_comparison(results_dir, output_dir):
    """
    Plots a scatter plot comparing the model's predicted delta vs. the ground truth delta.
    A perfect model would have all points on the y=x line.
    """
    print("Generating Delta comparison scatter plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    
    models_to_plot = ['FAN', 'LSTM']
    
    for i, model_name in enumerate(models_to_plot):
        ax = axes[i]
        pred_files = glob.glob(os.path.join(results_dir, f"{model_name.lower()}*_preds.npz"))
        if not pred_files:
            print(f"Warning: No prediction file found for {model_name}. Skipping plot.")
            continue
        
        data = np.load(pred_files[0])
        predictions = data['predictions']
        ground_truths = data['ground_truths']
        
        # Subsample for clarity if there are too many points
        if len(predictions) > 5000:
            indices = np.random.choice(len(predictions), 5000, replace=False)
            predictions = predictions[indices]
            ground_truths = ground_truths[indices]
            
        ax.scatter(ground_truths, predictions, alpha=0.3, s=10, label=f'{model_name} Predictions')
        
        # Plot the perfect prediction line y=x
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Prediction')
        
        ax.set_title(f'Predicted vs. True Delta ({model_name})', fontsize=16)
        ax.set_xlabel('True Optimal Delta (Δ*)', fontsize=12)
        ax.set_ylabel('Model Predicted Delta (δ)', fontsize=12)
        ax.legend()
        ax.axis('equal')
        ax.grid(True)
        
    plt.suptitle('Comparison of Model Hedging Performance', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(output_dir, "delta_prediction_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.close()

def plot_mshe_vs_n(results_dir, output_dir):
    """
    [IMPORTANT] Plots MSHE vs. Hedging Frequency (N) on a log-log scale.
    This plot is crucial to show the convergence rate.
    
    NOTE: This function assumes you have run evaluations for different N
    and saved files like 'fan_N50_metrics.json', 'lstm_N100_metrics.json', etc.
    """
    print("\nGenerating MSHE vs. N convergence plot...")
    # TODO: As a user, you need to generate data and train models for various
    # hedging frequencies (N) to use this function effectively.
    # For now, this is a template of how you would do it.
    
    # --- Mock Data for Demonstration ---
    # Replace this with a loader for your actual result files.
    mock_data = {
        'FAN': {'N': [10, 20, 50, 100], 'mshe': [0.08, 0.05, 0.025, 0.014]},
        'LSTM': {'N': [10, 20, 50, 100], 'mshe': [0.12, 0.09, 0.06, 0.045]}
    }
    # --- End of Mock Data ---

    plt.figure(figsize=(10, 7))
    
    for model_name, data in mock_data.items():
        N = np.array(data['N'])
        mshe = np.array(data['mshe'])
        plt.loglog(N, mshe, marker='o', linestyle='-', label=f'{model_name}')

        # Fit a line in log-space to find the slope (convergence rate)
        log_N = np.log(N)
        log_mshe = np.log(mshe)
        slope, _ = np.polyfit(log_N, log_mshe, 1)
        print(f"  - Estimated convergence rate for {model_name}: {slope:.3f}")

    # Plot theoretical convergence rate
    H = cfg.H
    theoretical_slope = 2 * H - 1
    N_ref = np.array([10, 100])
    # Anchor the line at the first FAN data point for visualization
    mshe_ref = mock_data['FAN']['mshe'][0] * (N_ref / mock_data['FAN']['N'][0])**theoretical_slope
    plt.loglog(N_ref, mshe_ref, 'k--', label=f'Theoretical Slope ({theoretical_slope:.2f})')
    
    plt.title('MSHE vs. Hedging Frequency (N)', fontsize=18)
    plt.xlabel('Number of Hedging Steps (N)', fontsize=14)
    plt.ylabel('Mean Square Hedge Error (MSHE)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--")
    
    output_path = os.path.join(output_dir, "mshe_vs_n_convergence.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Plots from Evaluation Results")
    parser.add_argument('--results-dir', type=str, default='results', help='Directory containing evaluation result files.')
    parser.add_argument('--output-dir', type=str, default='figures', help='Directory to save the generated plots.')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate all plots
    plot_delta_comparison(args.results_dir, args.output_dir)
    plot_mshe_vs_n(args.results_dir, args.output_dir)