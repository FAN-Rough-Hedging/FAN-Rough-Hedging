# evaluate.py

import torch
from torch.utils.data import DataLoader, random_split
import argparse
import os
import json
import numpy as np
from tqdm import tqdm

# Import project modules
import src.config as cfg
from src.training.dataset import HedgingDataset
from src.models.fan_model import FAN
from src.models.baselines import LSTMHedger

def evaluate(model, test_loader, device):
    """
    Evaluates the model on the test set and computes MSE and MSHE.
    """
    model.eval()
    
    total_mse = 0.0
    total_mshe = 0.0
    
    # For visualization later
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Evaluating"):
            features, labels = features.to(device), labels.to(device)
            
            # Get model prediction
            outputs = model(features)
            
            # --- Metric Calculation ---
            # 1. Mean Squared Error (MSE)
            squared_error = (outputs - labels)**2
            total_mse += squared_error.sum().item()
            
            # 2. Mean Square Hedge Error (MSHE)
            # MSHE = E[(delta_pred - delta_true)^2 * S_t^2 * sigma_t^2]
            # We need to reconstruct S_t and sigma_t^2 from features
            # features layout: [S_t/K, T-t, sqrt(sigma_sq_t), V_hist...]
            moneyness = features[:, 0]
            sqrt_sigma_sq = features[:, 2]
            
            S_t = moneyness * cfg.K
            sigma_sq_t = sqrt_sigma_sq**2
            
            # Calculate the instantaneous variance of the stock price process
            price_process_variance = (S_t**2) * sigma_sq_t
            
            weighted_squared_error = squared_error.squeeze() * price_process_variance
            total_mshe += weighted_squared_error.sum().item()

            predictions.extend(outputs.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())

    num_samples = len(test_loader.dataset)
    avg_mse = total_mse / num_samples
    avg_mshe = total_mshe / num_samples
    
    results = {
        'mse': avg_mse,
        'mshe': avg_mshe
    }
    
    return results, np.array(predictions), np.array(ground_truths)

def main(args):
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    full_dataset = HedgingDataset(data_dir=cfg.OUTPUT_DIR)
    
    # Split data consistently for testing
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    # Use a fixed generator for reproducible splits
    generator = torch.Generator().manual_seed(42)
    _, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating on {len(test_dataset)} test samples.")

    # --- 3. Initialize and Load Model ---
    num_total_features = 3 + cfg.NUM_HIST_FEATURES
    
    if args.model == 'FAN':
        model = FAN(num_features=num_total_features, num_hist_features=cfg.NUM_HIST_FEATURES)
    elif args.model == 'LSTM':
        model = LSTMHedger(num_features=num_total_features, num_hist_features=cfg.NUM_HIST_FEATURES)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    print(f"Loaded trained model '{args.model}' from {args.model_path}")

    # --- 4. Run Evaluation ---
    metrics, predictions, ground_truths = evaluate(model, test_loader, device)

    print(f"\nEvaluation Results for {args.model}:")
    print(f"  - Mean Squared Error (MSE): {metrics['mse']:.8f}")
    print(f"  - Mean Square Hedge Error (MSHE): {metrics['mshe']:.8f}")

    # --- 5. Save Results ---
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Save metrics
    metrics_path = args.output_file.replace('.json', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # Save predictions for plotting
    preds_path = args.output_file.replace('.json', '_preds.npz')
    np.savez(preds_path, predictions=predictions, ground_truths=ground_truths)
    print(f"Predictions saved to {preds_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Hedging Models")
    parser.add_argument('--model', type=str, required=True, choices=['FAN', 'LSTM'], help='Model to evaluate.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model (.pt) file.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for evaluation.')
    parser.add_argument('--output-file', type=str, required=True, help='Base path to save the output metrics (.json) and predictions.')
    
    args = parser.parse_args()
    main(args)