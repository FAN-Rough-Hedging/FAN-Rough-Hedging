# trainer.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
from tqdm import tqdm

# Import project modules
import src.config as cfg
from src.training.dataset import HedgingDataset
from src.models.fan_model import FAN
from src.models.baselines import LSTMHedger
from src.training.loss import get_loss_function

def main(args):
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    dataset = HedgingDataset(data_dir=cfg.OUTPUT_DIR)
    
    # Split data into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    # --- 3. Initialize Model ---
    num_total_features = 3 + cfg.NUM_HIST_FEATURES # Moneyness, TTM, Vol + Historical
    
    if args.model == 'FAN':
        model = FAN(num_features=num_total_features, num_hist_features=cfg.NUM_HIST_FEATURES)
    elif args.model == 'LSTM':
        model = LSTMHedger(num_features=num_total_features, num_hist_features=cfg.NUM_HIST_FEATURES)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
        
    model.to(device)
    print(f"Model '{args.model}' initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # --- 4. Loss and Optimizer ---
    criterion = get_loss_function()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

    # --- 5. Training Loop ---
    best_val_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(args.save_dir, f'{args.model}_best.pt')
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path}")

    print("Training finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Hedging Models on Rough Volatility Data")
    parser.add_argument('--model', type=str, required=True, choices=['FAN', 'LSTM'], help='Model to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='trained_models', help='Directory to save trained models.')
    
    args = parser.parse_args()
    main(args)
