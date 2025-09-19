# src/models/baselines.py

import torch
import torch.nn as nn
import src.config as cfg

class LSTMHedger(nn.Module):
    """
    An LSTM-based model for benchmarking against FAN.
    """
    def __init__(self, num_features, num_hist_features, hidden_dim=64, num_layers=2):
        super().__init__()
        num_scalar_features = num_features - num_hist_features
        
        self.scalar_proj = nn.Linear(num_scalar_features, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        scalar_feats = x[:, :-cfg.NUM_HIST_FEATURES]
        hist_feats = x[:, -cfg.NUM_HIST_FEATURES:].unsqueeze(-1)

        scalar_proj = self.scalar_proj(scalar_feats)
        
        _, (hn, _) = self.lstm(hist_feats)
        hist_proj = hn[-1] # Get hidden state of the last layer
        
        combined = torch.cat([scalar_proj, hist_proj], dim=1)
        
        return self.output_head(combined)

# 你也可以在这里添加一个标准的 Transformer 模型
# class TransformerHedger(nn.Module): ...