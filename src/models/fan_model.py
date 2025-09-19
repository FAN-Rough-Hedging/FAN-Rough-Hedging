# src/models/fan_model.py

import torch
import torch.nn as nn
import numpy as np
import src.config as cfg

class FractionalAttention(nn.Module):
    """
    The core Fractional Attention mechanism.
    It applies a pre-computed, non-learnable attention mask based on the
    power-law kernel (t-s)^(H-3/2), capturing the long-range memory of
    rough volatility models.
    """
    def __init__(self, num_hist_features, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.num_hist_features = num_hist_features
        
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        
        self.register_buffer('fractional_kernel_mask', self._create_fractional_kernel())

    def _create_fractional_kernel(self):
        """
        Creates the (t-s)^(H-3/2) kernel. This is the "inductive bias".
        The kernel is fixed and not learned.
        """
        indices = torch.arange(self.num_hist_features, dtype=torch.float32)
        # Create a matrix of time differences |t-s|
        time_diffs = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0)) + 1e-9 # Add epsilon for stability
        
        # Power-law kernel from Proposition 2 in the paper
        H = cfg.H
        power = H - 1.5 
        kernel = torch.pow(time_diffs, power)
        
        # The kernel should only look backwards in time (causal)
        # However, for a static feature vector, a full look is also plausible.
        # Let's use a full kernel as the time context is static.
        
        # Normalize the kernel row-wise (like softmax, but fixed)
        kernel = kernel / kernel.sum(dim=-1, keepdim=True)
        
        return kernel.unsqueeze(0) # Add batch dimension

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, model_dim)
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Standard attention score calculation
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.model_dim)
        
        # KEY INNOVATION: Modulate scores with the fixed fractional kernel
        # This forces the model to prioritize dependencies according to the power-law
        attention_weights = torch.softmax(scores + torch.log(self.fractional_kernel_mask + 1e-9), dim=-1)
        
        output = torch.matmul(attention_weights, v)
        return output

class FAN(nn.Module):
    """
    Fractional Attention Network (FAN) for optimal hedging.
    """
    def __init__(self, num_features, num_hist_features, model_dim=64, num_layers=2):
        super().__init__()
        num_scalar_features = num_features - num_hist_features
        
        # 1. Embedding layer for scalar and historical features
        self.scalar_embed = nn.Linear(num_scalar_features, model_dim)
        self.hist_embed = nn.Linear(1, model_dim) # Embed each historical point
        
        # 2. Fractional Attention layers
        self.attention_layers = nn.ModuleList(
            [FractionalAttention(num_hist_features, model_dim) for _ in range(num_layers)]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(num_layers)])

        # 3. Output head
        self.output_head = nn.Sequential(
            nn.Linear(model_dim * (num_hist_features + 1), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature tensor of shape (batch_size, num_features)
        """
        # Split features into scalar and historical parts
        scalar_feats = x[:, :-cfg.NUM_HIST_FEATURES]
        hist_feats = x[:, -cfg.NUM_HIST_FEATURES:].unsqueeze(-1) # -> (batch, seq_len, 1)
        
        # Embed features
        scalar_embedded = self.scalar_embed(scalar_feats) # -> (batch, model_dim)
        hist_embedded = self.hist_embed(hist_feats)     # -> (batch, seq_len, model_dim)
        
        # Process historical features through FAN layers
        attn_output = hist_embedded
        for i, layer in enumerate(self.attention_layers):
            attn_output = self.layer_norms[i](attn_output + layer(attn_output))
            
        # Combine scalar and processed historical features
        combined_features = torch.cat(
            [scalar_embedded.unsqueeze(1), attn_output], 
            dim=1
        ).view(x.size(0), -1) # Flatten for the output head
        
        # Predict the final delta
        delta_pred = self.output_head(combined_features)
        return delta_pred