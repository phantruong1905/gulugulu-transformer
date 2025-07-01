import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import math

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.positional_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.positional_embeddings.data = pe.unsqueeze(0) * 0.1  # Scale for stability

    def forward(self, x):
        return x + self.positional_embeddings[:, :x.size(1), :]

class CleanWaveletTransformer(nn.Module):
    def __init__(self, input_dim, wavelet_levels, d_model, nhead, num_layers, drop_out, pred_len=None):
        super().__init__()
        self.input_dim = input_dim
        self.wavelet_levels = wavelet_levels
        self.d_model = d_model
        self.pred_len = pred_len
        self.drop_out = drop_out
        # Each wavelet level gets its own processing stream
        # Input embedding for each level: convert input_dim features to d_model
        self.level_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(drop_out)
            ) for _ in range(wavelet_levels)
        ])

        # Positional encoding
        self.pos_encoder = PositionalEncoding(max_len=500, d_model=d_model)

        # One transformer encoder stream per wavelet level
        self.level_streams = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True  # Pre-norm for better gradient flow
                ), num_layers=num_layers
            ) for _ in range(wavelet_levels)
        ])

        # Cross-level fusion mechanism
        self.level_fusion = nn.Sequential(
            nn.Linear(d_model * pred_len * wavelet_levels, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(drop_out)
        )

        # Temporal aggregation options
        self.temporal_aggregation = "last"  # Options: "last", "attention", "conv"

        # If using attention-based temporal aggregation
        if self.temporal_aggregation == "attention":
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=0.1,
                batch_first=True
            )
            # Learnable query for temporal attention
            self.temporal_query = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # If using convolutional temporal aggregation
        elif self.temporal_aggregation == "conv":
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1)
            )

        # Final output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(drop_out),
            nn.Linear(d_model // 2, pred_len)  # Direct prediction of future values
        )

        # Apply proper weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights properly to prevent vanishing gradients"""
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization for linear layers
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim, wavelet_levels]
        Returns: [batch_size, pred_len] - direct prediction of future values
        """
        batch_size, seq_len = x.shape[:2]

        # Process each wavelet level in its own stream
        level_outputs = []
        for level_idx in range(self.wavelet_levels):
            # Extract this level across all features: [batch, seq_len, input_dim]
            level_data = x[:, :, :, level_idx]  # [batch_size, seq_len, input_dim]

            # Embed this level's features to d_model
            level_embedded = self.level_embeddings[level_idx](level_data)  # [batch, seq_len, d_model]

            # Add positional encoding
            level_encoded = self.pos_encoder(level_embedded.transpose(0, 1)).transpose(0, 1)

            # Pass through this level's transformer stream
            level_output = self.level_streams[level_idx](level_encoded)  # [batch, seq_len, d_model]

            # Temporal aggregation - preserve sequential information
            if self.temporal_aggregation == "last":
                last_repr = level_output[:, -self.pred_len:, :]  # [B, pred_len, d_model]
                residual_input = level_embedded[:, -self.pred_len:, :]  # [B, pred_len, d_model]
                level_repr = (last_repr + residual_input).reshape(batch_size, -1)


            elif self.temporal_aggregation == "attention":
                # Attention-based temporal aggregation
                query = self.temporal_query.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 1, d_model]
                level_repr, _ = self.temporal_attention(
                    query, level_output, level_output
                )  # [batch, 1, d_model]
                level_repr = level_repr.squeeze(1)  # [batch, d_model]

            elif self.temporal_aggregation == "conv":
                # Convolutional temporal aggregation
                level_conv_input = level_output.transpose(1, 2)  # [batch, d_model, seq_len]
                level_repr = self.temporal_conv(level_conv_input).squeeze(-1)  # [batch, d_model]

            elif self.temporal_aggregation == "avg":
                level_repr = torch.mean(level_output, dim=1)

            else:  # fallback to weighted recent timesteps
                # Exponential decay weighting (more weight to recent timesteps)
                weights = torch.exp(torch.arange(seq_len, dtype=torch.float, device=x.device) * 0.1)
                weights = weights / weights.sum()  # normalize
                weights = weights.view(1, -1, 1)  # [1, seq_len, 1]
                level_repr = torch.sum(level_output * weights, dim=1)  # [batch, d_model]

            level_outputs.append(level_repr)

        # Fuse all wavelet levels
        fused_repr = torch.cat(level_outputs, dim=-1)  # [batch, d_model * wavelet_levels]
        fused_repr = self.level_fusion(fused_repr)     # [batch, d_model]

        # Direct prediction of future values
        prediction = self.output_projection(fused_repr)  # [batch, pred_len]
        return prediction