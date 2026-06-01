from __future__ import annotations

import torch
import torch.nn as nn


class TokenTransformerEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_heads: int, num_layers: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        if in_channels == hidden_dim:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, feature_map: torch.Tensor, pos_embed: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = self.input_proj(feature_map)
        batch, channels, height, width = projected.shape
        tokens = projected.flatten(2).transpose(1, 2)
        pos_tokens = pos_embed.flatten(2).transpose(1, 2)
        encoded = self.encoder(tokens + pos_tokens)
        encoded = self.norm(encoded)
        encoded_map = encoded.transpose(1, 2).reshape(batch, channels, height, width)
        return {"tokens": encoded, "feature_map": encoded_map}
