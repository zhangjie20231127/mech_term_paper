from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionEmbeddingSine2D(nn.Module):
    def __init__(self, num_pos_feats: int, temperature: int = 10000, normalize: bool = True, scale: float | None = None) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = feature_map.shape
        y_embed = torch.arange(1, height + 1, device=feature_map.device, dtype=feature_map.dtype).view(1, height, 1).repeat(batch, 1, width)
        x_embed = torch.arange(1, width + 1, device=feature_map.device, dtype=feature_map.dtype).view(1, 1, width).repeat(batch, height, 1)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=feature_map.device, dtype=feature_map.dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
