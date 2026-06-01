from __future__ import annotations

import math

import torch
import torch.nn as nn


class GroupWiseLinear(nn.Module):
    def __init__(self, num_queries: int, hidden_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, num_queries, hidden_dim))
        self.bias = nn.Parameter(torch.empty(1, num_queries)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.weight.size(-1))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        logits = (query_features * self.weight).sum(dim=-1)
        if self.bias is not None:
            logits = logits + self.bias
        return logits


class GroupWiseLinearAdd(nn.Module):
    def __init__(self, num_queries: int, hidden_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, num_queries, hidden_dim))
        self.bias = nn.Parameter(torch.empty(1, num_queries)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.weight.size(-1))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        logits = (query_features * self.weight).sum(dim=-1)
        if self.bias is not None:
            logits = logits + self.bias
        return logits


class TaskQueryGenerator(nn.Module):
    def __init__(self, num_queries: int, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.confounding_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, encoded_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = encoded_tokens.size(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        attended_queries, attn_weights = self.cross_attn(query=queries, key=encoded_tokens, value=encoded_tokens)
        query_features = self.norm(queries + attended_queries + self.ffn(attended_queries))
        confounding_queries = self.norm(queries + self.confounding_proj(queries - attended_queries))
        return {
            "Q": query_features,
            "Q_bar": confounding_queries,
            "query_attention": attn_weights,
        }
