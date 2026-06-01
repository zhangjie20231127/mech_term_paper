from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDisentangler(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, num_heads: int, tau: float) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.score_head = nn.Linear(embedding_dim, 1)
        self.num_heads = num_heads
        self.tau = tau
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        if self.head_dim * num_heads != embedding_dim:
            raise ValueError("embedding_dim must be divisible by num_heads")

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = tensor.shape
        tensor = tensor.view(batch, tokens, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, query_features: torch.Tensor) -> dict[str, torch.Tensor]:
        device_type = query_features.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            tokens = self.proj(query_features.float())

            q = self._split_heads(self.query(tokens))
            k = self._split_heads(self.key(tokens))
            v = self._split_heads(self.value(tokens))

            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            soft_attention = torch.softmax(scores, dim=-1)
            mean_attention = soft_attention.mean(dim=1)

            uniform_noise = torch.rand_like(mean_attention).clamp_(1e-6, 1.0 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(uniform_noise))
            adjacency = torch.sigmoid((torch.log(mean_attention.clamp_min(1e-6)) + gumbel_noise) / self.tau)
            adjacency = adjacency * (1.0 - torch.eye(adjacency.size(-1), device=adjacency.device).unsqueeze(0))

            attended = torch.matmul(soft_attention, v)
            attended = attended.permute(0, 2, 1, 3).contiguous().view(tokens.size(0), tokens.size(1), self.embedding_dim)

            node_scores = torch.sigmoid(self.score_head(attended))
            causal_mask = node_scores
            spurious_mask = 1.0 - causal_mask

            q_causal = attended * causal_mask
            q_spurious = attended * spurious_mask
            z_causal = q_causal.mean(dim=1)
            z_spurious = q_spurious.mean(dim=1)

            return {
                "tokens": attended,
                "adjacency": adjacency,
                "causal_mask": causal_mask.squeeze(-1),
                "spurious_mask": spurious_mask.squeeze(-1),
                "q_causal": q_causal,
                "q_spurious": q_spurious,
                "z_causal": z_causal,
                "z_spurious": z_spurious,
            }


def dag_penalty(adjacency: torch.Tensor) -> torch.Tensor:
    mean_adjacency = adjacency.float().mean(dim=0)
    matrix = mean_adjacency * mean_adjacency
    return torch.trace(torch.matrix_exp(matrix)) - matrix.size(0)


def uniform_multilabel_penalty(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    target = torch.full_like(logits, 0.5)
    return F.binary_cross_entropy_with_logits(logits, target)


def uniform_multiclass_penalty(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=1)
    return -log_probs.mean(dim=1).mean()


def batch_triplet_loss(embeddings: torch.Tensor, targets: torch.Tensor, margin: float) -> torch.Tensor:
    embeddings = embeddings.float()
    targets = targets.float()
    if embeddings.size(0) < 3:
        return embeddings.new_tensor(0.0)

    distances = torch.cdist(embeddings, embeddings, p=2)
    overlap = torch.matmul(targets, targets.T)
    eye = torch.eye(targets.size(0), device=targets.device, dtype=torch.bool)
    positive_mask = (overlap > 0) & (~eye)
    negative_mask = overlap == 0

    losses = []
    for index in range(embeddings.size(0)):
        positive_candidates = torch.where(positive_mask[index])[0]
        negative_candidates = torch.where(negative_mask[index])[0]
        if positive_candidates.numel() == 0 or negative_candidates.numel() == 0:
            continue

        pos_scores = overlap[index, positive_candidates]
        pos_index = positive_candidates[pos_scores.argmax()]
        neg_distances = distances[index, negative_candidates]
        neg_index = negative_candidates[neg_distances.argmin()]
        anchor = embeddings[index : index + 1]
        positive = embeddings[pos_index : pos_index + 1]
        negative = embeddings[neg_index : neg_index + 1]
        losses.append(F.triplet_margin_loss(anchor, positive, negative, margin=margin, p=2))

    if not losses:
        return embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()
