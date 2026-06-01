from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import ResNetBackbone
from .crocodile_blocks import CausalityMapBlock, CrocodileFeatureBlock
from .disentangler import CausalDisentangler
from .heads import GroupWiseLinear, GroupWiseLinearAdd, TaskQueryGenerator
from .position import PositionEmbeddingSine2D
from .transformer import TokenTransformerEncoder


class LabelWiseAdapter(nn.Module):
    """Lightweight residual adapter applied per label query: Q -> Q + scale * MLP(LayerNorm(Q))"""

    def __init__(self, hidden_dim: int = 256, adapter_dim: int = 64, scale: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(adapter_dim, hidden_dim)
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        # Q: [B, num_queries, hidden_dim]
        residual = self.norm(Q)
        residual = self.down(residual)
        residual = self.act(residual)
        residual = self.up(residual)
        return Q + self.scale * residual


class CrocodileCrltgaNet(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.backbone = ResNetBackbone(
            pretrained=config.get("pretrained", False),
            freeze_until=config.get("freeze_backbone_until", ""),
        )
        hidden = config["embedding_dim"]
        bypass_complex_block = bool(config.get("bypass_complex_feature_block", True))
        freeze_feature_block_seq = bool(config.get("freeze_feature_block_seq", False))
        freeze_feature_block_dnet = bool(config.get("freeze_feature_block_dnet", False))
        self.disease_feature_block = CrocodileFeatureBlock(
            in_channels=self.backbone.out_channels,
            hidden_dim=hidden,
            freeze_seq=freeze_feature_block_seq,
            freeze_dnet=freeze_feature_block_dnet,
            bypass_complex_block=bypass_complex_block,
        )
        self.domain_feature_block = CrocodileFeatureBlock(
            in_channels=self.backbone.out_channels,
            hidden_dim=hidden,
            freeze_seq=freeze_feature_block_seq,
            freeze_dnet=freeze_feature_block_dnet,
            bypass_complex_block=bypass_complex_block,
        )
        self.causality_map_extractor = CausalityMapBlock()
        self.position_embedding = PositionEmbeddingSine2D(num_pos_feats=hidden // 2)
        self.disease_transformer = TokenTransformerEncoder(
            in_channels=hidden,
            hidden_dim=hidden,
            num_heads=config["transformer_heads"],
            num_layers=config["transformer_layers"],
            ff_dim=config["transformer_ff_dim"],
            dropout=config["transformer_dropout"],
        )
        self.domain_transformer = TokenTransformerEncoder(
            in_channels=hidden,
            hidden_dim=hidden,
            num_heads=config["transformer_heads"],
            num_layers=config["transformer_layers"],
            ff_dim=config["transformer_ff_dim"],
            dropout=config["transformer_dropout"],
        )
        self.disease_query_generator = TaskQueryGenerator(
            num_queries=config["num_labels"],
            hidden_dim=hidden,
            num_heads=config["transformer_heads"],
            dropout=config["transformer_dropout"],
        )
        self.domain_query_generator = TaskQueryGenerator(
            num_queries=2,
            hidden_dim=hidden,
            num_heads=config["transformer_heads"],
            dropout=config["transformer_dropout"],
        )
        self.disease_disentangler = CausalDisentangler(
            in_channels=hidden,
            embedding_dim=hidden,
            num_heads=config["gat_heads"],
            tau=config["gumbel_tau"],
        )
        self.domain_disentangler = CausalDisentangler(
            in_channels=hidden,
            embedding_dim=hidden,
            num_heads=config["gat_heads"],
            tau=config["gumbel_tau"],
        )
        num_labels = config["num_labels"]
        self.intervention_dropout = nn.Dropout(p=config["intervention_scale"])
        self.disease_head = GroupWiseLinear(num_queries=num_labels, hidden_dim=hidden)
        self.disease_spurious_head = GroupWiseLinear(num_queries=num_labels, hidden_dim=hidden)
        self.disease_intervention_head = GroupWiseLinearAdd(num_queries=num_labels, hidden_dim=hidden)
        self.domain_head = GroupWiseLinear(num_queries=2, hidden_dim=hidden)
        self.domain_spurious_head = GroupWiseLinear(num_queries=2, hidden_dim=hidden)
        self.domain_intervention_head = GroupWiseLinearAdd(num_queries=2, hidden_dim=hidden)
        self.use_label_adapter = bool(config.get("use_label_adapter", False))
        if self.use_label_adapter:
            adapter_dim = int(config.get("label_adapter_dim", 64))
            adapter_scale = float(config.get("label_adapter_scale", 0.1))
            self.disease_label_adapter = LabelWiseAdapter(
                hidden_dim=hidden, adapter_dim=adapter_dim, scale=adapter_scale
            )
        self.freeze_disease_feature_block = bool(config.get("freeze_disease_feature_block", False))
        self.detach_domain_feature_map = bool(config.get("detach_domain_feature_map", False))
        self.freeze_domain_branch = bool(config.get("freeze_domain_branch", False))
        self._frozen_disease_modules: list[nn.Module] = []
        if self.freeze_disease_feature_block:
            self._frozen_disease_modules = [self.disease_feature_block]
            for module in self._frozen_disease_modules:
                for param in module.parameters():
                    param.requires_grad = False
            self._set_frozen_disease_modules_eval()
        self._frozen_domain_modules: list[nn.Module] = []
        if self.freeze_domain_branch:
            self._frozen_domain_modules = [
                self.domain_feature_block,
                self.domain_transformer,
                self.domain_query_generator,
                self.domain_disentangler,
                self.domain_head,
                self.domain_spurious_head,
                self.domain_intervention_head,
            ]
            for module in self._frozen_domain_modules:
                for param in module.parameters():
                    param.requires_grad = False
            self._set_frozen_domain_modules_eval()

    def _build_intervention(self, q: torch.Tensor, q_bar: torch.Tensor) -> torch.Tensor:
        batch_size = q.size(0)
        random_idx = torch.randperm(batch_size, device=q.device)
        return self.intervention_dropout(q_bar[random_idx]) + q

    def _set_frozen_domain_modules_eval(self) -> None:
        for module in self._frozen_domain_modules:
            module.eval()

    def _set_frozen_disease_modules_eval(self) -> None:
        for module in self._frozen_disease_modules:
            module.eval()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.backbone(x)
        disease_src = self.disease_feature_block(feature_map)
        domain_input = feature_map.detach() if self.detach_domain_feature_map else feature_map
        domain_src = self.domain_feature_block(domain_input)
        disease_pos = self.position_embedding(disease_src)
        domain_pos = self.position_embedding(domain_src)
        disease_encoded = self.disease_transformer(disease_src, disease_pos)
        domain_encoded = self.domain_transformer(domain_src, domain_pos)
        disease_queries = self.disease_query_generator(disease_encoded["tokens"])
        domain_queries = self.domain_query_generator(domain_encoded["tokens"])
        disease_q_for_head = self.disease_label_adapter(disease_queries["Q"]) if self.use_label_adapter else disease_queries["Q"]
        disease_features = self.disease_disentangler(disease_queries["Q"])
        domain_features = self.domain_disentangler(domain_queries["Q"])
        disease_q_intervened = self._build_intervention(disease_q_for_head, disease_queries["Q_bar"])
        domain_q_intervened = self._build_intervention(domain_queries["Q"], domain_queries["Q_bar"])
        z_x = self.disease_head(disease_q_for_head)
        z_c = self.disease_spurious_head(disease_queries["Q_bar"])
        z_c_cap = self.disease_intervention_head(disease_q_intervened)
        z_x_domain = self.domain_head(domain_queries["Q"])
        z_c_domain = self.domain_spurious_head(domain_queries["Q_bar"])
        z_c_cap_domain = self.domain_intervention_head(domain_q_intervened)
        domain_probs = torch.softmax(self.domain_logits_for_warning(z_c_domain), dim=1)
        spurious_warning_score = domain_probs[:, 1]
        return {
            "feature_map": feature_map,
            "disease_feature_map": disease_src,
            "domain_feature_map": domain_src,
            "position_embedding": disease_pos,
            "disease_tokens": disease_encoded["tokens"],
            "domain_tokens": domain_encoded["tokens"],
            "disease_query_attention": disease_queries["query_attention"],
            "domain_query_attention": domain_queries["query_attention"],
            "disease_query_features": disease_queries["Q"],
            "domain_query_features": domain_queries["Q"],
            "disease_q_bar": disease_queries["Q_bar"],
            "domain_q_bar": domain_queries["Q_bar"],
            "disease_causality_map": self.causality_map_extractor(disease_queries["Q"]),
            "disease_causal_queries": disease_features["q_causal"],
            "disease_spurious_queries": disease_features["q_spurious"],
            "disease_intervened_queries": disease_q_intervened,
            "domain_causal_queries": domain_features["q_causal"],
            "domain_spurious_queries": domain_features["q_spurious"],
            "domain_intervened_queries": domain_q_intervened,
            "z_x": z_x,
            "z_c": z_c,
            "z_c_cap": z_c_cap,
            "z_x_domain": z_x_domain,
            "z_c_domain": z_c_domain,
            "z_c_cap_domain": z_c_cap_domain,
            "disease_logits": z_x,
            "disease_spurious_logits": z_c,
            "disease_intervened_logits": z_c_cap,
            "domain_logits": z_x_domain,
            "domain_spurious_logits": z_c_domain,
            "domain_intervened_logits": z_c_cap_domain,
            "disease_embeddings": disease_features["z_causal"],
            "disease_adjacency": disease_features["adjacency"],
            "disease_masks": disease_features["causal_mask"],
            "domain_masks": domain_features["causal_mask"],
            "domain_adjacency": domain_features["adjacency"],
            "spurious_warning_score": spurious_warning_score,
        }

    def domain_logits_for_warning(self, z_c_domain: torch.Tensor) -> torch.Tensor:
        return z_c_domain

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_disease_feature_block:
            self._set_frozen_disease_modules_eval()
        if mode and self.freeze_domain_branch:
            self._set_frozen_domain_modules_eval()
        return self
