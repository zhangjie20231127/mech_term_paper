from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_until: str = "") -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        try:
            model = resnet50(weights=weights)
        except Exception:
            model = resnet50(weights=None)
        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.out_channels = 2048
        self._frozen_modules: list[nn.Module] = []
        self._freeze_early_layers(freeze_until)
        self._set_frozen_modules_eval()

    def _freeze_early_layers(self, freeze_until: str) -> None:
        if not freeze_until:
            return

        freeze_modules = []
        if freeze_until in {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4"}:
            freeze_modules.extend([self.stem[0], self.stem[1]])
        if freeze_until in {"layer1", "layer2", "layer3", "layer4"}:
            freeze_modules.append(self.stem[4])
        if freeze_until in {"layer2", "layer3", "layer4"}:
            freeze_modules.append(self.stem[5])
        if freeze_until in {"layer3", "layer4"}:
            freeze_modules.append(self.stem[6])
        if freeze_until == "layer4":
            freeze_modules.append(self.stem[7])

        self._frozen_modules = freeze_modules
        for module in freeze_modules:
            for param in module.parameters():
                param.requires_grad = False

    def _set_frozen_modules_eval(self) -> None:
        for module in self._frozen_modules:
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._frozen_modules:
            self._set_frozen_modules_eval()
        return self

    def forward(self, x):
        return self.stem(x)
