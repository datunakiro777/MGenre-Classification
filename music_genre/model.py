"""Compact four-block CNN for log-mel spectrogram crops."""

from __future__ import annotations

import torch
from torch import nn


class GenreCNN(nn.Module):
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        blocks = []
        in_channels = 1
        for out_channels in (32, 64, 128, 256):
            blocks += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
