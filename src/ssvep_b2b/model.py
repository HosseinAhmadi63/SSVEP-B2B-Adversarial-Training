from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int = 64, kernel_size: int = 3) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = kernel_size
        self.convolution = nn.Conv1d(channels, channels, kernel_size=kernel_size)

    def forward(self, inputs: Tensor) -> Tensor:
        temporal = self.convolution(F.pad(inputs, (self.kernel_size - 1, 0)))
        return F.relu(temporal + inputs)


class CNNTCN(nn.Module):
    def __init__(self, num_classes: int, seed: int = 42) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self.num_classes = num_classes
        self.seed = seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.temporal = CausalResidualBlock(channels=64, kernel_size=3)
            self.classifier = nn.Linear(64, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape (batch, 1, channels, time)")
        if inputs.shape[1] != 1:
            raise ValueError("inputs must contain one input feature map")
        if inputs.shape[2] < 2 or inputs.shape[3] < 2:
            raise ValueError("channel and time dimensions must both be at least two")
        features = F.relu(self.conv1(inputs))
        features = self.pool(features)
        features = F.relu(self.conv2(features))
        features = features.flatten(start_dim=2)
        features = self.temporal(features)
        features = features.mean(dim=-1)
        return self.classifier(features)


CNNTCNClassifier = CNNTCN


def build_model(num_classes: int, seed: int = 42) -> CNNTCN:
    return CNNTCN(num_classes=num_classes, seed=seed)
