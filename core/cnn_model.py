"""
CNN Model Architecture for ELA-based Image Tampering Detection.
Four convolutional blocks + fully connected head.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2D → BatchNorm → ReLU → MaxPool"""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TamperCNN(nn.Module):
    """
    ELA-based Tampering Detection CNN.

    Input : (B, 3, 224, 224) — normalized ELA image
    Output: (B, 2)           — logits [authentic, tampered]
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   64,  pool=True),   # 224 → 112
            ConvBlock(64,  128, pool=True),   # 112 → 56
            ConvBlock(128, 256, pool=True),   # 56  → 28
            ConvBlock(256, 512, pool=True),   # 28  → 14
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))  # → 512×4×4 = 8192

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
