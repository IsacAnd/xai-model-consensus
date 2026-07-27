"""CNN simples para classificação binária a partir de log-mel spectrogram."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=(2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):
    """Backbone convolucional puro (4 blocos) + head de classificação."""

    def __init__(self, n_classes: int = 2, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: [B, 1, n_mels, T]
        feats = self.features(x)
        pooled = self.pool(feats)
        return self.classifier(pooled)


if __name__ == "__main__":
    m = CNN()
    dummy = torch.randn(4, 1, 128, 201)
    print(m(dummy).shape)  # -> [4, 2]
