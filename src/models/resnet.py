"""ResNet18 (a menor ResNet padrão do torchvision) adaptada para log-mel 1-canal."""

import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNetSmall(nn.Module):
    def __init__(self, n_classes: int = 2, in_channels: int = 1):
        super().__init__()
        backbone = tvm.resnet18(weights=None)

        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=False,
        )

        backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    m = ResNetSmall()
    dummy = torch.randn(2, 1, 128, 201)
    print(m(dummy).shape)
