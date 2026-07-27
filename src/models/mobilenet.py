"""MobileNetV3-Small adaptada para log-mel 1-canal."""

import torch
import torch.nn as nn
import torchvision.models as tvm


class MobileNetSmall(nn.Module):
    def __init__(self, n_classes: int = 2, in_channels: int = 1):
        super().__init__()
        backbone = tvm.mobilenet_v3_small(weights=None)

        old_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=False,
        )

        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, n_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    m = MobileNetSmall()
    dummy = torch.randn(2, 1, 128, 201)
    print(m(dummy).shape)
