"""
ResNet18 ampliada para classificação binária a partir de
log-mel spectrograms.

Características:
    - ResNet18 (reduzido de ResNet34 para equalizar parâmetros
      com o restante da bateria de modelos, ~10-14M em vez de ~21M)
    - Pesos inicializados aleatoriamente
    - Sem pré-treinamento ImageNet
    - Entrada de 1 canal
    - Classificação binária

Entrada:
    [B, 1, n_mels, T]

Saída:
    [B, n_classes]
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNet(nn.Module):

    def __init__(
        self,
        n_classes: int = 2,
        in_channels: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        backbone = tvm.resnet18(
            weights=None
        )

        old_conv = backbone.conv1

        backbone.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=False,
        )

        in_features = backbone.fc.in_features

        backbone.fc = nn.Sequential(

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                in_features,
                n_classes,
            ),
        )

        self.backbone = backbone

    def forward(self, x):

        # x:
        # [B, 1, n_mels, T]

        return self.backbone(x)


if __name__ == "__main__":

    model = ResNet()

    dummy = torch.randn(
        4,
        1,
        128,
        201,
    )

    output = model(
        dummy
    )

    print(
        output.shape
    )  # -> [4, 2]