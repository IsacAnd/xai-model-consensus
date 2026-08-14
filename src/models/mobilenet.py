"""
MobileNetV3-Large ampliada para classificação binária
a partir de log-mel spectrograms.

Características:
    - MobileNetV3-Large com width_mult=1.5 (aumentado do padrão 1.0
      para equalizar parâmetros com o restante da bateria de
      modelos, ~9.4M em vez de ~4.2M)
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


class MobileNet(nn.Module):

    def __init__(
        self,
        n_classes: int = 2,
        in_channels: int = 1,
        dropout: float = 0.3,
        width_mult: float = 1.5,
    ):
        super().__init__()

        backbone = tvm.mobilenet_v3_large(
            weights=None,
            width_mult=width_mult,
        )

        old_conv = backbone.features[0][0]

        backbone.features[0][0] = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=False,
        )

        in_features = backbone.classifier[-1].in_features

        backbone.classifier[-1] = nn.Linear(
            in_features,
            n_classes,
        )

        if isinstance(
            backbone.classifier[-2],
            nn.Dropout,
        ):
            backbone.classifier[-2] = nn.Dropout(
                p=dropout,
            )

        self.backbone = backbone

    def forward(self, x):

        # x:
        # [B, 1, n_mels, T]

        return self.backbone(x)


if __name__ == "__main__":

    model = MobileNet()

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