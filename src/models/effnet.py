"""
EfficientNet-B3 ampliada para classificação binária
a partir de log-mel spectrograms.

Características:
    - EfficientNet-B3 (aumentado de B2 para equalizar parâmetros
      com o restante da bateria de modelos, ~10.7M em vez de ~7.7M)
    - Pesos inicializados aleatoriamente
    - Sem pré-treinamento ImageNet
    - Entrada de 1 canal
    - Classificação binária
    - Compatível com espectrogramas de tamanho variável

Entrada:
    [B, 1, n_mels, T]

Saída:
    [B, n_classes]
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


class EfficientNet(nn.Module):

    def __init__(
        self,
        n_classes: int = 2,
        in_channels: int = 1,
        dropout: float = 0.4,
    ):
        super().__init__()

        backbone = tvm.efficientnet_b3(
            weights=None
        )

        old_conv = backbone.features[0][0]

        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=False,
        )

        backbone.features[0][0] = new_conv

        in_features = backbone.classifier[1].in_features

        backbone.classifier = nn.Sequential(

            nn.Dropout(
                p=dropout,
                inplace=True,
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

    model = EfficientNet()

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