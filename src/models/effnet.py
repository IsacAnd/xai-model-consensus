"""
EfficientNet (versão pequena = efficientnet_b0, a menor da família) adaptada para
receber log-mel de 1 canal. Pesos aleatórios (não usamos pré-treino ImageNet,
já que o domínio - espectrograma - é bem diferente de imagem natural e queremos
comparação justa de arquitetura "from scratch" com os demais modelos).
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


class EfficientNetSmall(nn.Module):
    def __init__(self, n_classes: int = 2, in_channels: int = 1):
        super().__init__()
        backbone = tvm.efficientnet_b0(weights=None)

        # adapta a primeira conv para aceitar 1 canal em vez de 3
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=False,
        )
        backbone.features[0][0] = new_conv

        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, n_classes)

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    m = EfficientNetSmall()
    dummy = torch.randn(2, 1, 128, 201)
    print(m(dummy).shape)
