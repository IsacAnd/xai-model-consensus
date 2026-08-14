"""
Conformer para classificação binária de deepfake de voz.

Entrada:
    [B, 1, n_mels, T]

Saída:
    [B, n_classes]

Arquitetura:

    log-mel
        ↓
    convolução temporal/frequencial
        ↓
    Conformer blocks
        ↓
    temporal pooling
        ↓
    classifier

ConformerBase:
    embed_dim    = 384
    depth        = 8
    n_heads      = 6
    ff_expansion = 1.2  # reduzido do padrão (4) para equalizar
                         # parâmetros com ViT/AST/Swin (~14.3M)

Treinamento:
    - from scratch
    - entrada de 1 canal
    - sem pesos pré-treinados
"""


import torch
import torch.nn as nn


class ConformerConvModule(nn.Module):

    def __init__(
        self,
        dim,
        kernel_size=15,
        dropout=0.1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            dim
        )

        self.pointwise_conv1 = nn.Conv1d(
            dim,
            dim * 2,
            kernel_size=1,
        )

        self.glu = nn.GLU(
            dim=1
        )

        self.depthwise_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )

        self.batch_norm = nn.BatchNorm1d(
            dim
        )

        self.activation = nn.SiLU()

        self.pointwise_conv2 = nn.Conv1d(
            dim,
            dim,
            kernel_size=1,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):

        residual = x

        x = self.norm(
            x
        )

        x = x.transpose(
            1,
            2,
        )

        x = self.pointwise_conv1(
            x
        )

        x = self.glu(
            x
        )

        x = self.depthwise_conv(
            x
        )

        x = self.batch_norm(
            x
        )

        x = self.activation(
            x
        )

        x = self.pointwise_conv2(
            x
        )

        x = self.dropout(
            x
        )

        # [B, T, D]

        x = x.transpose(
            1,
            2,
        )

        return residual + x


class FeedForwardModule(nn.Module):

    def __init__(
        self,
        dim,
        expansion=4,
        dropout=0.1,
    ):
        super().__init__()

        hidden_dim = int(
            dim * expansion
        )

        self.net = nn.Sequential(

            nn.LayerNorm(
                dim
            ),

            nn.Linear(
                dim,
                hidden_dim,
            ),

            nn.SiLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                dim,
            ),

            nn.Dropout(
                dropout
            ),
        )

    def forward(self, x):

        return self.net(
            x
        )


class ConformerBlock(nn.Module):

    def __init__(
        self,
        dim=384,
        n_heads=6,
        ff_expansion=1.2,
        conv_kernel=15,
        dropout=0.1,
    ):
        super().__init__()

        self.ff1 = FeedForwardModule(
            dim=dim,
            expansion=ff_expansion,
            dropout=dropout,
        )

        self.norm_attn = nn.LayerNorm(
            dim
        )

        self.attention = nn.MultiheadAttention(

            embed_dim=dim,

            num_heads=n_heads,

            dropout=dropout,

            batch_first=True,
        )

        self.dropout_attn = nn.Dropout(
            dropout
        )

        self.conv = ConformerConvModule(

            dim=dim,

            kernel_size=conv_kernel,

            dropout=dropout,
        )

        self.ff2 = FeedForwardModule(

            dim=dim,

            expansion=ff_expansion,

            dropout=dropout,
        )

        self.final_norm = nn.LayerNorm(
            dim
        )

    def forward(self, x):

        x = x + 0.5 * self.ff1(
            x
        )

        residual = x

        x_norm = self.norm_attn(
            x
        )

        attn_out, _ = self.attention(

            x_norm,

            x_norm,

            x_norm,

            need_weights=False,
        )

        x = residual + self.dropout_attn(
            attn_out
        )

        x = self.conv(
            x
        )

        x = x + 0.5 * self.ff2(
            x
        )

        return self.final_norm(
            x
        )


class Conformer(nn.Module):

    def __init__(
        self,
        n_classes: int = 2,
        in_channels: int = 1,

        embed_dim: int = 384,
        depth: int = 8,
        n_heads: int = 6,

        # ff_expansion reduzido de 4 -> 1.2 para equalizar a capacidade
        # do Conformer com ViT/AST/Swin (~14.2-14.4M parâmetros).
        # Com expansion=4 (padrão do Conformer original) o modelo ficava
        # com ~27.6M, quase o dobro dos outros transformers da bateria,
        # o que confundia comparação de capacidade com família arquitetural.
        ff_expansion: float = 1.2,

        conv_kernel: int = 15,

        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(

            nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                64,
                embed_dim,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                embed_dim
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.AdaptiveAvgPool2d(
                (1, None)
            ),
        )

        self.blocks = nn.ModuleList(

            [

                ConformerBlock(

                    dim=embed_dim,

                    n_heads=n_heads,

                    ff_expansion=ff_expansion,

                    conv_kernel=conv_kernel,

                    dropout=dropout,
                )

                for _ in range(depth)
            ]
        )

        self.classifier = nn.Sequential(

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                embed_dim,
                256,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                128,
                n_classes,
            ),
        )

    def forward(self, x):

        x = self.input_projection(
            x
        )

        x = x.squeeze(
            2
        )

        x = x.transpose(
            1,
            2,
        )

        for block in self.blocks:

            x = block(
                x
            )

        x = x.mean(
            dim=1
        )

        return self.classifier(
            x
        )


if __name__ == "__main__":

    model = Conformer()

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