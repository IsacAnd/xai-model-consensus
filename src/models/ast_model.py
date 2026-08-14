"""
AST para classificação binária de deepfake de voz.

Entrada:
    [B, 1, n_mels, T]

Saída:
    [B, n_classes]

Arquitetura:

    log-mel spectrogram
            ↓
    espectro-temporal patch embedding
            ↓
    CLS token + positional embedding
            ↓
    Transformer Encoder
            ↓
    CLS representation
            ↓
    classifier


Versões:

    AST:
        embed_dim = 384
        depth     = 8
        n_heads   = 6

Treinamento:
    - from scratch
    - sem pesos pré-treinados
    - entrada de 1 canal
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class ASTPatchEmbed(nn.Module):

    def __init__(
        self,
        in_channels=1,
        embed_dim=384,
        patch_size=(16, 4),
    ):
        super().__init__()

        self.patch_size = patch_size

        self.proj = nn.Conv2d(

            in_channels,

            embed_dim,

            kernel_size=patch_size,

            stride=patch_size,
        )

    def forward(self, x):

        x = self.proj(
            x
        )

        b, c, h, w = x.shape

        x = x.flatten(
            2
        ).transpose(
            1,
            2,
        )

        return x, (
            h,
            w,
        )


class AST(nn.Module):

    def __init__(
        self,

        n_classes: int = 2,
        in_channels: int = 1,
        embed_dim: int = 384,
        depth: int = 8,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        patch_size=(16, 4),
        dropout: float = 0.1,
        max_patches: int = 512,
    ):
        super().__init__()

        self.patch_embed = ASTPatchEmbed(

            in_channels=in_channels,

            embed_dim=embed_dim,

            patch_size=patch_size,
        )

        self.cls_token = nn.Parameter(

            torch.zeros(
                1,
                1,
                embed_dim,
            )
        )

        self.pos_embed = nn.Parameter(

            torch.zeros(
                1,
                max_patches + 1,
                embed_dim,
            )
        )

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.pos_embed,
            std=0.02,
        )

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=embed_dim,

            nhead=n_heads,

            dim_feedforward=int(
                embed_dim * mlp_ratio
            ),

            dropout=dropout,

            activation="gelu",

            batch_first=True,

            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=depth,
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.head = nn.Sequential(

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                embed_dim,
                n_classes,
            ),
        )

    def _get_pos_embed(
        self,
        n_patches,
        device,
    ):

        if (
            n_patches + 1
            == self.pos_embed.shape[1]
        ):

            return self.pos_embed

        cls_pos = self.pos_embed[
            :,
            :1,
            :
        ]

        patch_pos = self.pos_embed[
            :,
            1:,
            :
        ]

        patch_pos = patch_pos.transpose(
            1,
            2,
        )

        patch_pos = F.interpolate(

            patch_pos,

            size=n_patches,

            mode="linear",

            align_corners=False,
        )

        patch_pos = patch_pos.transpose(
            1,
            2,
        )

        return torch.cat(

            [
                cls_pos,
                patch_pos,
            ],

            dim=1,
        )

    def forward(self, x):

        b = x.shape[0]

        patches, _ = self.patch_embed(
            x
        )

        # [B, N, D]

        n_patches = patches.shape[1]

        cls_tokens = self.cls_token.expand(

            b,

            -1,

            -1,
        )

        tokens = torch.cat(

            [
                cls_tokens,
                patches,
            ],

            dim=1,
        )

        tokens = tokens + self._get_pos_embed(

            n_patches,

            x.device,
        )

        encoded = self.encoder(
            tokens
        )

        encoded = self.norm(
            encoded
        )

        cls_out = encoded[
            :,
            0
        ]

        return self.head(
            cls_out
        )


if __name__ == "__main__":

    model = AST()

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