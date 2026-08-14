"""
Swin Transformer para classificação de log-mel spectrograms.

Entrada:
    [B, 1, n_mels, T]

Saída:
    [B, n_classes]

Características:
    - Window-based Multi-Head Self Attention
    - Shifted Windows
    - treinamento from scratch
    - entrada de 1 canal
    - tamanho variável de espectrograma
    - maior capacidade que SwinSmall

    SwinBase:
        embed_dim = 384
        depth     = 8
        heads     = 6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        dropout=0.1,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        assert dim % num_heads == 0

        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=True,
        )

        self.proj = nn.Linear(
            dim,
            dim,
        )

        self.attn_drop = nn.Dropout(
            dropout
        )

        self.proj_drop = nn.Dropout(
            dropout
        )

        # ------------------------------------------------------------
        # Relative position bias (padrão Swin Transformer):
        # cada par de posições dentro da janela recebe um bias
        # aprendido, indexado pela posição relativa entre elas.
        # ------------------------------------------------------------

        self.relative_position_bias_table = nn.Parameter(

            torch.zeros(
                (2 * window_size - 1) * (2 * window_size - 1),
                num_heads,
            )
        )

        nn.init.trunc_normal_(
            self.relative_position_bias_table,
            std=0.02,
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)

        coords = torch.stack(
            torch.meshgrid(
                coords_h,
                coords_w,
                indexing="ij",
            )
        )  # [2, Wh, Ww]

        coords_flatten = torch.flatten(
            coords,
            1,
        )  # [2, Wh*Ww]

        relative_coords = (
            coords_flatten[:, :, None]
            - coords_flatten[:, None, :]
        )  # [2, N, N]

        relative_coords = relative_coords.permute(
            1,
            2,
            0,
        ).contiguous()  # [N, N, 2]

        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1

        relative_position_index = relative_coords.sum(-1)  # [N, N]

        self.register_buffer(
            "relative_position_index",
            relative_position_index,
        )

    def _get_relative_position_bias(self):

        n = self.window_size * self.window_size

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            n,
            n,
            -1,
        )  # [N, N, n_heads]

        bias = bias.permute(
            2,
            0,
            1,
        ).contiguous()  # [n_heads, N, N]

        return bias.unsqueeze(0)  # [1, n_heads, N, N]

    def forward(self, x):

        # ------------------------------------------------------------
        # x:
        # [num_windows * B, window_size², C]
        # ------------------------------------------------------------

        B_, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(
                B_,
                N,
                3,
                self.num_heads,
                self.head_dim,
            )
            .permute(
                2,
                0,
                3,
                1,
                4,
            )
        )

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        q = q * self.scale

        attn = q @ k.transpose(
            -2,
            -1,
        )

        attn = attn + self._get_relative_position_bias()

        attn = F.softmax(
            attn,
            dim=-1,
        )

        attn = self.attn_drop(
            attn
        )

        x = (
            attn @ v
        ).transpose(
            1,
            2,
        ).reshape(
            B_,
            N,
            C,
        )

        x = self.proj(
            x
        )

        x = self.proj_drop(
            x
        )

        return x


class MLP(nn.Module):

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        hidden_dim = int(
            dim * mlp_ratio
        )

        self.net = nn.Sequential(

            nn.Linear(
                dim,
                hidden_dim,
            ),

            nn.GELU(),

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

        return self.net(x)


class SwinBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=4,
        shift_size=0,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        self.dim = dim

        self.window_size = window_size

        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(
            dim
        )

        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(
            dim
        )

        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, x):

        B, H, W, C = x.shape

        shortcut = x

        x = self.norm1(
            x
        )

        if self.shift_size > 0:

            x = torch.roll(
                x,
                shifts=(
                    -self.shift_size,
                    -self.shift_size,
                ),
                dims=(
                    1,
                    2,
                ),
            )

        pad_h = (
            self.window_size
            - H % self.window_size
        ) % self.window_size

        pad_w = (
            self.window_size
            - W % self.window_size
        ) % self.window_size

        if pad_h > 0 or pad_w > 0:

            x = F.pad(
                x,
                (
                    0,
                    0,
                    0,
                    pad_w,
                    0,
                    pad_h,
                ),
            )

        Hp, Wp = x.shape[1:3]

        x = x.view(
            B,
            Hp // self.window_size,
            self.window_size,
            Wp // self.window_size,
            self.window_size,
            C,
        )

        x = x.permute(
            0,
            1,
            3,
            2,
            4,
            5,
        ).contiguous()

        x = x.view(
            -1,
            self.window_size * self.window_size,
            C,
        )

        x = self.attn(
            x
        )

        x = x.view(
            B,
            Hp // self.window_size,
            Wp // self.window_size,
            self.window_size,
            self.window_size,
            C,
        )

        x = x.permute(
            0,
            1,
            3,
            2,
            4,
            5,
        ).contiguous()

        x = x.view(
            B,
            Hp,
            Wp,
            C,
        )

        # Remove padding

        x = x[
            :,
            :H,
            :W,
            :,
        ]

        if self.shift_size > 0:

            x = torch.roll(
                x,
                shifts=(
                    self.shift_size,
                    self.shift_size,
                ),
                dims=(
                    1,
                    2,
                ),
            )

        x = shortcut + x

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


class Swin(nn.Module):

    def __init__(
        self,
        n_classes=2,
        in_channels=1,

        embed_dim=384,
        depth=8,
        n_heads=6,

        patch_size=(16, 4),
        window_size=4,

        mlp_ratio=4.0,

        dropout=0.1,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(

            in_channels,

            embed_dim,

            kernel_size=patch_size,

            stride=patch_size,
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

        blocks = []

        for i in range(depth):

            # Alternância:

            # W-MSA
            # SW-MSA
            # W-MSA
            # SW-MSA
            # ...

            shift = (
                window_size // 2
                if i % 2 == 1
                else 0
            )

            blocks.append(

                SwinBlock(

                    dim=embed_dim,

                    num_heads=n_heads,

                    window_size=window_size,

                    shift_size=shift,

                    mlp_ratio=mlp_ratio,

                    dropout=dropout,
                )
            )

        self.blocks = nn.Sequential(
            *blocks
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

    def forward(self, x):

        x = self.patch_embed(
            x
        )

        # [B, C, H, W]

        x = x.permute(
            0,
            2,
            3,
            1,
        )

        x = self.blocks(
            x
        )
        x = self.norm(
            x
        )

        x = x.mean(
            dim=(1, 2)
        )
        return self.head(
            x
        )


if __name__ == "__main__":

    model = Swin()

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