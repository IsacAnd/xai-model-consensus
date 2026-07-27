"""
ViT pequeno (from scratch) para log-mel spectrograms.

Não usamos torchvision.models.vit_* porque essas implementações assumem
imagens 224x224 fixas (embeddings posicionais fixos), o que exigiria
redimensionar o espectrograma e distorcer a relação tempo/frequência.
Aqui o patch embedding é feito via Conv2d, então o modelo se adapta a
qualquer (n_mels, T) automaticamente, com positional embedding aprendido
dinamicamente para o grid resultante.
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels=1, embed_dim=192, patch_size=(16, 4)):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                               stride=patch_size)

    def forward(self, x):
        # x: [B, 1, n_mels, T] -> [B, embed_dim, H', W']
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
        return x, (h, w)


class ViTSmall(nn.Module):
    def __init__(self, n_classes: int = 2, in_channels: int = 1,
                 embed_dim: int = 192, depth: int = 6, n_heads: int = 3,
                 mlp_ratio: float = 4.0, patch_size=(16, 4), dropout: float = 0.1,
                 max_patches: int = 512):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # pos embedding dimensionado para o maior grid esperado; interpolado
        # dinamicamente se o input gerar um número de patches diferente
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    def _get_pos_embed(self, n_patches: int, device):
        if n_patches + 1 == self.pos_embed.shape[1]:
            return self.pos_embed
        # interpola o positional embedding (exceto o token cls) para o n_patches atual
        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]
        patch_pos = patch_pos.transpose(1, 2)  # [1, dim, N]
        patch_pos = torch.nn.functional.interpolate(
            patch_pos, size=n_patches, mode="linear", align_corners=False
        )
        patch_pos = patch_pos.transpose(1, 2)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x):
        b = x.shape[0]
        patches, (h, w) = self.patch_embed(x)         # [B, N, D]
        n_patches = patches.shape[1]

        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)
        tokens = tokens + self._get_pos_embed(n_patches, x.device)

        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)
        cls_out = encoded[:, 0]
        return self.head(cls_out)


if __name__ == "__main__":
    m = ViTSmall()
    dummy = torch.randn(2, 1, 128, 201)
    print(m(dummy).shape)
