"""CRNN: mesma base convolucional da CNN + GRU bidirecional sobre o eixo temporal."""

import torch
import torch.nn as nn

from src.models.cnn import ConvBlock


class CRNN(nn.Module):
    def __init__(self, n_classes: int = 2, in_channels: int = 1,
                 rnn_hidden: int = 128, rnn_layers: int = 2):
        super().__init__()
        # 3 blocos convolucionais (menos pooling em freq que a CNN pura, para
        # sobrar uma dimensão de frequência razoável a ser achatada em features)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.rnn_input_proj = None  # criado dinamicamente no forward (lazy init)
        self.rnn_hidden = rnn_hidden
        self.rnn = nn.GRU(
            input_size=128,  # projetado via Conv1x1 abaixo para tamanho fixo
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if rnn_layers > 1 else 0.0,
        )
        self.freq_reduce = nn.AdaptiveAvgPool2d((1, None))  # colapsa eixo de frequência
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(rnn_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: [B, 1, n_mels, T]
        feats = self.conv(x)               # [B, C, F', T']
        feats = self.freq_reduce(feats)    # [B, C, 1, T']
        feats = feats.squeeze(2)           # [B, C, T']
        feats = feats.permute(0, 2, 1)     # [B, T', C]  (sequência para a GRU)

        rnn_out, _ = self.rnn(feats)       # [B, T', 2*hidden]
        pooled = rnn_out.mean(dim=1)       # pooling temporal médio
        return self.classifier(pooled)


if __name__ == "__main__":
    m = CRNN()
    dummy = torch.randn(4, 1, 128, 201)
    print(m(dummy).shape)  # -> [4, 2]
