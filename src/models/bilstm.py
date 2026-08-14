"""
CNN + BiLSTM para classificação binária de deepfake de voz.

Entrada:
    [B, 1, n_mels, T]

Saída:
    [B, n_classes]

Arquitetura:

    log-mel spectrogram
            ↓
    CNN convolucional ampliada
            ↓
    redução do eixo de frequência (avg + max)
            ↓
    sequência temporal
            ↓
    BiLSTM
            ↓
    temporal pooling (avg + max)
            ↓
    classifier

Versão ampliada:

    CNN:
        32 → 64 → 128 → 256 → 512

    BiLSTM:
        hidden_size = 256
        num_layers  = 3
        bidirectional = True
"""

import torch
import torch.nn as nn

from src.models.cnn import ConvBlock


class BiLSTM(nn.Module):

    def __init__(
        self,
        n_classes: int = 2,
        in_channels: int = 1,

        rnn_hidden: int = 256,
        rnn_layers: int = 3,

    ):
        super().__init__()

        self.conv = nn.Sequential(

            ConvBlock(
                in_channels,
                32,
                dropout=0.05,
            ),

            ConvBlock(
                32,
                64,
                dropout=0.10,
            ),

            ConvBlock(
                64,
                128,
                dropout=0.10,
            ),

            ConvBlock(
                128,
                256,
                dropout=0.15,
            ),

            ConvBlock(
                256,
                512,
                dropout=0.15,
            ),
        )

        self.freq_reduce_avg = nn.AdaptiveAvgPool2d(
            (1, None)
        )

        self.freq_reduce_max = nn.AdaptiveMaxPool2d(
            (1, None)
        )

        self.rnn_hidden = rnn_hidden

        self.rnn = nn.LSTM(

            input_size=512 * 2,

            hidden_size=rnn_hidden,

            num_layers=rnn_layers,

            batch_first=True,

            bidirectional=True,

            dropout=(
                0.2
                if rnn_layers > 1
                else 0.0
            ),
        )

        self.classifier = nn.Sequential(

            nn.Dropout(
                0.4
            ),

            nn.Linear(
                rnn_hidden * 2 * 2,
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

        feats = self.conv(
            x
        )

        avg = self.freq_reduce_avg(
            feats
        )

        max_ = self.freq_reduce_max(
            feats
        )

        avg = avg.squeeze(2)

        max_ = max_.squeeze(2)

        feats = torch.cat(
            [avg, max_],
            dim=1,
        )

        # [B, 512*2, T']

        feats = feats.permute(
            0,
            2,
            1,
        )

        lstm_out, _ = self.rnn(
            feats
        )

        avg = lstm_out.mean(
            dim=1
        )

        max_ = lstm_out.max(
            dim=1
        ).values

        pooled = torch.cat(
            [avg, max_],
            dim=1,
        )

        return self.classifier(
            pooled
        )


if __name__ == "__main__":

    model = BiLSTM()

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