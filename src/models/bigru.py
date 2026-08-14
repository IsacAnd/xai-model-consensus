"""
BiGRU para classificação binária a partir de log-mel spectrogram.

Arquitetura:

    Log-mel spectrogram
            ↓
    5 × ConvBlock
            ↓
    DilatedBlock
            ↓
    SE Attention
            ↓
    Frequency reduction
            ↓
    BiGRU
            ↓
    Average + Max temporal pooling
            ↓
    Classifier

Entrada:
    [B, 1, n_mels, T]

Saída:
    [B, n_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        pool=(2, 2),
        dropout=0.1,
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                out_ch
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(
                out_ch
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                pool
            ),

            nn.Dropout2d(
                dropout
            ),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16,
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            8,
        )

        self.pool = nn.AdaptiveAvgPool2d(
            1
        )

        self.fc = nn.Sequential(

            nn.Linear(
                channels,
                hidden,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Linear(
                hidden,
                channels,
            ),

            nn.Sigmoid(),
        )

    def forward(self, x):

        b, c, _, _ = x.shape

        weights = self.pool(x)

        weights = weights.view(
            b,
            c,
        )

        weights = self.fc(
            weights
        )

        weights = weights.view(
            b,
            c,
            1,
            1,
        )

        return x * weights


class DilatedBlock(nn.Module):

    def __init__(
        self,
        channels,
        dilation=2,
    ):
        super().__init__()

        padding = dilation

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

    def forward(self, x):
        return self.block(x)


class BiGRU(nn.Module):
    """
    CRNN ampliada:

        CNN-Large backbone
            +
        BiGRU

    Entrada:
        [B, 1, n_mels, T]

    Saída:
        [B, n_classes]
    """

    def __init__(
        self,
        n_classes: int = 2,
        in_channels: int = 1,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
    ):
        super().__init__()

        self.features = nn.Sequential(

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

        self.dilated = DilatedBlock(
            512,
            dilation=2,
        )

        self.attention = SEBlock(
            512
        )

        self.freq_reduce_avg = nn.AdaptiveAvgPool2d(
            (1, None)
        )

        self.freq_reduce_max = nn.AdaptiveMaxPool2d(
            (1, None)
        )

        self.rnn_input_size = 512 * 2

        self.rnn = nn.GRU(
            input_size=self.rnn_input_size,
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

        rnn_output_dim = rnn_hidden * 2

        # Average + Max pooling temporal
        pooled_dim = rnn_output_dim * 2

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(
                0.4
            ),

            nn.Linear(
                pooled_dim,
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

        feats = self.features(x)

        feats = self.dilated(
            feats
        )

        feats = self.attention(
            feats
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

        feats = feats.transpose(
            1,
            2,
        )

        rnn_out, _ = self.rnn(
            feats
        )

        avg = rnn_out.mean(
            dim=1
        )

        max_ = rnn_out.max(
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

    model = BiGRU()

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