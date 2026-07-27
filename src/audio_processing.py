"""
Etapa 1/2 - Processamento de áudio.

Pipeline por arquivo:
  1. Carrega o áudio (mono, resample se necessário para AUDIO.sample_rate)
  2. Random crop de AUDIO.crop_seconds (pad circular se o áudio for mais curto)
  3. Extrai log-mel spectrogram (128 mel bins, 25 ms / 10 ms, log scaling)
  4. Salva o tensor resultante em cache_dir/<split>/<file_id>.pt

O cache evita recomputar o logmel a cada época (custo caro de I/O + FFT).
Para manter reprodutibilidade, cada worker usa um gerador de números
aleatórios próprio, semeado a partir de (seed global + índice do item + época opcional).
"""

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from config import AUDIO, PATHS


class LogMelExtractor:
    """Encapsula a transformação waveform -> log-mel spectrogram."""

    def __init__(self):
        self.sample_rate = AUDIO.sample_rate
        fmax = AUDIO.fmax if AUDIO.fmax is not None else AUDIO.sample_rate / 2

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=AUDIO.n_fft,
            win_length=AUDIO.win_length,
            hop_length=AUDIO.hop_length,
            n_mels=AUDIO.n_mels,
            f_min=AUDIO.fmin,
            f_max=fmax,
            power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=AUDIO.top_db
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: [1, n_samples] -> retorna [1, n_mels, n_frames] (log-mel, dB)."""
        mel = self.mel_transform(waveform)          # [1, n_mels, T]
        log_mel = self.db_transform(mel)             # escala log (dB), já com clamp por top_db
        return log_mel


def load_waveform(path: Path, target_sr: int) -> torch.Tensor:
    """
    Carrega áudio, converte para mono e faz resample se necessário.
    Usa `soundfile` diretamente (em vez de `torchaudio.load`) porque versões
    recentes do torchaudio exigem o backend opcional `torchcodec` (com
    dependência de ffmpeg) para I/O, o que é uma fonte comum de dor de cabeça
    em clusters (ex.: CENAPAD). `soundfile`/libsndfile lê FLAC/WAV nativamente
    sem essa dependência extra.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [n_samples, n_channels]
    waveform = torch.from_numpy(data.T)  # [n_channels, n_samples]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def random_crop_or_pad(waveform: torch.Tensor, crop_samples: int,
                        generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Padroniza a duração do áudio para exatamente `crop_samples`.
    - Se o áudio for mais longo: random crop (posição inicial aleatória).
    - Se for mais curto: pad circular (repete o sinal) para preencher a duração.
    """
    n = waveform.shape[-1]
    if n == crop_samples:
        return waveform
    if n > crop_samples:
        max_start = n - crop_samples
        if generator is not None:
            start = int(torch.randint(0, max_start + 1, (1,), generator=generator).item())
        else:
            start = int(torch.randint(0, max_start + 1, (1,)).item())
        return waveform[:, start:start + crop_samples]
    # pad circular (wrap): repete o áudio até cobrir crop_samples
    n_repeats = crop_samples // n + 1
    tiled = waveform.repeat(1, n_repeats)
    return tiled[:, :crop_samples]


def cache_path_for(file_id: str, split: str) -> Path:
    # hash curto evita problemas com nomes de arquivo muito longos/estranhos
    safe_id = hashlib.md5(file_id.encode()).hexdigest()[:16] + "_" + Path(file_id).stem[:40]
    return PATHS.cache_dir / split / f"{safe_id}.pt"


def compute_and_cache_logmel(
    audio_path: Path,
    file_id: str,
    split: str,
    extractor: LogMelExtractor,
    generator: Optional[torch.Generator] = None,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retorna o log-mel (computa e salva em cache se ainda não existir).
    Nota: como há random crop, o cache guarda UM recorte específico. Para treino
    isso já funciona como uma forma leve de data augmentation entre epochs se
    `force_recompute=True` for usado periodicamente; por padrão, reaproveita o
    cache para velocidade (comportamento determinístico após a 1a chamada).
    """
    out_path = cache_path_for(file_id, split)
    if out_path.exists() and not force_recompute:
        return torch.load(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = load_waveform(audio_path, AUDIO.sample_rate)
    waveform = random_crop_or_pad(waveform, AUDIO.crop_samples, generator=generator)
    log_mel = extractor(waveform)  # [1, n_mels, T]
    torch.save(log_mel, out_path)
    return log_mel
