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
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [n_samples, n_channels]
    waveform = torch.from_numpy(data.T)  # [n_channels, n_samples]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def random_crop_or_pad(waveform: torch.Tensor, crop_samples: int,
                        generator: Optional[torch.Generator] = None) -> torch.Tensor:
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


def random_crop_or_pad_logmel(log_mel: torch.Tensor, target_frames: int,
                               generator: Optional[torch.Generator] = None) -> torch.Tensor:
    t = log_mel.shape[-1]
    if t == target_frames:
        return log_mel
    if t > target_frames:
        max_start = t - target_frames
        if generator is not None:
            start = int(torch.randint(0, max_start + 1, (1,), generator=generator).item())
        else:
            start = int(torch.randint(0, max_start + 1, (1,)).item())
        return log_mel[..., start:start + target_frames]
    # pad circular (wrap) ao longo do eixo de frames
    n_repeats = target_frames // t + 1
    tiled = log_mel.repeat(1, 1, n_repeats)
    return tiled[..., :target_frames]


def cache_path_for(file_id: str, split: str) -> Path:
    # hash curto evita problemas com nomes de arquivo muito longos/estranhos.
    # FIX: sufixo "_full" porque agora o cache guarda o log-mel COMPLETO
    # (sem crop), não o recorte de 2s como antes. Isso também evita colidir
    # com arquivos de cache antigos gerados pela versão anterior do código.
    safe_id = hashlib.md5(file_id.encode()).hexdigest()[:16] + "_" + Path(file_id).stem[:40]
    return PATHS.cache_dir / split / f"{safe_id}_full.pt"


def compute_and_cache_logmel(
    audio_path: Path,
    file_id: str,
    split: str,
    extractor: LogMelExtractor,
    generator: Optional[torch.Generator] = None,
    force_recompute: bool = False,
) -> torch.Tensor:
    out_path = cache_path_for(file_id, split)
    if out_path.exists() and not force_recompute:
        full_log_mel = torch.load(out_path)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        waveform = load_waveform(audio_path, AUDIO.sample_rate)
        full_log_mel = extractor(waveform)  # [1, n_mels, T_completo] - SEM crop
        torch.save(full_log_mel, out_path)

    return random_crop_or_pad_logmel(
        full_log_mel, AUDIO.expected_frames, generator=generator
    )