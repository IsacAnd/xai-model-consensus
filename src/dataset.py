import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from config import AUDIO, PATHS
from src.audio_processing import LogMelExtractor, compute_and_cache_logmel

logger = logging.getLogger(__name__)


class ADDDataset(Dataset):
    """
    Espera um DataFrame com colunas: file_id, path, split, label.
    `label` deve ser 0 (bonafide) ou 1 (spoof); linhas sem label são descartadas
    (ex.: eval sem rótulo público) a menos que `allow_unlabeled=True`.
    """

    def __init__(self, manifest: pd.DataFrame, split: str,
                 allow_unlabeled: bool = False, force_recompute: bool = False):
        df = manifest[manifest["split"] == split].copy()
        if not allow_unlabeled:
            before = len(df)
            df = df.dropna(subset=["label"])
            if len(df) < before:
                logger.info("Split=%s: %d linhas sem label descartadas",
                            split, before - len(df))
        df["label"] = df["label"].astype("Int64")
        self.df = df.reset_index(drop=True)
        self.split = split
        self.extractor = LogMelExtractor()
        self.force_recompute = force_recompute

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.split == "train":
            generator = torch.Generator()
            generator.manual_seed(torch.seed() & 0xFFFFFFFF)  # semente nova a cada chamada
        else:
            generator = torch.Generator().manual_seed(idx)  # dev/eval: determinístico

        log_mel = compute_and_cache_logmel(
            audio_path=Path(row["path"]),
            file_id=row["file_id"],
            split=self.split,
            extractor=self.extractor,
            generator=generator,
            force_recompute=self.force_recompute,
        )
        label = int(row["label"]) if not pd.isna(row["label"]) else -1
        return log_mel, label, row["file_id"]


def collate_fn(batch):
    """Empilha [1, n_mels, T] -> [B, 1, n_mels, T]. Corta/pad para T uniforme se preciso."""
    mels, labels, file_ids = zip(*batch)
    t_target = AUDIO.expected_frames
    fixed = []
    for m in mels:
        t = m.shape[-1]
        if t > t_target:
            m = m[..., :t_target]
        elif t < t_target:
            pad = t_target - t
            m = torch.nn.functional.pad(m, (0, pad))
        fixed.append(m)
    x = torch.stack(fixed, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y, list(file_ids)


def load_manifest(split: str) -> pd.DataFrame:
    path = PATHS.manifest_dir / f"manifest_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode src/data_organization.py primeiro (Etapa 1)."
        )
    return pd.read_csv(path)