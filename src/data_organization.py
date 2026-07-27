"""
Etapa 1 - Organização dos dados (ASVspoof 5).

O ASVspoof5 já vem com splits oficiais de train/dev/eval, então aqui apenas:
  1. Parseamos os protocolos oficiais (TSV/TXT) e localizamos os áudios em disco.
  2. Construímos manifests unificados (CSV) com colunas: file_id, path, split, label, attack_tag.
  3. A partir do manifest de eval, sorteamos (seed fixa) 1000 exemplares que serão
     reaproveitados depois na Etapa 3/4 (XAI + SHAP) — assim a amostra é fixada
     desde já e é reprodutível.

IMPORTANTE: o formato exato de colunas dos protocolos do ASVspoof5 pode variar
por arquivo (train/dev/eval têm layouts levemente diferentes, e o eval trial list
não inclui o rótulo até a fase de scoring). Por isso o parser abaixo é robusto:
ele detecta automaticamente qual coluna contém o file_id (batendo com os arquivos
de áudio existentes em disco) e qual coluna contém o rótulo (bonafide/spoof),
em vez de assumir índices fixos. Confira o cabeçalho oficial em
https://zenodo.org/records/14498691 caso seu protocolo tenha nomes de arquivo
diferentes dos esperados (ex.: eval sem rótulo público -> use os keys liberados
após a fase de challenge, ou trate o eval como "unlabeled" e pondere isso na
Etapa 3).
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import PATHS, XAI

logger = logging.getLogger(__name__)

LABEL_TOKENS = {"bonafide": 0, "spoof": 1}
AUDIO_EXTS = (".flac", ".wav", ".ogg")


def _find_audio_file(file_id: str, audio_dir: Path) -> Optional[Path]:
    """Tenta localizar o arquivo de áudio correspondente a um file_id no protocolo."""
    stem = Path(file_id).stem
    for ext in AUDIO_EXTS:
        candidate = audio_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_protocol_file(protocol_path: Path, audio_dir: Path, split: str,
                         has_label: bool = True) -> pd.DataFrame:
    """
    Parseia um arquivo de protocolo (whitespace/tab separated, sem assumir header)
    detectando automaticamente as colunas de file_id e label.
    """
    if not protocol_path.exists():
        raise FileNotFoundError(
            f"Protocolo não encontrado: {protocol_path}. Verifique config.PATHS."
        )

    raw = pd.read_csv(protocol_path, sep=r"\s+", header=None, engine="python",
                       dtype=str, comment="#")

    # detecta coluna de label (bonafide/spoof) olhando os valores únicos de cada coluna
    label_col = None
    if has_label:
        for col in raw.columns:
            values = set(raw[col].dropna().str.lower().unique())
            if values and values.issubset(LABEL_TOKENS.keys()):
                label_col = col
                break

    # detecta coluna de file_id: aquela cujo maior número de valores bate com
    # arquivos existentes de fato em audio_dir (checa uma amostra por performance)
    sample = raw.head(200)
    best_col, best_hits = None, -1
    for col in raw.columns:
        hits = sum(
            _find_audio_file(v, audio_dir) is not None
            for v in sample[col].dropna().unique()
        )
        if hits > best_hits:
            best_hits, best_col = hits, col
    file_id_col = best_col

    if file_id_col is None or best_hits == 0:
        raise ValueError(
            f"Não foi possível localizar a coluna de file_id em {protocol_path} "
            f"a partir dos áudios em {audio_dir}. Confira os paths em config.py."
        )

    # tenta também achar coluna de attack_tag (bonafide vira 'bonafide' explicitamente
    # em vários protocolos ASVspoof, ou "-" quando não aplicável)
    attack_col = None
    for col in raw.columns:
        if col in (label_col, file_id_col):
            continue
        values = raw[col].dropna().unique()
        if any(str(v).lower().startswith("a") and str(v)[1:].isdigit() for v in values):
            attack_col = col
            break

    records = []
    for _, row in raw.iterrows():
        file_id = row[file_id_col]
        audio_path = _find_audio_file(file_id, audio_dir)
        if audio_path is None:
            continue
        label = None
        if label_col is not None:
            label = LABEL_TOKENS.get(str(row[label_col]).lower())
        attack_tag = row[attack_col] if attack_col is not None else None
        records.append({
            "file_id": Path(file_id).stem,
            "path": str(audio_path),
            "split": split,
            "label": label,
            "attack_tag": attack_tag,
        })

    df = pd.DataFrame(records)
    logger.info("Split=%s: %d/%d linhas do protocolo casadas com áudio em disco",
                split, len(df), len(raw))
    return df


def build_manifests(save: bool = True) -> dict:
    """Gera (e opcionalmente salva) os manifests de train/dev/eval em data/*.csv."""
    PATHS.ensure_dirs()

    train_df = parse_protocol_file(PATHS.train_protocol, PATHS.train_audio_dir,
                                    split="train", has_label=True)
    dev_df = parse_protocol_file(PATHS.dev_protocol, PATHS.dev_audio_dir,
                                  split="dev", has_label=True)
    # eval trial list pode não trazer rótulo público (depende da fase do challenge);
    # o parser lida com isso automaticamente (label fica None se não detectar coluna)
    eval_df = parse_protocol_file(PATHS.eval_protocol, PATHS.eval_audio_dir,
                                   split="eval", has_label=True)

    manifests = {"train": train_df, "dev": dev_df, "eval": eval_df}

    if save:
        for split, df in manifests.items():
            out_path = PATHS.manifest_dir / f"manifest_{split}.csv"
            df.to_csv(out_path, index=False)
            logger.info("Manifest salvo: %s (%d linhas)", out_path, len(df))

    return manifests


def sample_xai_subset(eval_df: pd.DataFrame, n_samples: int = None,
                       seed: int = None, save: bool = True) -> pd.DataFrame:
    """
    Sorteia (seed fixa) N exemplares do eval para uso posterior nas Etapas 3/4
    (geração de mapas SHAP + métricas de consenso). Mantém a mesma proporção
    bonafide/spoof do eval original quando possível (amostragem estratificada).
    """
    n_samples = n_samples or XAI.n_shap_samples
    seed = seed if seed is not None else XAI.seed

    labeled = eval_df.dropna(subset=["label"])
    if labeled.empty:
        logger.warning(
            "Eval sem rótulos públicos: amostragem XAI será feita sem estratificação."
        )
        subset = eval_df.sample(n=min(n_samples, len(eval_df)), random_state=seed)
    else:
        frac = min(1.0, n_samples / len(labeled))
        # Nota: usar GroupBy.sample() (e não .apply(lambda g: g.sample(...)))
        # é proposital. A partir do pandas 3.0, GroupBy.apply() passou a
        # excluir por padrão a própria coluna de agrupamento (aqui, "label")
        # do DataFrame recebido pela função — o que fazia o subset resultante
        # perder a coluna "label" silenciosamente. GroupBy.sample() não passa
        # por apply() internamente, então preserva todas as colunas em
        # qualquer versão do pandas (2.x ou 3.x).
        subset = labeled.groupby("label", group_keys=False).sample(
            frac=frac, random_state=seed
        )
        # ajusta para exatamente n_samples (arredondamentos do groupby podem sobrar/faltar)
        if len(subset) > n_samples:
            subset = subset.sample(n=n_samples, random_state=seed)
        elif len(subset) < n_samples:
            remaining = labeled.drop(subset.index)
            extra = remaining.sample(n=n_samples - len(subset), random_state=seed)
            subset = pd.concat([subset, extra])

    subset = subset.reset_index(drop=True)
    if save:
        out_path = PATHS.manifest_dir / "xai_subset_1000.csv"
        subset.to_csv(out_path, index=False)
        logger.info("Subset XAI salvo: %s (%d linhas)", out_path, len(subset))
    return subset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manifests = build_manifests()
    sample_xai_subset(manifests["eval"])