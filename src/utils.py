"""Utilitários compartilhados: early stopping, seed, logging setup."""

import logging
import random

import numpy as np
import pandas as pd
import torch

from config import PATHS

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_path=None, level=logging.INFO):
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


class EarlyStopping:
    """
    Critério de seleção padronizado para todos os modelos: monitora uma métrica
    de validação (por padrão, EER) e para o treino se não houver melhora por
    `patience` épocas. Também guarda o "melhor" estado para salvar o checkpoint.
    """

    def __init__(self, patience: int = 8, mode: str = "min", min_delta: float = 1e-4):
        assert mode in ("min", "max")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def _is_better(self, score, best):
        if self.mode == "min":
            return score < best - self.min_delta
        return score > best + self.min_delta

    def step(self, score: float) -> bool:
        """Retorna True se `score` é a melhor métrica vista até agora."""
        if self.best_score is None or self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def merge_shap_manifests() -> pd.DataFrame:
    """
    Combina os manifestos por modelo (results/shap/shap_manifest_<modelo>.csv,
    um por task da Etapa 4) em um único results/shap/shap_manifest.csv,
    consumido pela Etapa 5.

    Cada modelo escreve seu próprio arquivo em src/xai.py (em vez de um
    read-modify-write compartilhado), justamente para evitar race condition
    quando a Etapa 4 roda como job array do Slurm (1 modelo por task, todas
    em paralelo). Esta função faz a junção uma única vez, e só deve ser
    chamada depois que TODAS as tasks da Etapa 4 já terminaram (ex.: pela
    Etapa 5, que roda como job único com `--dependency=afterok:<job_id>`
    em relação ao array da Etapa 4).
    """
    parts = sorted(PATHS.shap_dir.glob("shap_manifest_*.csv"))
    if not parts:
        raise FileNotFoundError(
            f"Nenhum shap_manifest_<modelo>.csv encontrado em {PATHS.shap_dir}. "
            f"Rode a Etapa 4 (src/xai.py) para pelo menos 1 modelo antes."
        )

    dfs = [pd.read_csv(p) for p in parts]
    merged = pd.concat(dfs, ignore_index=True)
    # drop_duplicates por segurança, caso uma task tenha sido re-executada
    # manualmente (mantém a versão mais recente por [model, file_id])
    merged = merged.drop_duplicates(subset=["model", "file_id"], keep="last")

    merged.to_csv(PATHS.shap_manifest_path, index=False)
    logger.info(
        "Manifesto SHAP combinado salvo em %s (%d linhas, modelos: %s)",
        PATHS.shap_manifest_path, len(merged),
        sorted(merged["model"].unique().tolist()),
    )
    return merged