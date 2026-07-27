"""Utilitários compartilhados: early stopping, seed, logging setup."""

import logging
import random

import numpy as np
import torch


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
