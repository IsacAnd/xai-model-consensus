"""
Ponto de entrada único do pipeline (Etapa 1 + Etapa 2 + Etapa 3 + Etapa 4).

Exemplos:
    python main.py --stage all --model all
    python main.py --stage organize
    python main.py --stage train --model efficientnet
    python main.py --stage xai --model cnn
    python main.py --stage similarity
"""

import argparse
import logging

from config import MODEL_NAMES
from src.data_organization import build_manifests, sample_xai_subset
from src.evaluate import evaluate_model
from src.similarity import run_similarity
from src.train import train_one_model
from src.xai import run_xai

logger = logging.getLogger(__name__)


def run_organize():
    manifests = build_manifests()
    sample_xai_subset(manifests["eval"])


def run_train(model: str):
    models_to_run = MODEL_NAMES if model == "all" else [model]
    for name in models_to_run:
        train_one_model(name)


def run_evaluate(model: str, split: str = "eval"):
    models_to_run = MODEL_NAMES if model == "all" else [model]
    for name in models_to_run:
        try:
            evaluate_model(name, split=split)
        except FileNotFoundError as e:
            logger.warning(str(e))


def main():
    parser = argparse.ArgumentParser(description="Pipeline ADD - Etapas 1, 2, 3 e 4")
    parser.add_argument("--stage",
                         choices=["organize", "train", "evaluate", "xai", "similarity", "all"],
                         default="all")
    parser.add_argument("--model", choices=MODEL_NAMES + ["all"], default="all")
    parser.add_argument("--eval-split", choices=["train", "dev", "eval"], default="eval")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.stage in ("organize", "all"):
        logger.info(">>> Etapa 1: organização dos dados")
        run_organize()

    if args.stage in ("train", "all"):
        logger.info(">>> Etapa 2: treinamento dos modelos")
        run_train(args.model)

    if args.stage in ("evaluate", "all"):
        logger.info(">>> Etapa 3: avaliação quantitativa (split=%s)", args.eval_split)
        run_evaluate(args.model, args.eval_split)

    if args.stage in ("xai", "all"):
        logger.info(">>> Etapa 4: geração dos mapas SHAP (matrizes + heatmaps)")
        run_xai(args.model)

    if args.stage in ("similarity", "all"):
        logger.info(">>> Etapa 5: métricas de comparação entre mapas SHAP")
        run_similarity()


if __name__ == "__main__":
    main()