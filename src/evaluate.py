"""
Etapa 3 - Avaliação quantitativa.

Roda o(s) melhor(es) checkpoint(s) (salvos pela Etapa 2) sobre o split de
`eval` OFICIAL COMPLETO do ASVspoof5 (não o subset de 1000 usado depois na
Etapa 4/XAI) e reporta accuracy, precision, recall, f1, roc-auc e eer.

Uso:
    python -m src.evaluate --model cnn
    python -m src.evaluate --model all      # avalia os 6 modelos e gera tabela comparativa

Saídas:
    results/logs/eval_<modelo>.json   -> métricas detalhadas de cada modelo
    results/logs/eval_summary.csv     -> tabela comparativa (uma linha por modelo)
"""

import argparse
import json
import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import MODEL_NAMES, PATHS, TRAIN
from src.dataset import ADDDataset, collate_fn, load_manifest
from src.metrics import compute_all_metrics
from src.models import build_model
from src.utils import setup_logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_inference(model, loader, device):
    import numpy as np
    model.eval()
    all_probs, all_labels, all_ids = [], [], []
    for x, y, ids in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())
        all_ids.extend(ids)
    return np.concatenate(all_labels), np.concatenate(all_probs), all_ids


def evaluate_model(model_name: str, split: str = "eval") -> dict:
    PATHS.ensure_dirs()
    checkpoint_path = PATHS.checkpoint_dir / f"{model_name}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_path}. "
            f"Rode a Etapa 2 (src/train.py --model {model_name}) primeiro."
        )

    device = torch.device(TRAIN.device if torch.cuda.is_available() else "cpu")
    # weights_only=False: necessário porque o checkpoint carrega dicts extras
    # (val_metrics, audio_config), não só tensores. Seguro aqui porque o
    # próprio pipeline (train.py) gerou o arquivo - não é um checkpoint de
    # terceiros de fonte não confiável.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_model(model_name, n_classes=TRAIN.n_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Checkpoint carregado: %s (epoch=%d, val_%s=%.4f)",
                checkpoint_path, ckpt["epoch"], TRAIN.early_stopping_metric,
                ckpt["val_metrics"][TRAIN.early_stopping_metric])

    manifest = load_manifest(split)
    ds = ADDDataset(manifest, split=split)
    loader = DataLoader(ds, batch_size=TRAIN.batch_size, shuffle=False,
                         num_workers=TRAIN.num_workers, collate_fn=collate_fn)

    y_true, y_score, file_ids = run_inference(model, loader, device)
    y_pred = (y_score >= 0.5).astype(int)
    metrics = compute_all_metrics(y_true, y_pred, y_score)
    metrics["model"] = model_name
    metrics["split"] = split
    metrics["n_samples"] = int(len(y_true))

    logger.info(
        "[%s] acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f eer=%.4f (n=%d)",
        model_name, metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1"], metrics["roc_auc"], metrics["eer"], metrics["n_samples"],
    )

    out_path = PATHS.log_dir / f"eval_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Métricas salvas em %s", out_path)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Etapa 3 - Avaliação quantitativa")
    parser.add_argument("--model", type=str, default="all",
                         choices=MODEL_NAMES + ["all"])
    parser.add_argument("--split", type=str, default="eval",
                         choices=["train", "dev", "eval"],
                         help="Split a avaliar (padrão: eval oficial completo)")
    args = parser.parse_args()

    setup_logging(PATHS.log_dir / "evaluate.log")

    models_to_run = MODEL_NAMES if args.model == "all" else [args.model]
    all_metrics = []
    for name in models_to_run:
        try:
            all_metrics.append(evaluate_model(name, split=args.split))
        except FileNotFoundError as e:
            logger.warning(str(e))

    if all_metrics:
        summary = pd.DataFrame(all_metrics)[
            ["model", "split", "n_samples", "accuracy", "precision",
             "recall", "f1", "roc_auc", "eer"]
        ].sort_values("eer")
        summary_path = PATHS.log_dir / "eval_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Resumo comparativo salvo em %s\n%s", summary_path,
                     summary.to_string(index=False))


if __name__ == "__main__":
    main()