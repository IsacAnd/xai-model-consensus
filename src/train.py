"""
Etapa 2 - Treinamento dos modelos.

Hiperparâmetros padronizados entre todos os modelos (config.TRAIN):
  - Optimizer: AdamW
  - Batch size: 32
  - Epochs: até 50 (com early stopping)
  - LR: 3e-4, com ReduceLROnPlateau (mesmo scheduler para todos)
  - Critério de seleção: menor EER de validação (early_stopping_metric)

Uso:
    python -m src.train --model cnn
    python -m src.train --model all      # treina os 6 modelos em sequência
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import AUDIO, MODEL_NAMES, PATHS, TRAIN
from src.dataset import ADDDataset, collate_fn, load_manifest
from src.metrics import compute_all_metrics
from src.models import build_model
from src.utils import EarlyStopping, set_seed, setup_logging

logger = logging.getLogger(__name__)


def build_optimizer(model: nn.Module):
    return torch.optim.AdamW(model.parameters(), lr=TRAIN.lr,
                              weight_decay=TRAIN.weight_decay)


def build_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=TRAIN.scheduler_factor,
        patience=TRAIN.scheduler_patience,
    )


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    all_probs, all_labels, losses = [], [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item() * x.size(0))
        probs = torch.softmax(logits, dim=1)[:, 1]  # prob. classe "spoof"
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    y_score = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = compute_all_metrics(y_true, y_pred, y_score)
    metrics["loss"] = float(np.sum(losses) / len(y_true))
    return metrics


def train_one_model(model_name: str):
    set_seed(TRAIN.seed)
    PATHS.ensure_dirs()

    log_path = PATHS.log_dir / f"train_{model_name}.log"
    setup_logging(log_path)
    logger.info("==== Treinando modelo: %s ====", model_name)
    logger.info("Config áudio: %s", AUDIO)
    logger.info("Config treino: %s", TRAIN)

    device = torch.device(TRAIN.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("CUDA não disponível, treinando em CPU (lento).")

    # --- dados ---
    train_manifest = load_manifest("train")
    dev_manifest = load_manifest("dev")

    train_ds = ADDDataset(train_manifest, split="train")
    dev_ds = ADDDataset(dev_manifest, split="dev")

    train_loader = DataLoader(
        train_ds, batch_size=TRAIN.batch_size, shuffle=True,
        num_workers=TRAIN.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=TRAIN.batch_size, shuffle=False,
        num_workers=TRAIN.num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )

    # --- modelo / otimização ---
    model = build_model(model_name, n_classes=TRAIN.n_classes).to(device)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=TRAIN.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=TRAIN.mixed_precision and device.type == "cuda")

    early_stopper = EarlyStopping(
        patience=TRAIN.early_stopping_patience,
        mode=TRAIN.early_stopping_mode,
    )

    checkpoint_path = PATHS.checkpoint_dir / f"{model_name}_best.pt"
    history = []

    for epoch in range(1, TRAIN.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=TRAIN.mixed_precision and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)

        train_loss = running_loss / n_seen
        val_metrics = evaluate(model, dev_loader, device, criterion)
        scheduler.step(val_metrics["loss"])

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %02d/%d | train_loss=%.4f | val_loss=%.4f | val_eer=%.4f | "
            "val_acc=%.4f | val_f1=%.4f | val_auc=%.4f | lr=%.2e | %.1fs",
            epoch, TRAIN.epochs, train_loss, val_metrics["loss"], val_metrics["eer"],
            val_metrics["accuracy"], val_metrics["f1"], val_metrics["roc_auc"],
            current_lr, elapsed,
        )

        history.append({"epoch": epoch, "train_loss": train_loss, "lr": current_lr,
                         **{f"val_{k}": v for k, v in val_metrics.items()}})

        selection_score = val_metrics[TRAIN.early_stopping_metric]
        is_best = early_stopper.step(selection_score)
        if is_best:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "model_name": model_name,
                "audio_config": AUDIO.__dict__,
            }, checkpoint_path)
            logger.info("  -> novo melhor checkpoint salvo (%s=%.4f)",
                         TRAIN.early_stopping_metric, selection_score)

        if early_stopper.should_stop:
            logger.info("Early stopping ativado na epoch %d (sem melhora por %d épocas).",
                         epoch, TRAIN.early_stopping_patience)
            break

    # salva histórico completo de treino
    history_path = PATHS.log_dir / f"history_{model_name}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Histórico salvo em %s", history_path)
    logger.info("Melhor %s de validação: %.4f", TRAIN.early_stopping_metric,
                early_stopper.best_score)

    return checkpoint_path, history


def main():
    parser = argparse.ArgumentParser(description="Etapa 2 - Treinamento dos modelos ADD")
    parser.add_argument("--model", type=str, default="all",
                         choices=MODEL_NAMES + ["all"],
                         help="Qual modelo treinar (ou 'all' para treinar todos em sequência)")
    args = parser.parse_args()

    models_to_run = MODEL_NAMES if args.model == "all" else [args.model]
    for name in models_to_run:
        train_one_model(name)


if __name__ == "__main__":
    main()
