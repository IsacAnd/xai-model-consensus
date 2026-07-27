"""
Etapa 4 - Geração dos mapas SHAP (matrizes brutas + heatmaps).

Para cada um dos 6 modelos, carrega o melhor checkpoint (Etapa 2) e roda
shap.GradientExplainer sobre os 1000 exemplares de data/xai_subset_1000.csv
(gerado na Etapa 1 por src/data_organization.py:sample_xai_subset).

O GradientExplainer foi escolhido por funcionar de forma uniforme nas 6
arquiteturas (incluindo CRNN, com GRU, e ViT, com atenção) sem precisar de
tratamento especial por modelo, diferente do DeepExplainer.

A distribuição de referência ("background") do explainer é uma amostra
aleatória (seed fixa, config.XAI.seed) de config.XAI.n_background exemplos
do split de TREINO — os mesmos manifests/cache de log-mel usados nas
Etapas 2/3 (via src.dataset.ADDDataset + collate_fn), então não é preciso
reimplementar nada do carregamento/crop/cache de áudio aqui.

Para cada exemplar do subset:
  - o SHAP é calculado em relação à classe PREDITA pelo modelo (explica a
    decisão do modelo, inclusive nos casos de erro em relação ao rótulo real)
  - a matriz bruta (1, n_mels, frames) é salva em .npy — insumo puro,
    NÃO normalizado, para a Etapa 5 (cosseno, SSIM, Spearman, distância
    euclidiana, JSD)
  - um heatmap .png (log-mel + |SHAP| normalizado só para exibição) é salvo
    para inspeção visual / figuras do artigo

Uso:
    python -m src.xai --model cnn
    python -m src.xai --model all      # roda os 6 modelos em sequência

Saídas:
    results/shap/matrices/<modelo>/<file_id>.npy
    results/shap/heatmaps/<modelo>/<file_id>.png
    results/shap/shap_manifest.csv   -> model, file_id, label, pred, matrix_path, heatmap_path
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

from config import MODEL_NAMES, PATHS, TRAIN, XAI
from src.dataset import ADDDataset, collate_fn, load_manifest
from src.models import build_model
from src.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


def _load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    checkpoint_path = PATHS.checkpoint_dir / f"{model_name}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_path}. "
            f"Rode a Etapa 2 (src/train.py --model {model_name}) antes da Etapa 4."
        )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(model_name, n_classes=TRAIN.n_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Checkpoint carregado: %s (epoch=%d, val_%s=%.4f)",
                checkpoint_path, ckpt["epoch"], TRAIN.early_stopping_metric,
                ckpt["val_metrics"][TRAIN.early_stopping_metric])
    return model


def _load_xai_subset() -> pd.DataFrame:
    path = PATHS.manifest_dir / "xai_subset_1000.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode a Etapa 1 (src/data_organization.py) primeiro."
        )
    return pd.read_csv(path)


def _background_batch(device: torch.device) -> torch.Tensor:
    """Amostra fixa (seed XAI.seed) de XAI.n_background exemplos do split de
    treino, usada como distribuição de referência do GradientExplainer."""
    train_manifest = load_manifest("train")
    train_ds = ADDDataset(train_manifest, split="train")

    rng = np.random.RandomState(XAI.seed)
    n = min(XAI.n_background, len(train_ds))
    idx = rng.choice(len(train_ds), size=n, replace=False).tolist()

    loader = DataLoader(Subset(train_ds, idx), batch_size=n, shuffle=False,
                         collate_fn=collate_fn)
    x, _, _ = next(iter(loader))
    return x.to(device)


def _extract_sample_shap(shap_values, sample_idx: int, class_idx: int) -> np.ndarray:
    """Extrai a matriz SHAP (1, n_mels, frames) de um exemplar/classe,
    compatível com as duas convenções de retorno do shap.GradientExplainer
    para modelos multi-classe:
      - lista de arrays, um por classe, cada um (batch, 1, n_mels, frames)
        [formato usado em versões mais antigas do shap, < 0.45]
      - array único (batch, 1, n_mels, frames, n_classes), com a dimensão
        de classes empilhada no último eixo [formato usado a partir do
        shap >= 0.45, ver changelog do shap]
    """
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[class_idx])[sample_idx]
    else:
        arr = np.asarray(shap_values)
        sv = arr[sample_idx, ..., class_idx]

    if sv.ndim != 3:
        raise ValueError(
            f"Shape inesperado ao extrair a matriz SHAP: {sv.shape} (esperado "
            f"3 dimensões, ex.: (1, n_mels, frames)). O formato de retorno de "
            f"shap.GradientExplainer.shap_values() pode ter mudado de novo na "
            f"sua versão do shap - ajuste _extract_sample_shap()."
        )
    return sv


def _normalize_for_plot(matrix: np.ndarray) -> np.ndarray:
    """Normaliza |SHAP| para [0, 1] só para o heatmap. A matriz bruta salva
    em disco (.npy) não passa por isso — a Etapa 5 normaliza a partir dela."""
    m = np.abs(matrix)
    m = m - m.min()
    denom = m.max()
    if denom > 1e-12:
        m = m / denom
    return m


def _save_heatmap(spectrogram: np.ndarray, shap_matrix: np.ndarray,
                   out_path: Path, title: str) -> None:
    spec = np.asarray(spectrogram).squeeze()
    heat = _normalize_for_plot(shap_matrix).squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(spec, origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title("log-mel")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("mel bin")

    axes[1].imshow(spec, origin="lower", aspect="auto", cmap="gray_r")
    im = axes[1].imshow(heat, origin="lower", aspect="auto", cmap=XAI.cmap, alpha=0.6)
    axes[1].set_title("SHAP |importância| (normalizado p/ plot)")
    axes[1].set_xlabel("frame")

    fig.suptitle(title, fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def generate_shap_maps(model_name: str) -> None:
    set_seed(XAI.seed)
    PATHS.ensure_dirs()

    device = torch.device(TRAIN.device if torch.cuda.is_available() else "cpu")
    model = _load_model(model_name, device)

    xai_subset = _load_xai_subset()
    explain_ds = ADDDataset(xai_subset, split="eval")
    explain_loader = DataLoader(
        explain_ds, batch_size=XAI.shap_batch_size, shuffle=False,
        collate_fn=collate_fn,
    )

    background = _background_batch(device)
    explainer = shap.GradientExplainer(model, background)

    matrix_dir = PATHS.shap_matrix_dir / model_name
    heatmap_dir = PATHS.shap_heatmap_dir / model_name
    matrix_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(explain_ds)
    logger.info("Gerando mapas SHAP para '%s' (%d exemplares, batch=%d, background=%d, device=%s)",
                model_name, n_total, XAI.shap_batch_size, background.shape[0], device)

    rows = []
    n_done = 0
    for x, y, file_ids in explain_loader:
        x = x.to(device)
        shap_values = explainer.shap_values(x, nsamples=XAI.gradient_nsamples)
        # formato varia por versão do shap - ver _extract_sample_shap acima

        with torch.no_grad():
            preds = model(x).argmax(dim=1).cpu().numpy()

        for j, file_id in enumerate(file_ids):
            pred_class = int(preds[j])
            sv = _extract_sample_shap(shap_values, j, pred_class)  # (1, n_mels, frames)

            matrix_path = matrix_dir / f"{file_id}.npy"
            np.save(matrix_path, sv.astype(np.float32))

            heatmap_path = heatmap_dir / f"{file_id}.png"
            _save_heatmap(
                x[j].detach().cpu().numpy(), sv, heatmap_path,
                title=f"{model_name} | {file_id} | label={int(y[j])} pred={pred_class}",
            )

            rows.append({
                "model": model_name,
                "file_id": file_id,
                "label": int(y[j]),
                "pred": pred_class,
                "matrix_path": str(matrix_path),
                "heatmap_path": str(heatmap_path),
            })

        n_done += len(file_ids)
        logger.info("  [%s] %d/%d exemplares processados", model_name, n_done, n_total)

    df_new = pd.DataFrame(rows)
    manifest_path = PATHS.shap_manifest_path
    if manifest_path.exists():
        df_old = pd.read_csv(manifest_path)
        df_old = df_old[df_old["model"] != model_name]  # substitui reruns antigos do mesmo modelo
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(manifest_path, index=False)
    logger.info("Manifesto SHAP atualizado: %s (%d linhas)", manifest_path, len(df_new))


def run_xai(model: str) -> None:
    models_to_run = MODEL_NAMES if model == "all" else [model]
    for name in models_to_run:
        try:
            generate_shap_maps(name)
        except FileNotFoundError as e:
            logger.warning(str(e))


def main():
    parser = argparse.ArgumentParser(description="Etapa 4 - geração dos mapas SHAP")
    parser.add_argument("--model", type=str, default="all", choices=MODEL_NAMES + ["all"])
    args = parser.parse_args()

    setup_logging(PATHS.log_dir / "xai.log")
    run_xai(args.model)


if __name__ == "__main__":
    main()