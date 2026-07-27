"""
Etapa 5 - Métricas de comparação entre mapas SHAP de diferentes arquiteturas.

Carrega as matrizes SHAP brutas salvas na Etapa 4 (uma por par modelo x
file_id, em results/shap/matrices/<modelo>/<file_id>.npy), normaliza-as e
calcula, para cada par de modelos e cada exemplar do subset XAI (os mesmos
1000 exemplares sorteados na Etapa 1), cinco métricas de
similaridade/distância entre os respectivos mapas de importância:

  - cosseno      -> alinhamento global do vetor de importância
  - SSIM         -> similaridade estrutural espacial (mapa 2D tempo-frequência)
  - Spearman     -> correlação de ranking, robusta a diferenças de escala
                    entre a magnitude do SHAP de arquiteturas distintas
  - euclidiana   -> magnitude ponto a ponto da diferença entre os mapas
  - Jensen-Shannon -> diferença entre as distribuições de importância,
                    tratando |SHAP| normalizado (soma = 1) como uma "massa
                    de probabilidade" sobre os pixels tempo-frequência

Duas matrizes só são comparadas quando os dois modelos têm uma matriz SHAP
salva para o MESMO file_id - o pareamento é feito a partir de
results/shap/shap_manifest.csv (gerado na Etapa 4). Isso permite rodar a
Etapa 4 de forma incremental (ex.: um modelo por vez, via job array no
Slurm) e ainda assim comparar corretamente apenas os exemplares em comum.

Duas normalizações distintas são usadas (ver `_normalize_minmax` e
`_normalize_prob`) porque a magnitude bruta do SHAP não é comparável entre
arquiteturas diferentes (cada rede tem sua própria escala de gradientes);
sem normalizar, cosseno/SSIM/euclidiana/Spearman ficariam dominados pela
diferença de escala em vez da diferença de padrão espacial, e a JSD exige
por definição duas distribuições de probabilidade (soma = 1).

Saídas:
  results/similarity/pairwise_metrics.csv     - uma linha por (file_id, par de modelos)
  results/similarity/summary_by_pair.csv      - média/desvio/n por par de modelos,
                                                 agregando sobre todos os exemplares
                                                 em comum (a "matriz" de consenso
                                                 cross-architecture do artigo)
  results/similarity/consensus_per_sample.csv - média das 5 métricas sobre TODOS
                                                 os pares de modelos, por exemplar -
                                                 insumo cru para uma eventual Etapa 6
                                                 (métrica de consenso agregada)

Uso:
    python -m src.similarity
"""

import itertools
import logging

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim_fn

from config import PATHS, SIMILARITY
from src.utils import setup_logging

logger = logging.getLogger(__name__)

METRIC_COLS = ["cosine", "ssim", "spearman", "euclidean", "jsd"]


def _load_shap_manifest() -> pd.DataFrame:
    path = PATHS.shap_manifest_path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode a Etapa 4 (src/xai.py) antes da Etapa 5."
        )
    return pd.read_csv(path)


def _load_matrix(matrix_path: str) -> np.ndarray:
    """Carrega a matriz SHAP bruta (1, n_mels, frames) e retorna a
    magnitude |SHAP| - o sinal (contribuição a favor/contra a classe
    predita) não é diretamente comparável entre arquiteturas distintas,
    então as métricas de consenso comparam "onde" cada modelo colocou
    importância, não em que direção."""
    m = np.load(matrix_path).astype(np.float64)
    return np.abs(m)


def _normalize_minmax(m: np.ndarray) -> np.ndarray:
    """Normalização [0, 1], usada para cosseno, SSIM, Spearman e distância
    euclidiana: preserva a estrutura espacial relativa do mapa, mas coloca
    modelos diferentes na mesma escala."""
    lo, hi = float(m.min()), float(m.max())
    denom = hi - lo
    if denom < SIMILARITY.eps:
        return np.zeros_like(m)
    return (m - lo) / denom


def _normalize_prob(m: np.ndarray) -> np.ndarray:
    """Normalização por soma = 1, usada apenas para a JSD (trata |SHAP|
    como uma distribuição de massa de importância sobre os pixels
    tempo-frequência)."""
    total = float(m.sum())
    if total < SIMILARITY.eps:
        # matriz ~toda zero (modelo "sem opinião" no exemplar): distribuição
        # uniforme, para manter a JSD bem definida em vez de propagar NaN
        return np.full_like(m, 1.0 / m.size)
    return m / total


def compare_pair(matrix_a: np.ndarray, matrix_b: np.ndarray) -> dict:
    """Calcula as 5 métricas de comparação entre dois mapas SHAP com a
    MESMA shape (1, n_mels, frames) - garantida pelo crop/pad padronizado
    do pipeline (Etapas 2/4, via AUDIO.expected_frames)."""
    a2d, b2d = matrix_a.squeeze(0), matrix_b.squeeze(0)

    a_mm, b_mm = _normalize_minmax(a2d), _normalize_minmax(b2d)
    a_flat, b_flat = a_mm.ravel(), b_mm.ravel()

    # cosseno (alinhamento global)
    norm_a, norm_b = np.linalg.norm(a_flat), np.linalg.norm(b_flat)
    if norm_a < SIMILARITY.eps or norm_b < SIMILARITY.eps:
        cosine = float("nan")
    else:
        cosine = float(np.dot(a_flat, b_flat) / (norm_a * norm_b))

    # SSIM (estrutura espacial 2D, direto no mapa normalizado [0, 1])
    ssim_val = float(ssim_fn(a_mm, b_mm, data_range=SIMILARITY.ssim_data_range))

    # Spearman (correlação de ranking - já é invariante a normalizações
    # monotônicas, mas usa-se o mapa normalizado por consistência com as
    # demais métricas e para lidar de forma idêntica com o caso degenerado
    # de matriz constante)
    if a_flat.std() < SIMILARITY.eps or b_flat.std() < SIMILARITY.eps:
        spearman_corr = float("nan")
    else:
        spearman_corr, _ = spearmanr(a_flat, b_flat)
        spearman_corr = float(spearman_corr)

    # distância euclidiana (magnitude ponto a ponto da diferença)
    euclidean = float(np.linalg.norm(a_flat - b_flat))

    # Jensen-Shannon (distribuição de importância)
    a_p, b_p = _normalize_prob(a2d).ravel(), _normalize_prob(b2d).ravel()
    jsd_distance = float(jensenshannon(a_p, b_p, base=SIMILARITY.jsd_log_base))
    # scipy.spatial.distance.jensenshannon retorna a DISTÂNCIA (raiz da
    # divergência); eleva ao quadrado para reportar a divergência em si
    jsd = jsd_distance ** 2

    return {
        "cosine": cosine,
        "ssim": ssim_val,
        "spearman": spearman_corr,
        "euclidean": euclidean,
        "jsd": jsd,
    }


def compute_pairwise_metrics(manifest: pd.DataFrame = None) -> pd.DataFrame:
    """Para cada file_id do subset XAI, calcula as 5 métricas entre CADA par
    de modelos que tenha uma matriz SHAP salva para aquele exemplar."""
    if manifest is None:
        manifest = _load_shap_manifest()

    rows = []
    grouped = manifest.groupby("file_id")
    n_files = len(grouped)
    logger.info("Comparando mapas SHAP entre modelos para %d exemplares...", n_files)

    for i, (file_id, group) in enumerate(grouped, start=1):
        # reruns da Etapa 4 para o mesmo modelo já são deduplicados dentro
        # de src/xai.py (substitui o manifest antigo daquele modelo), mas
        # mantém-se essa proteção por robustez
        group = group.drop_duplicates(subset="model", keep="last")
        available_models = sorted(group["model"].unique())
        if len(available_models) < 2:
            continue

        label = group["label"].iloc[0]
        matrices = {
            row["model"]: _load_matrix(row["matrix_path"])
            for _, row in group.iterrows()
        }

        for model_a, model_b in itertools.combinations(available_models, 2):
            metrics = compare_pair(matrices[model_a], matrices[model_b])
            rows.append({
                "file_id": file_id,
                "label": label,
                "model_a": model_a,
                "model_b": model_b,
                **metrics,
            })

        if i % 100 == 0 or i == n_files:
            logger.info("  %d/%d exemplares comparados", i, n_files)

    return pd.DataFrame(rows)


def summarize_by_pair(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Média/desvio-padrão/n de cada métrica, agregado por par de modelos
    (sobre todos os exemplares em que ambos apareceram) - a matriz de
    consenso cross-architecture, ponto central da Etapa 5 do artigo."""
    summary = (
        pairwise.groupby(["model_a", "model_b"])[METRIC_COLS]
        .agg(["mean", "std", "count"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("cosine_mean", ascending=False)
    return summary


def summarize_by_sample(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Média das 5 métricas sobre TODOS os pares de modelos, por exemplar -
    um proxy simples de quão consensual foi a explicação daquele exemplar
    entre as 6 arquiteturas. Serve como insumo cru para uma eventual
    Etapa 6 (métrica de consenso agregada, com pesos/combinação a definir),
    e já permite, por si só, inspecionar quais exemplares geram mais
    divergência entre os modelos."""
    consensus = (
        pairwise.groupby(["file_id", "label"])[METRIC_COLS]
        .mean()
        .reset_index()
        .rename(columns={c: f"mean_{c}" for c in METRIC_COLS})
    )
    return consensus


def run_similarity() -> None:
    PATHS.ensure_dirs()
    manifest = _load_shap_manifest()

    pairwise = compute_pairwise_metrics(manifest)
    if pairwise.empty:
        logger.warning(
            "Nenhum par de modelos com SHAP salvo para o mesmo exemplar - "
            "rode a Etapa 4 (src/xai.py) para pelo menos 2 modelos antes da Etapa 5."
        )
        return

    pairwise_path = PATHS.similarity_pairwise_path
    pairwise.to_csv(pairwise_path, index=False)
    logger.info("Métricas par-a-par salvas em %s (%d linhas)", pairwise_path, len(pairwise))

    summary = summarize_by_pair(pairwise)
    summary_path = PATHS.similarity_summary_path
    summary.to_csv(summary_path, index=False)
    logger.info("Resumo por par de modelos salvo em %s\n%s",
                summary_path, summary.to_string(index=False))

    consensus = summarize_by_sample(pairwise)
    consensus_path = PATHS.similarity_consensus_path
    consensus.to_csv(consensus_path, index=False)
    logger.info("Consenso por exemplar salvo em %s (%d linhas)",
                consensus_path, len(consensus))


def main():
    setup_logging(PATHS.log_dir / "similarity.log")
    run_similarity()


if __name__ == "__main__":
    main()