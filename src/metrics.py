import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              roc_auc_score, roc_curve)


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Equal Error Rate: ponto em que FPR (false alarm) == FNR (miss rate).
    y_score deve ser a probabilidade/score da classe "spoof" (classe positiva = 1).
    Retorna NaN se y_true tiver só uma classe (EER indefinido nesse caso).
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    diff = np.abs(fpr - fnr)
    if np.all(np.isnan(diff)):
        return float("nan")
    idx = np.nanargmin(diff)
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer)


def compute_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    FIX: threshold ótimo pelo ponto de EER (onde FPR ~= FNR), em vez do
    0.5 fixo usado antes em train.py/evaluate.py. Com classes desbalanceadas
    (ASVspoof5), 0.5 quase nunca é o corte ideal e é o que estava produzindo
    precision alta / recall baixo mesmo com boa capacidade discriminativa
    (AUC/EER estáveis). Retorna 0.5 como fallback se y_true tiver só 1 classe.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    diff = np.abs(fpr - fnr)
    if np.all(np.isnan(diff)):
        return 0.5
    idx = np.nanargmin(diff)
    return float(thresholds[idx])


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         y_score: np.ndarray) -> dict:
    """
    y_true: rótulos binários (0=bonafide, 1=spoof)
    y_pred: predições binárias (0/1) após threshold (ex.: argmax ou threshold ótimo)
    y_score: probabilidade da classe 1 (spoof), usada para ROC-AUC e EER
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")  # ocorre se só houver uma classe no batch/split
    eer = compute_eer(y_true, y_score)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "eer": eer,
    }