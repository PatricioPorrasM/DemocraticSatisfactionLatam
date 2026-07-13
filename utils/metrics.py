import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, cohen_kappa_score,
    mean_absolute_error, roc_auc_score,
)
from typing import Dict


def evaluar(
    y_true, y_pred, y_prob=None, nombre: str = "", subperiodo: str = "", split: str = "test",
) -> Dict:
    metricas = {
        "modelo"           : nombre,
        "subperiodo"       : subperiodo,
        "split"            : split,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro"         : f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted"      : f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa_lineal"     : cohen_kappa_score(y_true, y_pred, weights="linear"),
        "kappa_cuadratico" : cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mae_ordinal"      : mean_absolute_error(y_true, y_pred),
        "auroc_macro"      : np.nan,
    }
    if y_prob is not None:
        try:
            metricas["auroc_macro"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except Exception:
            pass
    if nombre:
        print(f"  {'─'*48}")
        print(f"  {nombre} — {subperiodo} [{split}]")
        print(f"  {'─'*48}")
        for k, v in metricas.items():
            if k in ("modelo", "subperiodo"):
                continue
            print(f"    {k:<22}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    return metricas
