import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    cohen_kappa_score, mean_absolute_error, roc_auc_score,
)
from typing import Dict


def evaluar(
    y_true,
    y_pred,
    y_prob=None,
    nombre: str = "",
    estrategia_balanceo: str = "",
    variante_target: str = "ordinal_4clases",
    split: str = "test",
) -> Dict:
    """
    Calcula el conjunto completo de métricas de evaluación.

    Métricas calculadas:
    - accuracy           : exactitud estándar (% predicciones correctas)
    - balanced_accuracy  : exactitud balanceada (media del recall por clase)
    - f1_macro           : F1 promedio con igual peso por clase
    - f1_weighted        : F1 ponderado por frecuencia de clase
    - kappa_lineal       : Kappa con penalización lineal
    - kappa_cuadratico   : Kappa con penalización cuadrática (MÉTRICA PRINCIPAL)
    - mae_ordinal        : Error Absoluto Medio tratando clases como enteros
    - auroc_macro        : Área bajo la curva ROC promediada OvR

    Parámetros
    ----------
    y_true              : etiquetas reales
    y_pred              : predicciones
    y_prob              : probabilidades por clase (para AUROC, opcional)
    nombre              : nombre del modelo
    estrategia_balanceo : 'sin_balanceo', 'pesos_clase' o 'smotenc'
    variante_target     : 'ordinal_4clases', 'binario' o 'likert_continuo'
    split               : 'train', 'val' o 'test'
    """
    metricas = {
        "modelo"              : nombre,
        "estrategia_balanceo" : estrategia_balanceo,
        "variante_target"     : variante_target,
        "split"               : split,
        "accuracy"            : accuracy_score(y_true, y_pred),
        "balanced_accuracy"   : balanced_accuracy_score(y_true, y_pred),
        "f1_macro"            : f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted"         : f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa_lineal"        : cohen_kappa_score(y_true, y_pred, weights="linear"),
        "kappa_cuadratico"    : cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mae_ordinal"         : mean_absolute_error(y_true, y_pred),
        "auroc_macro"         : np.nan,
    }
    if y_prob is not None:
        try:
            metricas["auroc_macro"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except Exception:
            pass

    if nombre:
        print(f"  {'─'*52}")
        print(f"  {nombre} | {estrategia_balanceo} | {variante_target} [{split}]")
        print(f"  {'─'*52}")
        omitir = ("modelo", "estrategia_balanceo", "variante_target", "split")
        for k, v in metricas.items():
            if k in omitir:
                continue
            marca = " ← PRINCIPAL" if k == "kappa_cuadratico" else ""
            print(f"    {k:<22}: {v:.4f}{marca}" if isinstance(v, float)
                  else f"    {k}: {v}")
    return metricas
