"""
utils/metrics.py
================
Métricas de evaluación del proyecto.

Contenido:
- `evaluar()`                : conjunto de 8 métricas agregadas (una fila por modelo).
- `metricas_por_clase()`     : precision / recall / F1 / soporte por categoría.
- `matriz_confusion_df()`    : matriz de confusión etiquetada (conteos o porcentajes).
- `reporte_detallado()`      : reporte completo del modelo principal (agregadas +
                               por categoría + matriz de confusión) con guardado en CSV.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    cohen_kappa_score, mean_absolute_error, roc_auc_score,
    confusion_matrix, precision_recall_fscore_support,
)
from typing import Dict, Optional

from .config import ETIQUETAS, PATHS


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS AGREGADAS
# ─────────────────────────────────────────────────────────────────────────────

def evaluar(
    y_true,
    y_pred,
    y_prob=None,
    nombre: str = "",
    estrategia_balanceo: str = "",
    variante_target: str = "ordinal_4clases",
    split: str = "test",
    verboso: bool = True,
) -> Dict:
    """
    Calcula el conjunto completo de métricas agregadas de evaluación.

    Métricas calculadas:
    - accuracy           : exactitud estándar (% predicciones correctas)
    - balanced_accuracy  : exactitud balanceada (media del recall por clase)
    - f1_macro           : F1 promedio con igual peso por clase
    - f1_weighted        : F1 ponderado por frecuencia de clase
    - kappa_lineal       : Kappa con penalización lineal
    - kappa_cuadratico   : Kappa con penalización cuadrática (MÉTRICA PRINCIPAL)
    - mae_ordinal        : Error Absoluto Medio tratando clases como enteros
    - auroc_macro        : Área bajo la curva ROC promediada OvR

    Para el desglose por categoría y la matriz de confusión del modelo
    principal, usar `reporte_detallado()`.

    Parámetros
    ----------
    y_true              : etiquetas reales
    y_pred              : predicciones
    y_prob              : probabilidades por clase (para AUROC, opcional)
    nombre              : nombre del modelo
    estrategia_balanceo : 'sin_balanceo', 'pesos_clase' o 'smotenc'
    variante_target     : 'ordinal_4clases' o 'binario'
    split               : 'train', 'val' o 'test'
    verboso             : si False, no imprime el resumen por consola
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
            y_prob_arr = np.asarray(y_prob)
            if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2:
                # Caso binario: roc_auc_score espera la probabilidad de la clase positiva
                metricas["auroc_macro"] = roc_auc_score(y_true, y_prob_arr[:, 1])
            else:
                metricas["auroc_macro"] = roc_auc_score(
                    y_true, y_prob_arr, multi_class="ovr", average="macro"
                )
        except Exception:
            pass

    if nombre and verboso:
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


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS POR CATEGORÍA Y MATRIZ DE CONFUSIÓN
# ─────────────────────────────────────────────────────────────────────────────

def _clases_presentes(y_true, y_pred) -> list:
    """Lista ordenada de clases observadas en y_true o y_pred."""
    vals = set(np.asarray(y_true).ravel().tolist()) | set(np.asarray(y_pred).ravel().tolist())
    return sorted(int(v) for v in vals)


def _nombres_clases(clases: list, etiquetas: Optional[Dict] = None) -> list:
    """
    Nombres legibles para cada clase.

    Usa `etiquetas` si se entrega; si no, usa ETIQUETAS del proyecto cuando
    el número de clases coincide (variante ordinal de 4 clases) y en caso
    contrario genera etiquetas genéricas (variante binaria u otras).
    """
    if etiquetas is None:
        etiquetas = ETIQUETAS if len(clases) == len(ETIQUETAS) else {}
    return [f"{c} — {etiquetas[c]}" if c in etiquetas else f"Clase {c}" for c in clases]


def metricas_por_clase(
    y_true,
    y_pred,
    etiquetas: Optional[Dict] = None,
    incluir_promedios: bool = True,
) -> pd.DataFrame:
    """
    Precision, recall, F1 y soporte para cada categoría del target.

    Parámetros
    ----------
    y_true            : etiquetas reales
    y_pred            : predicciones
    etiquetas         : dict {clase: etiqueta}. Por defecto ETIQUETAS del proyecto.
    incluir_promedios : añade las filas de promedio macro y ponderado.

    Retorna
    -------
    DataFrame con columnas: clase, etiqueta, precision, recall, f1, soporte,
    pct_soporte. Las filas de promedio llevan clase = NaN.
    """
    clases  = _clases_presentes(y_true, y_pred)
    nombres = _nombres_clases(clases, etiquetas)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=clases, zero_division=0)
    n_total = int(np.sum(s))

    filas = [
        {"clase": c, "etiqueta": n, "precision": round(float(pi), 4),
         "recall": round(float(ri), 4), "f1": round(float(fi), 4),
         "soporte": int(si),
         "pct_soporte": round(float(si) / n_total * 100, 2) if n_total else np.nan}
        for c, n, pi, ri, fi, si in zip(clases, nombres, p, r, f, s)
    ]

    if incluir_promedios:
        for nombre_prom, promedio in [("Promedio macro", "macro"),
                                      ("Promedio ponderado", "weighted")]:
            pp, rp, fp, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=clases, average=promedio, zero_division=0)
            filas.append({
                "clase": np.nan, "etiqueta": nombre_prom,
                "precision": round(float(pp), 4), "recall": round(float(rp), 4),
                "f1": round(float(fp), 4), "soporte": n_total, "pct_soporte": 100.0,
            })

    df_out = pd.DataFrame(filas)
    df_out["clase"] = df_out["clase"].astype("Int64")   # entero con soporte de NA
    return df_out


def matriz_confusion_df(
    y_true,
    y_pred,
    etiquetas: Optional[Dict] = None,
    normalizar: Optional[str] = None,
) -> pd.DataFrame:
    """
    Matriz de confusión con índices y columnas etiquetados.

    Parámetros
    ----------
    y_true     : etiquetas reales
    y_pred     : predicciones
    etiquetas  : dict {clase: etiqueta}. Por defecto ETIQUETAS del proyecto.
    normalizar : None → conteos absolutos.
                 'true' → % sobre el total de cada clase real (por fila).
                 'pred' → % sobre el total de cada clase predicha (por columna).
                 'all'  → % sobre el total de registros.

    Retorna
    -------
    DataFrame (real × predicho).
    """
    clases  = _clases_presentes(y_true, y_pred)
    nombres = _nombres_clases(clases, etiquetas)
    cm = confusion_matrix(y_true, y_pred, labels=clases, normalize=normalizar)
    if normalizar is not None:
        cm = cm * 100
    return pd.DataFrame(
        cm,
        index=pd.Index(nombres, name="real"),
        columns=pd.Index(nombres, name="predicho"),
    )


def reporte_detallado(
    y_true,
    y_pred,
    y_prob=None,
    nombre: str = "",
    estrategia_balanceo: str = "",
    variante_target: str = "ordinal_4clases",
    split: str = "test",
    etiquetas: Optional[Dict] = None,
    guardar: bool = True,
    verboso: bool = True,
) -> Dict:
    """
    Reporte completo de evaluación para el modelo principal.

    Combina las métricas agregadas de `evaluar()` con el desglose por
    categoría (`metricas_por_clase`) y la matriz de confusión en conteos
    y en porcentajes por fila (`matriz_confusion_df`).

    Parámetros
    ----------
    y_true, y_pred, y_prob : entradas de evaluación (y_prob solo para AUROC).
    nombre                 : nombre del modelo.
    estrategia_balanceo    : estrategia de balanceo usada.
    variante_target        : 'ordinal_4clases' o 'binario'.
    split                  : conjunto evaluado.
    etiquetas              : dict {clase: etiqueta}.
    guardar                : si True escribe los CSV en results/tables/.
    verboso                : si True imprime todo el reporte.

    Retorna
    -------
    dict con claves: 'metricas', 'por_clase', 'confusion', 'confusion_pct'.
    """
    metricas  = evaluar(y_true, y_pred, y_prob, nombre, estrategia_balanceo,
                        variante_target, split, verboso=verboso)
    por_clase = metricas_por_clase(y_true, y_pred, etiquetas)
    cm_conteo = matriz_confusion_df(y_true, y_pred, etiquetas)
    cm_pct    = matriz_confusion_df(y_true, y_pred, etiquetas, normalizar="true")

    if verboso:
        print()
        print(f"  Precision / Recall / F1 por categoría — {nombre} [{split}]")
        print("  " + "─" * 78)
        print(por_clase.to_string(index=False, float_format="{:.4f}".format))
        print()
        print(f"  Matriz de confusión (conteos) — real (filas) × predicho (columnas)")
        print("  " + "─" * 78)
        print(cm_conteo.to_string())
        print()
        print(f"  Matriz de confusión (% por clase real)")
        print("  " + "─" * 78)
        print(cm_pct.to_string(float_format="{:.1f}".format))

    if guardar:
        carpeta = PATHS["FOLDER_RESULTS_TABLES"]
        carpeta.mkdir(parents=True, exist_ok=True)
        sufijo = "_".join(p for p in [nombre, estrategia_balanceo, variante_target, split] if p)
        por_clase.to_csv(carpeta / f"metricas_por_clase_{sufijo}.csv", index=False)
        cm_conteo.to_csv(carpeta / f"matriz_confusion_{sufijo}.csv")
        cm_pct.round(2).to_csv(carpeta / f"matriz_confusion_pct_{sufijo}.csv")
        if verboso:
            print()
            print(f"  ✓ metricas_por_clase_{sufijo}.csv")
            print(f"  ✓ matriz_confusion_{sufijo}.csv")
            print(f"  ✓ matriz_confusion_pct_{sufijo}.csv")

    return {
        "metricas"      : metricas,
        "por_clase"     : por_clase,
        "confusion"     : cm_conteo,
        "confusion_pct" : cm_pct,
    }