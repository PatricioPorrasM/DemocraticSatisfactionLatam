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
- `bootstrap_metricas()`     : intervalos de confianza por remuestreo de clústeres.
- `bootstrap_pareado()`      : diferencia entre dos modelos con IC y P(Δ>0).
- `comparar_modelos_bootstrap()` : comparación pareada de todos los modelos
                               contra una configuración de referencia.

Ponderación: todas las funciones aceptan `sample_weight`. El proyecto reporta
dos lecturas complementarias para el modelo principal: sin ponderar (predicción
por registro individual) y ponderada por el factor de expansión muestral
`X_020` del Latinobarómetro (lectura representativa de la muestra nacional).

Incertidumbre: las diferencias entre modelos se evalúan por remuestreo de
clústeres (país-año o país) y no por remuestreo de registros individuales,
porque los registros de un mismo país-año comparten los indicadores de V-Dem
y no son independientes (ver Cameron y Miller, 2015).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    cohen_kappa_score, mean_absolute_error, roc_auc_score,
    confusion_matrix, precision_recall_fscore_support,
)
from typing import Dict, List, Optional, Sequence, Tuple

from .config import ETIQUETAS, PATHS, PARAMETERS


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
    sample_weight=None,
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
    sample_weight       : pesos por observación. None = métricas sin ponderar
                          (predicción por registro individual); con el factor
                          de expansión `X_020` normalizado se obtiene la
                          lectura representativa de la muestra.
    verboso             : si False, no imprime el resumen por consola
    """
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    metricas = {
        "modelo"              : nombre,
        "estrategia_balanceo" : estrategia_balanceo,
        "variante_target"     : variante_target,
        "split"               : split,
        # Provenance: distingue las métricas de la corrida definitiva de las de
        # una prueba de humo, que comparten el mismo formato de archivo.
        "modo_ejecucion"      : PARAMETERS["MODO_EJECUCION"],
        "ponderado"           : sw is not None,
        "accuracy"            : accuracy_score(y_true, y_pred, sample_weight=sw),
        "balanced_accuracy"   : balanced_accuracy_score(y_true, y_pred, sample_weight=sw),
        "f1_macro"            : f1_score(y_true, y_pred, average="macro",    zero_division=0, sample_weight=sw),
        "f1_weighted"         : f1_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw),
        "kappa_lineal"        : cohen_kappa_score(y_true, y_pred, weights="linear",    sample_weight=sw),
        "kappa_cuadratico"    : cohen_kappa_score(y_true, y_pred, weights="quadratic", sample_weight=sw),
        "mae_ordinal"         : mean_absolute_error(y_true, y_pred, sample_weight=sw),
        "auroc_macro"         : np.nan,
    }
    if y_prob is not None:
        try:
            y_prob_arr = np.asarray(y_prob)
            if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2:
                # Caso binario: roc_auc_score espera la probabilidad de la clase positiva
                metricas["auroc_macro"] = roc_auc_score(
                    y_true, y_prob_arr[:, 1], sample_weight=sw)
            else:
                metricas["auroc_macro"] = roc_auc_score(
                    y_true, y_prob_arr, multi_class="ovr", average="macro",
                    sample_weight=sw,
                )
        except Exception:
            pass

    if nombre and verboso:
        print(f"  {'─'*52}")
        etiqueta_pond = "ponderado por X_020" if sw is not None else "sin ponderar"
        print(f"  {nombre} | {estrategia_balanceo} | {variante_target} "
              f"[{split}] ({etiqueta_pond})")
        print(f"  {'─'*52}")
        omitir = ("modelo", "estrategia_balanceo", "variante_target", "split",
                  "ponderado")
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
    sample_weight=None,
) -> pd.DataFrame:
    """
    Precision, recall, F1 y soporte para cada categoría del target.

    Parámetros
    ----------
    y_true            : etiquetas reales
    y_pred            : predicciones
    etiquetas         : dict {clase: etiqueta}. Por defecto ETIQUETAS del proyecto.
    incluir_promedios : añade las filas de promedio macro y ponderado.
    sample_weight     : pesos por observación. Si se entregan, el soporte pasa
                        a ser la suma de pesos de cada clase.

    Retorna
    -------
    DataFrame con columnas: clase, etiqueta, precision, recall, f1, soporte,
    pct_soporte. Las filas de promedio llevan clase = NaN.
    """
    sw      = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    clases  = _clases_presentes(y_true, y_pred)
    nombres = _nombres_clases(clases, etiquetas)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=clases, zero_division=0, sample_weight=sw)
    n_total = float(np.sum(s))

    filas = [
        {"clase": c, "etiqueta": n, "precision": round(float(pi), 4),
         "recall": round(float(ri), 4), "f1": round(float(fi), 4),
         "soporte": round(float(si), 2) if sw is not None else int(si),
         "pct_soporte": round(float(si) / n_total * 100, 2) if n_total else np.nan}
        for c, n, pi, ri, fi, si in zip(clases, nombres, p, r, f, s)
    ]

    if incluir_promedios:
        for nombre_prom, promedio in [("Promedio macro", "macro"),
                                      ("Promedio ponderado", "weighted")]:
            pp, rp, fp, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=clases, average=promedio,
                zero_division=0, sample_weight=sw)
            filas.append({
                "clase": np.nan, "etiqueta": nombre_prom,
                "precision": round(float(pp), 4), "recall": round(float(rp), 4),
                "f1": round(float(fp), 4),
                "soporte": round(n_total, 2) if sw is not None else int(n_total),
                "pct_soporte": 100.0,
            })

    df_out = pd.DataFrame(filas)
    df_out["clase"] = df_out["clase"].astype("Int64")   # entero con soporte de NA
    return df_out


def matriz_confusion_df(
    y_true,
    y_pred,
    etiquetas: Optional[Dict] = None,
    normalizar: Optional[str] = None,
    sample_weight=None,
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
    sw      = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    clases  = _clases_presentes(y_true, y_pred)
    nombres = _nombres_clases(clases, etiquetas)
    cm = confusion_matrix(y_true, y_pred, labels=clases,
                          normalize=normalizar, sample_weight=sw)
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
    sample_weight=None,
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
                        variante_target, split, sample_weight=sample_weight,
                        verboso=verboso)
    por_clase = metricas_por_clase(y_true, y_pred, etiquetas,
                                   sample_weight=sample_weight)
    cm_conteo = matriz_confusion_df(y_true, y_pred, etiquetas,
                                    sample_weight=sample_weight)
    cm_pct    = matriz_confusion_df(y_true, y_pred, etiquetas, normalizar="true",
                                    sample_weight=sample_weight)

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
        if sample_weight is not None:
            sufijo += "_ponderado"
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


# ─────────────────────────────────────────────────────────────────────────────
# INCERTIDUMBRE: BOOTSTRAP POR CLÚSTERES
#
# Los registros de un mismo país-año comparten los indicadores de V-Dem y
# proceden de la misma muestra nacional, por lo que no son independientes.
# Remuestrear registros individuales subestimaría el error estándar; se
# remuestrean clústeres completos con reemplazo (Cameron y Miller, 2015).
# ─────────────────────────────────────────────────────────────────────────────

def _kappa_cuad(y_t, y_p, sw=None):
    return cohen_kappa_score(y_t, y_p, weights="quadratic", sample_weight=sw)


def _kappa_lin(y_t, y_p, sw=None):
    return cohen_kappa_score(y_t, y_p, weights="linear", sample_weight=sw)


def _f1_macro(y_t, y_p, sw=None):
    return f1_score(y_t, y_p, average="macro", zero_division=0, sample_weight=sw)


METRICAS_BOOTSTRAP = {
    "kappa_cuadratico" : _kappa_cuad,
    "kappa_lineal"     : _kappa_lin,
    "mae_ordinal"      : lambda y_t, y_p, sw=None: mean_absolute_error(y_t, y_p, sample_weight=sw),
    "accuracy"         : lambda y_t, y_p, sw=None: accuracy_score(y_t, y_p, sample_weight=sw),
    "f1_macro"         : _f1_macro,
    "balanced_accuracy": lambda y_t, y_p, sw=None: balanced_accuracy_score(y_t, y_p, sample_weight=sw),
}

METRICAS_BOOTSTRAP_DEFECTO = ["kappa_cuadratico", "mae_ordinal", "accuracy", "f1_macro"]


def _preparar_clusteres(clusters) -> Tuple[List[np.ndarray], np.ndarray]:
    """Devuelve (lista de índices posicionales por clúster, etiquetas únicas)."""
    arr = np.asarray(pd.Series(clusters).astype(str).values)
    etiquetas = np.unique(arr)
    return [np.flatnonzero(arr == c) for c in etiquetas], etiquetas


def _replicas_bootstrap(idx_por_cluster: List[np.ndarray], B: int, seed: int):
    """Generador de índices posicionales de cada réplica bootstrap."""
    rng = np.random.default_rng(seed)
    n_cl = len(idx_por_cluster)
    for _ in range(B):
        elegidos = rng.integers(0, n_cl, n_cl)
        yield np.concatenate([idx_por_cluster[i] for i in elegidos])


def bootstrap_metricas(
    y_true,
    y_pred,
    clusters,
    metricas: Optional[Sequence[str]] = None,
    B: int = 1000,
    seed: int = 42,
    sample_weight=None,
    nivel: float = 0.95,
) -> pd.DataFrame:
    """
    Intervalos de confianza percentil por remuestreo de clústeres.

    Parámetros
    ----------
    y_true, y_pred : etiquetas reales y predichas.
    clusters       : etiqueta de clúster de cada observación (p. ej. país-año).
    metricas       : nombres presentes en METRICAS_BOOTSTRAP. Por defecto
                     kappa cuadrático, MAE ordinal, accuracy y F1 macro.
    B              : número de réplicas bootstrap.
    seed           : semilla del generador; fija las réplicas y las hace
                     comparables entre llamadas.
    sample_weight  : pesos por observación, opcional.
    nivel          : nivel de confianza del intervalo percentil.

    Retorna
    -------
    DataFrame con: metrica, valor, ic_inf, ic_sup, ee_bootstrap, B, n_clusteres.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    sw  = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    metricas = list(metricas or METRICAS_BOOTSTRAP_DEFECTO)
    desconocidas = [m for m in metricas if m not in METRICAS_BOOTSTRAP]
    if desconocidas:
        raise ValueError(f"Métricas no soportadas en bootstrap: {desconocidas}")

    idx_por_cluster, etiquetas = _preparar_clusteres(clusters)
    replicas = {m: [] for m in metricas}

    for idx in _replicas_bootstrap(idx_por_cluster, B, seed):
        sw_b = None if sw is None else sw[idx]
        for m in metricas:
            try:
                replicas[m].append(METRICAS_BOOTSTRAP[m](y_t[idx], y_p[idx], sw_b))
            except Exception:                                    # noqa: BLE001
                replicas[m].append(np.nan)

    alfa = (1 - nivel) / 2
    filas = []
    for m in metricas:
        vals = np.asarray(replicas[m], dtype=float)
        vals = vals[~np.isnan(vals)]
        filas.append({
            "metrica"     : m,
            "valor"       : round(float(METRICAS_BOOTSTRAP[m](y_t, y_p, sw)), 4),
            "ic_inf"      : round(float(np.quantile(vals, alfa)), 4),
            "ic_sup"      : round(float(np.quantile(vals, 1 - alfa)), 4),
            "ee_bootstrap": round(float(vals.std(ddof=1)), 4),
            "B"           : int(len(vals)),
            "n_clusteres" : int(len(etiquetas)),
        })
    return pd.DataFrame(filas)


def bootstrap_pareado(
    y_true,
    y_pred_a,
    y_pred_b,
    clusters,
    metrica: str = "kappa_cuadratico",
    B: int = 1000,
    seed: int = 42,
    sample_weight=None,
    nivel: float = 0.95,
    mayor_es_mejor: bool = True,
) -> Dict:
    """
    Compara dos modelos sobre las MISMAS réplicas bootstrap.

    Evaluar ambos modelos en cada réplica elimina la variabilidad común del
    remuestreo, de modo que el intervalo describe directamente la diferencia,
    que es la cantidad de interés cuando los dos modelos se evalúan sobre el
    mismo conjunto de prueba.

    Retorna
    -------
    dict con: metrica, valor_a, valor_b, delta, ic_inf, ic_sup,
    p_delta_favor_a, B, n_clusteres. `delta` se define de modo que un valor
    positivo siempre favorece al modelo A (se invierte el signo cuando la
    métrica es de tipo error, con mayor_es_mejor=False).
    """
    if metrica not in METRICAS_BOOTSTRAP:
        raise ValueError(f"Métrica no soportada: {metrica}")
    fn  = METRICAS_BOOTSTRAP[metrica]
    y_t = np.asarray(y_true)
    y_a = np.asarray(y_pred_a)
    y_b = np.asarray(y_pred_b)
    sw  = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    signo = 1.0 if mayor_es_mejor else -1.0

    idx_por_cluster, etiquetas = _preparar_clusteres(clusters)
    deltas = []
    for idx in _replicas_bootstrap(idx_por_cluster, B, seed):
        sw_b = None if sw is None else sw[idx]
        try:
            deltas.append(signo * (fn(y_t[idx], y_a[idx], sw_b) -
                                   fn(y_t[idx], y_b[idx], sw_b)))
        except Exception:                                        # noqa: BLE001
            deltas.append(np.nan)

    d = np.asarray(deltas, dtype=float)
    d = d[~np.isnan(d)]
    alfa = (1 - nivel) / 2
    va, vb = fn(y_t, y_a, sw), fn(y_t, y_b, sw)
    return {
        "metrica"        : metrica,
        "valor_a"        : round(float(va), 4),
        "valor_b"        : round(float(vb), 4),
        "delta"          : round(float(signo * (va - vb)), 4),
        "ic_inf"         : round(float(np.quantile(d, alfa)), 4),
        "ic_sup"         : round(float(np.quantile(d, 1 - alfa)), 4),
        "p_delta_favor_a": round(float((d > 0).mean()), 4),
        "B"              : int(len(d)),
        "n_clusteres"    : int(len(etiquetas)),
    }


def comparar_modelos_bootstrap(
    y_true,
    predicciones: Dict[str, np.ndarray],
    referencia: str,
    clusters,
    metrica: str = "kappa_cuadratico",
    B: int = 1000,
    seed: int = 42,
    sample_weight=None,
    mayor_es_mejor: bool = True,
) -> pd.DataFrame:
    """
    Compara todas las configuraciones contra una de referencia.

    Parámetros
    ----------
    predicciones : dict {etiqueta: y_pred}.
    referencia   : clave de `predicciones` que actúa como modelo A.

    Retorna
    -------
    DataFrame ordenado por delta descendente, con una fila por comparación y
    una columna `conclusion` que indica si el intervalo excluye el cero.
    """
    if referencia not in predicciones:
        raise KeyError(f"'{referencia}' no está en las predicciones entregadas")
    filas = []
    for etiqueta, y_p in predicciones.items():
        if etiqueta == referencia:
            continue
        res = bootstrap_pareado(
            y_true, predicciones[referencia], y_p, clusters, metrica,
            B=B, seed=seed, sample_weight=sample_weight,
            mayor_es_mejor=mayor_es_mejor)
        excluye_cero = (res["ic_inf"] > 0) or (res["ic_sup"] < 0)
        filas.append({
            "referencia"       : referencia,
            "comparado"        : etiqueta,
            "valor_ref"        : res["valor_a"],
            "valor_comp"       : res["valor_b"],
            "delta"            : res["delta"],
            "ic_inf"           : res["ic_inf"],
            "ic_sup"           : res["ic_sup"],
            "p_delta_favor_ref": res["p_delta_favor_a"],
            "conclusion"       : ("diferencia distinguible" if excluye_cero
                                  else "no distinguible (IC incluye 0)"),
        })
    return (pd.DataFrame(filas)
            .sort_values("delta", ascending=False)
            .reset_index(drop=True))
