"""
utils/xai.py
============
Incertidumbre y robustez de los rankings de importancia.

Una importancia SHAP media es una estimación puntual obtenida de un modelo
entrenado con una semilla sobre un conjunto de prueba finito. Cuando dos
variables tienen importancias próximas —o cuando los predictores están
correlacionados, como los índices de V-Dem— el orden del ranking no queda
identificado por los datos. Este módulo cuantifica esa incertidumbre por tres
vías complementarias:

1. `bootstrap_importancia_shap()` — remuestreo de clústeres (país-año o país)
   del conjunto de prueba: intervalo de confianza del valor |SHAP|, intervalo
   del rango y probabilidad de permanecer en el top-k.
2. `concordancia_rankings()` — correlación de Spearman y W de Kendall entre
   rankings obtenidos de fuentes distintas (modelos empatados en rendimiento,
   semillas o subconjuntos).
3. `resumen_estabilidad()` — lectura conjunta de ambas salidas.

Advertencia de interpretación: con predictores correlacionados los valores de
Shapley reparten una señal compartida entre las variables implicadas, por lo
que la lectura por bloque temático es más estable que la lectura por variable
individual (Aas, Jullum y Løland, 2021; Kumar et al., 2020).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Sequence

from .config import ETIQUETAS_FEATURES, bloque_de


def importancia_shap(df_shap: pd.DataFrame) -> pd.Series:
    """Importancia global |SHAP| media por variable."""
    return df_shap.abs().mean().sort_values(ascending=False)


def bootstrap_importancia_shap(
    df_shap: pd.DataFrame,
    clusters,
    B: int = 1000,
    seed: int = 42,
    top_k: Sequence[int] = (5, 15),
    nivel: float = 0.95,
) -> pd.DataFrame:
    """
    Intervalos de confianza del valor y del rango de importancia |SHAP|.

    Parámetros
    ----------
    df_shap  : DataFrame (n_registros × n_variables) de valores SHAP, tal como
               lo devuelve `utils.io.cargar_shap_values`.
    clusters : etiqueta de clúster de cada registro, en el mismo orden que las
               filas de df_shap (p. ej. país-año del conjunto de prueba).
    B        : número de réplicas bootstrap.
    seed     : semilla del generador.
    top_k    : valores de k para los que se reporta la probabilidad de que la
               variable permanezca entre las k más importantes.
    nivel    : nivel de confianza de los intervalos percentil.

    Retorna
    -------
    DataFrame con una fila por variable: variable, etiqueta, bloque,
    importancia, ic_inf, ic_sup, ee_bootstrap, rango, rango_ic_inf,
    rango_ic_sup y una columna pct_topk por cada k solicitado.
    """
    arr = np.asarray(pd.Series(clusters).astype(str).values)
    if len(arr) != len(df_shap):
        raise ValueError(
            f"clusters tiene {len(arr)} elementos y df_shap {len(df_shap)} filas"
        )
    etiquetas_cl = np.unique(arr)
    idx_por_cluster = [np.flatnonzero(arr == c) for c in etiquetas_cl]

    variables = list(df_shap.columns)
    matriz_abs = df_shap.abs().to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    n_cl = len(idx_por_cluster)
    imp_rep  = np.empty((B, len(variables)), dtype=float)
    rank_rep = np.empty((B, len(variables)), dtype=float)

    for b in range(B):
        elegidos = rng.integers(0, n_cl, n_cl)
        idx = np.concatenate([idx_por_cluster[i] for i in elegidos])
        imp = matriz_abs[idx].mean(axis=0)
        imp_rep[b] = imp
        # Rango 1 = más importante
        rank_rep[b] = pd.Series(imp).rank(ascending=False, method="min").to_numpy()

    alfa = (1 - nivel) / 2
    imp_obs  = matriz_abs.mean(axis=0)
    rank_obs = pd.Series(imp_obs).rank(ascending=False, method="min").to_numpy()

    filas = []
    for j, var in enumerate(variables):
        fila = {
            "variable"     : var,
            "etiqueta"     : ETIQUETAS_FEATURES.get(var, var),
            "bloque"       : bloque_de(var),
            "importancia"  : round(float(imp_obs[j]), 6),
            "ic_inf"       : round(float(np.quantile(imp_rep[:, j], alfa)), 6),
            "ic_sup"       : round(float(np.quantile(imp_rep[:, j], 1 - alfa)), 6),
            "ee_bootstrap" : round(float(imp_rep[:, j].std(ddof=1)), 6),
            "rango"        : int(rank_obs[j]),
            "rango_ic_inf" : int(np.quantile(rank_rep[:, j], alfa)),
            "rango_ic_sup" : int(np.quantile(rank_rep[:, j], 1 - alfa)),
        }
        for k in top_k:
            fila[f"pct_top{k}"] = round(float((rank_rep[:, j] <= k).mean() * 100), 1)
        filas.append(fila)

    return (pd.DataFrame(filas)
            .sort_values("importancia", ascending=False)
            .reset_index(drop=True))


def concordancia_rankings(
    importancias: Dict[str, pd.Series],
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Matriz de correlación de Spearman entre rankings de importancia.

    Parámetros
    ----------
    importancias : dict {etiqueta: Series indexada por variable}. Las etiquetas
                   pueden ser modelos, semillas o subconjuntos.
    variables    : subconjunto común a comparar. Si es None se usa la
                   intersección de los índices.

    Retorna
    -------
    DataFrame cuadrado con los coeficientes ρ de Spearman.
    """
    claves = list(importancias.keys())
    if len(claves) < 2:
        raise ValueError("Se requieren al menos dos rankings para comparar")
    if variables is None:
        comunes = set(importancias[claves[0]].index)
        for k in claves[1:]:
            comunes &= set(importancias[k].index)
        variables = sorted(comunes)
    if len(variables) < 3:
        raise ValueError(f"Solo {len(variables)} variables comunes; insuficiente")

    mat = pd.DataFrame(index=claves, columns=claves, dtype=float)
    for a in claves:
        for b in claves:
            r, _ = stats.spearmanr(importancias[a][variables],
                                   importancias[b][variables])
            mat.loc[a, b] = round(float(r), 4)
    return mat


def w_kendall(importancias: Dict[str, pd.Series],
              variables: Optional[List[str]] = None) -> Dict:
    """
    W de Kendall (coeficiente de concordancia) entre varios rankings.

    W = 1 indica acuerdo perfecto entre los rankings; W = 0, ausencia de
    acuerdo. Se calcula a partir del estadístico de Friedman sobre las
    matrices de rangos, con m rankings y n variables.

    Retorna
    -------
    dict con: W, n_rankings, n_variables, chi2, p_valor.
    """
    claves = list(importancias.keys())
    if variables is None:
        comunes = set(importancias[claves[0]].index)
        for k in claves[1:]:
            comunes &= set(importancias[k].index)
        variables = sorted(comunes)

    # Matriz de rangos: filas = rankings (jueces), columnas = variables
    rangos = np.vstack([
        pd.Series(importancias[k][variables]).rank(ascending=False).to_numpy()
        for k in claves
    ])
    m, n = rangos.shape
    suma_rangos = rangos.sum(axis=0)
    s = ((suma_rangos - suma_rangos.mean()) ** 2).sum()
    w = 12 * s / (m ** 2 * (n ** 3 - n))
    chi2 = m * (n - 1) * w
    p = 1 - stats.chi2.cdf(chi2, df=n - 1)
    return {
        "W"           : round(float(w), 4),
        "n_rankings"  : int(m),
        "n_variables" : int(n),
        "chi2"        : round(float(chi2), 4),
        "p_valor"     : round(float(p), 6),
    }


def resumen_estabilidad(
    df_boot: pd.DataFrame,
    mat_concordancia: Optional[pd.DataFrame] = None,
    top_n: int = 15,
    umbral_top: int = 5,
) -> None:
    """
    Imprime una lectura conjunta de la incertidumbre del ranking.

    Parámetros
    ----------
    df_boot          : salida de `bootstrap_importancia_shap`.
    mat_concordancia : salida de `concordancia_rankings`, opcional.
    top_n            : número de variables a describir.
    umbral_top       : k para la lectura de permanencia en el top-k.
    """
    col_top = f"pct_top{umbral_top}"
    top = df_boot.head(top_n)

    print(f"Amplitud del intervalo de rango en las {top_n} variables principales:")
    amplitud = (top["rango_ic_sup"] - top["rango_ic_inf"])
    print(f"  Amplitud media   : {amplitud.mean():.1f} posiciones")
    print(f"  Amplitud máxima  : {amplitud.max()} posiciones "
          f"({top.loc[amplitud.idxmax(), 'etiqueta']})")
    print()

    if col_top in df_boot.columns:
        estables = df_boot[df_boot[col_top] >= 95]
        frontera = df_boot[(df_boot[col_top] > 5) & (df_boot[col_top] < 95)]
        print(f"Permanencia en el top-{umbral_top} entre réplicas bootstrap:")
        print(f"  Variables estables (≥95%)   : {len(estables)} "
              f"{list(estables['etiqueta'])}")
        print(f"  Variables en la frontera    : {len(frontera)} "
              f"{list(frontera['etiqueta'])}")
        print()

    solapan = []
    for i in range(min(top_n, len(df_boot)) - 1):
        a, b = df_boot.iloc[i], df_boot.iloc[i + 1]
        if a["ic_inf"] <= b["ic_sup"]:
            solapan.append(f"{a['etiqueta']} vs {b['etiqueta']}")
    print(f"Pares consecutivos con intervalos de importancia solapados: {len(solapan)}")
    for par in solapan:
        print(f"  {par}")
    if solapan:
        print("  → el orden relativo de esos pares no queda identificado por los datos.")
    print()

    if mat_concordancia is not None:
        vals = [mat_concordancia.iloc[i, j]
                for i in range(len(mat_concordancia))
                for j in range(i)]
        if vals:
            print("Concordancia entre rankings (ρ de Spearman por pares):")
            print(f"  mínima {min(vals):.4f} | media {np.mean(vals):.4f} | "
                  f"máxima {max(vals):.4f}")
