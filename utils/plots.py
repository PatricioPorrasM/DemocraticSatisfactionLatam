"""
utils/plots.py
==============
Funciones de visualización reutilizables para el proyecto de tesis.

Organización:
- Sección 1: helpers base (colores, guardado)
- Sección 2: notebook 03 — evaluación comparativa
- Sección 3: notebook 04 — explicabilidad XAI
- Sección 4: notebook 05 — estabilidad temporal y regional
- Sección 5: notebook 06 — contraste teórico

Convención: todas las funciones que generan figuras reciben opcionalmente
`nombre_archivo` para guardar automáticamente. Si es None, no guardan.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from .config import (
    THEME, PATHS, ETIQUETAS, ETIQUETAS_FEATURES, BLOQUES,
    bloque_de, SUBPERIODOS,
)




# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — HELPERS BASE
# ═════════════════════════════════════════════════════════════════════════════

def model_color(model: str) -> str:
    """Retorna el color hex asociado al modelo según THEME."""
    return THEME["models"].get(model, "#888888")


def save_figure(nombre: str, carpeta: Optional[Path] = None,
                dpi: int = 150, bbox_inches: str = "tight") -> None:
    """
    Guarda la figura activa de matplotlib.

    Parámetros
    ----------
    nombre    : nombre del archivo sin extensión.
    carpeta   : directorio destino. Por defecto FOLDER_RESULTS_FIGURES.
    dpi       : resolución de salida.
    """
    if carpeta is None:
        carpeta = PATHS["FOLDER_RESULTS_FIGURES"]
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{nombre}.png"
    plt.tight_layout()
    plt.savefig(ruta, dpi=dpi, bbox_inches=bbox_inches)
    print(f"  ✓ Figura guardada: {ruta.name}")


def _barra_bloques(ax_bar: plt.Axes, variables: List[str],
                   bloques_lb: Optional[List[str]] = None) -> None:
    """
    Dibuja una barra lateral de colores de bloque temático en un eje.

    Parámetros
    ----------
    ax_bar    : eje de matplotlib donde se dibuja la barra.
    variables : lista ordenada de variables (eje Y del heatmap adyacente).
    bloques_lb: lista de nombres de bloque correspondientes a cada variable.
                Si None, se calcula automáticamente con bloque_de().
    """
    if bloques_lb is None:
        bloques_lb = [bloque_de(v) for v in variables]

    colores_bloques = THEME.get("blocks", {})
    for i, bloque in enumerate(bloques_lb):
        color = colores_bloques.get(bloque, "#AAAAAA")
        ax_bar.barh(i, 1, color=color, edgecolor="none")

    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.5, len(variables) - 0.5)
    ax_bar.axis("off")

    # Leyenda de bloques
    vistos = {}
    for b in bloques_lb:
        if b not in vistos:
            vistos[b] = colores_bloques.get(b, "#AAAAAA")
    patches = [mpatches.Patch(color=c, label=b) for b, c in vistos.items()]
    ax_bar.legend(handles=patches, loc="upper left",
                  bbox_to_anchor=(0, -0.02), fontsize=8, frameon=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — EVALUACIÓN COMPARATIVA (notebook 03)
# ═════════════════════════════════════════════════════════════════════════════

def plot_metricas_comparativas(
    df_res: pd.DataFrame,
    metricas: Optional[List[Tuple[str, str]]] = None,
    nombre_archivo: Optional[str] = "03_metricas_comparativas",
) -> None:
    """
    Gráfico de líneas con rendimiento de cada modelo por subperiodo.

    Parámetros
    ----------
    df_res         : DataFrame de resultados (split='test').
    metricas       : lista de (columna, título). Por defecto las 3 principales.
    nombre_archivo : nombre para guardar la figura. None = no guardar.
    """
    if metricas is None:
        metricas = [
            ("kappa_cuadratico", "Kappa Cuadrático (↑ mejor)"),
            ("f1_macro",         "F1 Macro (↑ mejor)"),
            ("mae_ordinal",      "MAE Ordinal (↓ mejor)"),
        ]

    modelos = ["OLO", "XGBoost", "CatBoost", "LightGBM", "TabNet"]
    fig, axes = plt.subplots(1, len(metricas), figsize=(16, 5))
    fig.suptitle(
        "Rendimiento comparativo de modelos\n"
        "(Expanding Window Walk-Forward Validation — conjunto de prueba)",
        fontsize=13, fontweight="bold",
    )

    for ax, (metrica, titulo) in zip(axes, metricas):
        for modelo in modelos:
            sub = df_res[df_res["modelo"] == modelo]
            if sub.empty or metrica not in sub.columns:
                continue
            ax.plot(sub["subperiodo"], sub[metrica],
                    marker="o", linewidth=2, markersize=7,
                    label=modelo, color=model_color(modelo))
        ax.set_title(titulo, fontweight="bold")
        ax.set_xlabel("Subperiodo")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if metrica == "mae_ordinal":
            ax.invert_yaxis()

    if nombre_archivo:
        save_figure(nombre_archivo)
    plt.show()


def plot_matrices_confusion(
    resultados_cm: Dict,
    solo_mejor: bool = False,
    mejor_modelo: Optional[str] = None,
    nombre_archivo: Optional[str] = "03_matrices_confusion",
) -> None:
    """
    Matriz(ces) de confusión normalizada por fila.

    Parámetros
    ----------
    resultados_cm : dict con claves (modelo, subperiodo) y valores
                    (y_true, y_pred).
    solo_mejor    : si True, solo dibuja el mejor modelo (requiere
                    mejor_modelo).
    mejor_modelo  : nombre del modelo a mostrar cuando solo_mejor=True.
    nombre_archivo: nombre para guardar. None = no guardar.
    """
    from sklearn.metrics import confusion_matrix

    etiq_cortas = ["Nada\nsat.", "No muy\nsat.", "Más bien\nsat.", "Muy\nsat."]
    sps = ["SP1", "SP2", "SP3"]

    if solo_mejor and mejor_modelo:
        modelos_plot = [mejor_modelo]
    else:
        modelos_plot = ["OLO", "XGBoost", "CatBoost", "LightGBM", "TabNet"]

    n_mod = len(modelos_plot)
    fig, axes = plt.subplots(n_mod, 3, figsize=(12, 4 * n_mod))
    if n_mod == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        "Matrices de confusión normalizadas (% por fila)\n"
        "Conjunto de prueba por subperiodo",
        fontweight="bold", fontsize=13,
    )

    for row, modelo in enumerate(modelos_plot):
        for col, sp in enumerate(sps):
            ax = axes[row, col]
            clave = (modelo, sp)
            if clave not in resultados_cm:
                ax.set_visible(False)
                continue
            y_true, y_pred = resultados_cm[clave]
            cm = confusion_matrix(y_true, y_pred, normalize="true")
            sns.heatmap(
                cm * 100, annot=True, fmt=".1f", cmap="Blues",
                ax=ax, cbar=False, linewidths=0.3,
                xticklabels=etiq_cortas, yticklabels=etiq_cortas,
                annot_kws={"size": 8},
            )
            ax.set_title(f"{modelo} — {sp}", fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel("Real")
            if row == n_mod - 1:
                ax.set_xlabel("Predicho")

    if nombre_archivo:
        save_figure(nombre_archivo)
    plt.show()


def plot_rendimiento_por_pais(
    df_mae: pd.DataFrame,
    subperiodo: str = "SP3",
    nombre_archivo: Optional[str] = None,
) -> None:
    """
    Gráfico de barras horizontales del MAE ordinal por país.

    Parámetros
    ----------
    df_mae        : DataFrame con columnas 'pais', 'modelo', 'mae_ordinal'.
    subperiodo    : título del subperiodo para el encabezado.
    nombre_archivo: nombre para guardar. None = no guardar.
    """
    modelos = df_mae["modelo"].unique().tolist()
    paises  = sorted(df_mae["pais"].unique(), key=lambda p:
                     df_mae[df_mae["pais"] == p]["mae_ordinal"].mean())

    fig, ax = plt.subplots(figsize=(10, max(6, len(paises) * 0.45)))
    y_pos = np.arange(len(paises))
    ancho = 0.8 / len(modelos)

    for i, modelo in enumerate(modelos):
        vals = [df_mae[(df_mae["pais"] == p) & (df_mae["modelo"] == modelo)]
                ["mae_ordinal"].values[0]
                if len(df_mae[(df_mae["pais"] == p) & (df_mae["modelo"] == modelo)]) > 0
                else np.nan
                for p in paises]
        ax.barh(y_pos + i * ancho, vals, ancho * 0.9,
                label=modelo, color=model_color(modelo), alpha=0.85)

    ax.set_yticks(y_pos + ancho * (len(modelos) - 1) / 2)
    ax.set_yticklabels(paises, fontsize=9)
    ax.set_xlabel("MAE Ordinal (↓ mejor)")
    ax.set_title(f"MAE Ordinal por país — {subperiodo}", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Umbral 0.5")
    ax.grid(True, axis="x", alpha=0.3)

    if nombre_archivo:
        save_figure(nombre_archivo)
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — EXPLICABILIDAD XAI (notebook 04)
# ═════════════════════════════════════════════════════════════════════════════

def plot_shap_bar_bloques(
    importancias: pd.Series,
    top_n: int = 20,
    titulo: str = "Importancia global (SHAP)",
    nombre_archivo: Optional[str] = None,
) -> None:
    """
    Gráfico de barras horizontales de importancia media |SHAP|,
    con colores por bloque temático.

    Parámetros
    ----------
    importancias  : Series con índice = variable, valor = |SHAP| medio.
    top_n         : número de variables a mostrar.
    titulo        : título del gráfico.
    nombre_archivo: nombre para guardar. None = no guardar.
    """
    top = importancias.sort_values(ascending=False).head(top_n)
    etiquetas  = [ETIQUETAS_FEATURES.get(v, v) for v in top.index]
    bloq_colors = [THEME.get("blocks", {}).get(bloque_de(v), "#AAAAAA")
                   for v in top.index]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.38)))
    ax.barh(etiquetas[::-1], top.values[::-1],
            color=bloq_colors[::-1], edgecolor="white", linewidth=0.4)
    ax.set_xlabel("|SHAP| medio")
    ax.set_title(titulo, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Leyenda de bloques
    vistos = {}
    for v in top.index:
        b = bloque_de(v)
        if b not in vistos:
            vistos[b] = THEME.get("blocks", {}).get(b, "#AAAAAA")
    patches = [mpatches.Patch(color=c, label=b) for b, c in vistos.items()]
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    if nombre_archivo:
        save_figure(nombre_archivo)
    plt.show()


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    top_n: int = 20,
    titulo: str = "SHAP Beeswarm",
    nombre_archivo: Optional[str] = None,
) -> None:
    """
    Beeswarm plot de valores SHAP usando la API nativa de shap.

    Si shap está disponible usa shap.plots.beeswarm; de lo contrario
    genera un violin plot como fallback.

    Parámetros
    ----------
    shap_values   : array (n_muestras × n_features) de valores SHAP.
    X             : DataFrame de features en el mismo orden.
    top_n         : número de features a mostrar.
    titulo        : título del gráfico.
    nombre_archivo: nombre para guardar. None = no guardar.
    """
    try:
        import shap
        # Renombrar columnas con etiquetas legibles
        X_plot = X.rename(columns=ETIQUETAS_FEATURES)
        cols_orig = list(X.columns)
        feat_order = (
            pd.Series(np.abs(shap_values).mean(axis=0), index=cols_orig)
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        idx_feat = [cols_orig.index(c) for c in feat_order]
        etiq_feat = [ETIQUETAS_FEATURES.get(c, c) for c in feat_order]

        exp = shap.Explanation(
            values=shap_values[:, idx_feat],
            data=X.iloc[:, idx_feat].values,
            feature_names=etiq_feat,
        )
        plt.figure(figsize=(10, max(5, top_n * 0.38)))
        shap.plots.beeswarm(exp, max_display=top_n, show=False)
        plt.title(titulo, fontweight="bold", pad=12)
        if nombre_archivo:
            save_figure(nombre_archivo)
        plt.show()

    except ImportError:
        # Fallback: violin plot por feature
        imp_orden = (
            pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
            .sort_values(ascending=False)
            .head(top_n)
        )
        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.38)))
        for i, feat in enumerate(imp_orden.index[::-1]):
            j = list(X.columns).index(feat)
            ax.scatter(shap_values[:, j],
                       np.full(len(shap_values), i) + np.random.normal(0, 0.05, len(shap_values)),
                       alpha=0.2, s=5, color=THEME.get("blocks", {}).get(bloque_de(feat), "#2E74B5"))
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([ETIQUETAS_FEATURES.get(f, f) for f in imp_orden.index[::-1]], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Valor SHAP")
        ax.set_title(titulo, fontweight="bold")
        if nombre_archivo:
            save_figure(nombre_archivo)
        plt.show()


def plot_ale(
    ale_values: np.ndarray,
    ale_quantiles: np.ndarray,
    feature: str,
    titulo: Optional[str] = None,
    nombre_archivo: Optional[str] = None,
) -> None:
    """
    Gráfico de Accumulated Local Effects (ALE) para una feature.

    Parámetros
    ----------
    ale_values    : array 1D con los efectos ALE.
    ale_quantiles : array 1D con los cuantiles (eje X).
    feature       : nombre técnico de la feature.
    titulo        : título del gráfico. Si None se usa la etiqueta de la feature.
    nombre_archivo: nombre para guardar. None = no guardar.
    """
    etiq  = ETIQUETAS_FEATURES.get(feature, feature)
    bloque = bloque_de(feature)
    color  = THEME.get("blocks", {}).get(bloque, "#2E74B5")
    titulo = titulo or f"ALE — {etiq}"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ale_quantiles, ale_values, color=color, linewidth=2.5)
    ax.fill_between(ale_quantiles, ale_values, alpha=0.15, color=color)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(etiq)
    ax.set_ylabel("Efecto ALE sobre satisfacción democrática")
    ax.set_title(titulo, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Anotar el umbral de mayor pendiente
    if len(ale_values) > 2:
        pendiente = np.abs(np.diff(ale_values))
        idx_max   = np.argmax(pendiente)
        ax.axvline(ale_quantiles[idx_max], color="red", linewidth=1,
                   linestyle=":", alpha=0.7, label=f"Mayor cambio: {ale_quantiles[idx_max]:.2f}")
        ax.legend(fontsize=9)

    if nombre_archivo:
        save_figure(nombre_archivo)
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — ESTABILIDAD TEMPORAL Y REGIONAL (notebook 05)
# ═════════════════════════════════════════════════════════════════════════════

def plot_heatmap_estabilidad(
    df_rankings: pd.DataFrame,
    metrica: str = "mean_abs_shap",
    titulo: str = "Cambio en importancia SHAP por subperiodo",
    nombre_archivo: Optional[str] = None,
) -> None:
    """
    Heatmap de importancia SHAP por variable y subperiodo,
    con barra lateral de bloques temáticos.

    Parámetros
    ----------
    df_rankings   : DataFrame con índice=variable, columnas=subperiodos,
                    valores=importancia (|SHAP| medio normalizado).
    metrica       : nombre de la métrica para el colorbar.
    titulo        : título del gráfico.
    nombre_archivo: nombre para guardar. None = no guardar.
    """
    # Ordenar variables por bloque
    vars_ord   = []
    bloqs_ord  = []
    for bloque in BLOQUES:
        for v in BLOQUES[bloque]:
            if v in df_rankings.index:
                vars_ord.append(v)
                bloqs_ord.append(bloque)

    # Variables no clasificadas al final
    for v in df_rankings.index:
        if v not in vars_ord:
            vars_ord.append(v)
            bloqs_ord.append("Sin clasificar")

    df_plot = df_rankings.loc[vars_ord]
    etiq_y  = [ETIQUETAS_FEATURES.get(v, v) for v in vars_ord]
    df_plot.index = etiq_y

    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=(12, max(8, len(vars_ord) * 0.38)),
        gridspec_kw={"width_ratios": [0.04, 0.96]},
    )
    _barra_bloques(ax_bar, vars_ord, bloqs_ord)

    sns.heatmap(
        df_plot, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.3, ax=ax_heat,
        annot_kws={"size": 8},
        cbar_kws={"label": metrica, "shrink": 0.6},
    )
    ax_heat.set_title(titulo, fontweight="bold", pad=12)
    ax_heat.set_xlabel("Subperiodo")
    ax_heat.set_ylabel("")
    ax_heat.tick_params(axis="y", labelsize=9)

    # Líneas separadoras entre bloques
    n_acum = 0
    bloques_vistos = []
    for v, b in zip(vars_ord, bloqs_ord):
        n_acum += 1
        if b not in bloques_vistos:
            bloques_vistos.append(b)
        else:
            if bloqs_ord[vars_ord.index(v) - 1] != b:
                ax_heat.axhline(n_acum - 1, color="white", linewidth=2)

    if nombre_archivo:
        save_figure(nombre_archivo)
    plt.show()


def plot_bump_chart(
    df_rankings: pd.DataFrame,
    top_n: int = 15,
    titulo: str = "Cambio en ranking de importancia SHAP",
    nombre_archivo: Optional[str] = None,
) -> None:
    """
    Bump chart del ranking de importancia SHAP entre subperiodos.

    Muestra las top_n variables del subperiodo más reciente (SP3)
    y traza su trayectoria de ranking a lo largo de los subperiodos.

    Parámetros
    ----------
    df_rankings   : DataFrame con índice=variable, columnas=subperiodos,
                    valores=importancia. Se calcula el ranking internamente.
    top_n         : número de variables a incluir (seleccionadas por SP3).
    titulo        : título del gráfico.
    nombre_archivo: nombre para guardar. None = no guardar.
    """
    # Calcular rankings (1 = más importante)
    df_rank = df_rankings.rank(ascending=False, method="min")
    sp_cols = [c for c in ["SP1", "SP2", "SP3"] if c in df_rank.columns]

    # Seleccionar top_n por el último subperiodo disponible
    ultimo_sp = sp_cols[-1]
    top_vars  = df_rank[ultimo_sp].sort_values().head(top_n).index.tolist()
    df_top    = df_rank.loc[top_vars, sp_cols]

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.5)))

    for var in top_vars:
        bloque = bloque_de(var)
        color  = THEME.get("blocks", {}).get(bloque, "#888888")
        y_vals = df_top.loc[var].values
        ax.plot(sp_cols, y_vals, marker="o", linewidth=2, markersize=8,
                color=color, alpha=0.85)
        # Etiqueta al final
        ax.text(sp_cols[-1], y_vals[-1],
                f"  {ETIQUETAS_FEATURES.get(var, var)}",
                va="center", fontsize=8, color=color)

    ax.invert_yaxis()
    ax.set_ylabel("Ranking de importancia (1 = más importante)")
    ax.set_title(titulo, fontweight="bold")
    ax.set_xticks(sp_cols)
    ax.set_xticklabels([SUBPERIODOS[sp]["descripcion"].split("(")[0].strip()
                        for sp in sp_cols], fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if nombre_archivo:
        save_figure(nombre_archivo)
    plt.show()


def plot_spearman_estabilidad(
    df_correlaciones: pd.DataFrame,
    titulo: str = "Correlación Spearman entre rankings SHAP",
    nombre_archivo: Optional[str] = None,
) -> None:
    """
    Heatmap de correlación de Spearman entre rankings de importancia SHAP
    de distintos subperiodos (triangular superior).

    Parámetros
    ----------
    df_correlaciones : DataFrame cuadrado (subperiodos × subperiodos)
                       con los coeficientes Spearman.
    titulo           : título del gráfico.
    nombre_archivo   : nombre para guardar. None = no guardar.
    """
    mask = np.triu(np.ones_like(df_correlaciones, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        df_correlaciones, annot=True, fmt=".3f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1,
        mask=~mask,   # mostrar solo triángulo inferior + diagonal
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "r Spearman", "shrink": 0.7},
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_title(titulo, fontweight="bold", pad=12)

    if nombre_archivo:
        save_figure(nombre_archivo)
    plt.show()


def plot_shap_por_subregion(
    df_shap_region: pd.DataFrame,
    top_n: int = 10,
    titulo: str = "Importancia SHAP media por subregión",
    nombre_archivo: Optional[str] = None,
) -> None:
    """
    Gráfico de barras agrupadas mostrando la importancia SHAP de
    las top_n features para cada subregión.

    Parámetros
    ----------
    df_shap_region : DataFrame con índice=variable y columnas=subregiones,
                     valores=|SHAP| medio por subregión.
    top_n          : número de variables a mostrar.
    titulo         : título del gráfico.
    nombre_archivo : nombre para guardar. None = no guardar.
    """
    subregiones = df_shap_region.columns.tolist()
    top_vars    = df_shap_region.mean(axis=1).sort_values(ascending=False).head(top_n).index
    df_plot     = df_shap_region.loc[top_vars]
    etiq_y      = [ETIQUETAS_FEATURES.get(v, v) for v in top_vars]

    colores_sr = sns.color_palette("colorblind", len(subregiones))
    x = np.arange(len(top_vars))
    ancho = 0.8 / len(subregiones)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (sr, color) in enumerate(zip(subregiones, colores_sr)):
        ax.bar(x + i * ancho, df_plot[sr].values,
               ancho * 0.9, label=sr, color=color, alpha=0.85)

    ax.set_xticks(x + ancho * (len(subregiones) - 1) / 2)
    ax.set_xticklabels(etiq_y, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("|SHAP| medio")
    ax.set_title(titulo, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    if nombre_archivo:
        save_figure(nombre_archivo)
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — CONTRASTE TEÓRICO (notebook 06)
# ═════════════════════════════════════════════════════════════════════════════

def plot_convergencias_teoricas(
    df_conv: pd.DataFrame,
    titulo: str = "Convergencia entre determinantes algorítmicos y teoría democrática",
    nombre_archivo: Optional[str] = None,
) -> None:
    """
    Heatmap de convergencia entre los rankings SHAP y las predicciones
    teóricas (Easton 1975, Norris 2011, Devine 2024).

    Parámetros
    ----------
    df_conv       : DataFrame con índice=bloque temático y columnas=teorías,
                    valores=% de variables del bloque en el top-N del ranking.
    titulo        : título del gráfico.
    nombre_archivo: nombre para guardar. None = no guardar.
    """
    fig, ax = plt.subplots(figsize=(10, max(5, len(df_conv) * 0.6)))
    sns.heatmap(
        df_conv * 100, annot=True, fmt=".0f", cmap="YlGn",
        vmin=0, vmax=100,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "% variables del bloque en top-N", "shrink": 0.7},
        annot_kws={"size": 11},
    )
    ax.set_title(titulo, fontweight="bold", pad=12)
    ax.set_ylabel("Bloque temático")
    ax.set_xlabel("Marco teórico")
    ax.tick_params(axis="y", labelsize=9)

    if nombre_archivo:
        save_figure(nombre_archivo)
    plt.show()


def plot_tabla_convergencias(
    df_tabla: pd.DataFrame,
    titulo: str = "Tabla de convergencias y divergencias: algoritmo vs. teoría",
    nombre_archivo: Optional[str] = None,
) -> None:
    """
    Tabla visual de convergencias y divergencias.

    Parámetros
    ----------
    df_tabla      : DataFrame con columnas 'variable', 'etiqueta', 'bloque',
                    'ranking_shap', 'prediccion_teorica', 'convergencia'.
    titulo        : título de la figura.
    nombre_archivo: nombre para guardar. None = no guardar.
    """
    fig, ax = plt.subplots(figsize=(14, max(4, len(df_tabla) * 0.4)))
    ax.axis("off")

    cols_show = [c for c in ["etiqueta", "bloque", "ranking_shap",
                              "prediccion_teorica", "convergencia"]
                 if c in df_tabla.columns]
    tabla = ax.table(
        cellText=df_tabla[cols_show].values,
        colLabels=[c.replace("_", " ").title() for c in cols_show],
        cellLoc="center",
        loc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1, 1.4)

    # Color de encabezado
    for j in range(len(cols_show)):
        tabla[0, j].set_facecolor(THEME.get("semantic", {}).get("text", "#1E3A5F"))
        tabla[0, j].set_text_props(color="white", weight="bold")

    # Color de filas por convergencia
    for i, row in enumerate(df_tabla.itertuples()):
        conv = getattr(row, "convergencia", "")
        color = ("#d4edda" if "✓" in str(conv) or conv == "Convergencia"
                 else "#f8d7da" if "✗" in str(conv) or conv == "Divergencia"
                 else "#fff3cd")
        for j in range(len(cols_show)):
            tabla[i + 1, j].set_facecolor(color)

    ax.set_title(titulo, fontweight="bold", pad=15, fontsize=12)

    if nombre_archivo:
        save_figure(nombre_archivo)
    plt.show()
