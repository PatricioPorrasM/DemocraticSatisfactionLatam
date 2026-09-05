"""
utils/plots.py
==============
Funciones de visualización reutilizables para el proyecto de tesis.

Organización:
- Sección 1: helpers base (colores, guardado)
- Sección 2: notebook 02 — EDA y matrices de correlación
- Sección 3: notebook 03 — evaluación comparativa
- Sección 4: notebook 04 — explicabilidad XAI
- Sección 5: notebook 05 — estabilidad temporal y regional
- Sección 6: notebook 06 — contraste teórico

Convención: todas las funciones que generan figuras reciben opcionalmente
`nombre_archivo` para guardar automáticamente. Si es None, no guardan.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from .config import (
    THEME, PATHS, ETIQUETAS, ETIQUETAS_FEATURES, BLOQUES,
    bloque_de,
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
    # El eje Y se invierte para que la primera variable quede arriba, igual que
    # en el heatmap adyacente (seaborn dibuja la fila 0 en la parte superior).
    ax_bar.set_ylim(len(variables) - 0.5, -0.5)
    ax_bar.axis("off")

    # Leyenda de bloques
    vistos = {}
    for b in bloques_lb:
        if b not in vistos:
            vistos[b] = colores_bloques.get(b, "#AAAAAA")
    patches = [mpatches.Patch(color=c, label=b) for b, c in vistos.items()]
    ax_bar.legend(handles=patches, loc="upper left",
                  bbox_to_anchor=(0, -0.02), fontsize=8, frameon=True)


def _bloques_en_eje(ax: plt.Axes, variables: List[str],
                    bloques_lb: Optional[List[str]] = None,
                    ancho: float = 0.55, sep: float = 0.25) -> List[mpatches.Patch]:
    """
    Dibuja la franja de bloques temáticos DENTRO del eje del heatmap.

    A diferencia de `_barra_bloques`, que usa un eje aparte, aquí los
    rectángulos se ubican en las coordenadas de datos del propio heatmap, de
    modo que la alineación se mantiene incluso con `square=True`.

    Retorna los parches para construir la leyenda.
    """
    if bloques_lb is None:
        bloques_lb = [bloque_de(v) for v in variables]

    colores_bloques = THEME.get("blocks", {})
    for i, bloque in enumerate(bloques_lb):
        ax.add_patch(mpatches.Rectangle(
            (-ancho - sep, i), ancho, 1,
            facecolor=colores_bloques.get(bloque, "#AAAAAA"),
            edgecolor="none", clip_on=False, zorder=5,
        ))

    vistos = {}
    for b in bloques_lb:
        if b not in vistos:
            vistos[b] = colores_bloques.get(b, "#AAAAAA")
    return [mpatches.Patch(color=c, label=b) for b, c in vistos.items()]


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — EDA Y MATRICES DE CORRELACIÓN (notebook 02)
# ═════════════════════════════════════════════════════════════════════════════

def plot_matriz_correlacion(
    datos: pd.DataFrame,
    variables: Optional[List[str]] = None,
    titulo: str = "Matriz de correlación",
    subtitulo: Optional[str] = None,
    metodo: str = "spearman",
    ordenar_por_bloque: bool = True,
    mostrar_bloques: bool = True,
    anotar: Optional[bool] = None,
    triangular: bool = True,
    umbral_alerta: float = 0.85,
    nombre_archivo: Optional[str] = None,
) -> pd.DataFrame:
    """
    Heatmap de la matriz de correlación entre variables.

    Diseñada para las tres matrices del NB02: Latinobarómetro, V-Dem y el
    dataset fusionado. Devuelve la matriz calculada para poder exportarla.

    Parámetros
    ----------
    datos             : DataFrame con las variables a correlacionar.
    variables         : subconjunto de columnas. Si None usa todas las numéricas.
    titulo            : título principal del gráfico.
    subtitulo         : segunda línea del título (unidad de análisis, n, etc.).
    metodo            : 'spearman' (por defecto), 'pearson' o 'kendall'.
    ordenar_por_bloque: reordena las variables según el orden de BLOQUES.
    mostrar_bloques   : dibuja la barra lateral de colores por bloque temático.
    anotar            : escribe el coeficiente en cada celda. Si None se
                        activa automáticamente con 30 variables o menos.
    triangular        : muestra solo el triángulo inferior y la diagonal.
    umbral_alerta     : informa por consola los pares con |r| por encima
                        de este valor (multicolinealidad).
    nombre_archivo    : nombre para guardar la figura. None = no guardar.

    Retorna
    -------
    DataFrame cuadrado con los coeficientes de correlación.
    """
    if variables is None:
        variables = [c for c in datos.columns
                     if pd.api.types.is_numeric_dtype(datos[c])]
    variables = [v for v in variables if v in datos.columns]

    if ordenar_por_bloque:
        ordenadas = [v for bloque in BLOQUES for v in BLOQUES[bloque]
                     if v in variables]
        ordenadas += [v for v in variables if v not in ordenadas]
        variables = ordenadas

    bloques_lb = [bloque_de(v) for v in variables]
    corr = datos[variables].corr(method=metodo)

    etiquetas = [ETIQUETAS_FEATURES.get(v, v) for v in variables]
    corr_plot = corr.copy()
    corr_plot.index   = etiquetas
    corr_plot.columns = etiquetas

    n = len(variables)
    if anotar is None:
        anotar = n <= 30
    # Solo tiene sentido pintar la barra de bloques si hay más de uno
    mostrar_bloques = mostrar_bloques and len(set(bloques_lb)) > 1

    lado = max(6.0, 0.42 * n + 3.5)
    mask = np.triu(np.ones_like(corr_plot, dtype=bool), k=1) if triangular else None

    fig, ax = plt.subplots(figsize=(lado, lado))

    sns.heatmap(
        corr_plot, mask=mask, annot=anotar, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.3, ax=ax,
        annot_kws={"size": max(5.5, 9 - 0.12 * n)},
        cbar_kws={"label": f"r {metodo.capitalize()}", "shrink": 0.55},
    )
    titulo_full = titulo if subtitulo is None else f"{titulo}\n{subtitulo}"
    ax.set_title(titulo_full, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=max(6, 9 - 0.06 * n), rotation=90)
    ax.tick_params(axis="y", labelsize=max(6, 9 - 0.06 * n), rotation=0)

    if mostrar_bloques:
        ancho_barra, sep_barra = 0.55, 0.25
        patches = _bloques_en_eje(ax, variables, bloques_lb,
                                  ancho=ancho_barra, sep=sep_barra)
        # Desplaza las etiquetas del eje Y para que la franja de bloques no
        # las tape: el desplazamiento se calcula en puntos a partir del
        # tamaño real de celda tras un primer render.
        fig.canvas.draw()
        celda_pt = ax.get_window_extent().width / max(n, 1) / fig.dpi * 72
        ax.tick_params(axis="y", pad=(ancho_barra + sep_barra) * celda_pt + 3)

        # Con el triángulo superior enmascarado esa zona queda libre: es el
        # lugar natural para la leyenda sin tapar celdas ni etiquetas.
        if triangular and n >= 10:
            ax.legend(handles=patches, loc="upper right", fontsize=9,
                      frameon=True, title="Bloque temático", title_fontsize=9)
        else:
            ax.legend(handles=patches, loc="upper left",
                      bbox_to_anchor=(0, -0.25), fontsize=8, frameon=True,
                      title="Bloque temático", title_fontsize=8)

    if nombre_archivo:
        save_figure(nombre_archivo)
    plt.show()

    # Reporte de pares altamente correlacionados
    pares = []
    for i in range(n):
        for j in range(i):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= umbral_alerta:
                pares.append((variables[i], variables[j], r))
    print(f"  Variables: {n} | método: {metodo}")
    if pares:
        print(f"  Pares con |r| ≥ {umbral_alerta}: {len(pares)}")
        for a, b, r in sorted(pares, key=lambda t: -abs(t[2])):
            print(f"    {ETIQUETAS_FEATURES.get(a, a):<38} ~ "
                  f"{ETIQUETAS_FEATURES.get(b, b):<38} r={r:+.3f}")
    else:
        print(f"  Ningún par supera |r| = {umbral_alerta} (sin multicolinealidad extrema)")

    return corr


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — EVALUACIÓN COMPARATIVA (notebook 03)
# ═════════════════════════════════════════════════════════════════════════════

def plot_metricas_comparativas(
    df_res: pd.DataFrame,
    metricas: Optional[List[Tuple[str, str]]] = None,
    nombre_archivo: Optional[str] = "03_metricas_comparativas",
) -> None:
    """
    Gráfico de barras con rendimiento de cada modelo por estrategia de balanceo.

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
        "(conjunto de prueba)",
        fontsize=13, fontweight="bold",
    )

    for ax, (metrica, titulo) in zip(axes, metricas):
        for modelo in modelos:
            sub = df_res[df_res["modelo"] == modelo]
            if sub.empty or metrica not in sub.columns:
                continue
            ax.plot(sub["estrategia_balanceo"], sub[metrica],
                    marker="o", linewidth=2, markersize=7,
                    label=modelo, color=model_color(modelo))
        ax.set_title(titulo, fontweight="bold")
        ax.set_xlabel("Estrategia de Balanceo")
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
    resultados_cm : dict con claves (modelo, estrategia) y valores
                    (y_true, y_pred).
    solo_mejor    : si True, solo dibuja el mejor modelo (requiere
                    mejor_modelo).
    mejor_modelo  : nombre del modelo a mostrar cuando solo_mejor=True.
    nombre_archivo: nombre para guardar. None = no guardar.
    """
    from sklearn.metrics import confusion_matrix

    etiq_cortas = ["Nada\nsat.", "No muy\nsat.", "Más bien\nsat.", "Muy\nsat."]
    sps = ["sin_balanceo", "pesos_clase", "smotenc"]

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
        "Conjunto de prueba 2023+2024 por estrategia de balanceo",
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


def plot_matriz_confusion_modelo(
    y_true,
    y_pred,
    etiquetas: Optional[Dict] = None,
    titulo: str = "Matriz de confusión",
    subtitulo: Optional[str] = None,
    resaltar_distancia: Optional[int] = 2,
    nombre_archivo: Optional[str] = None,
) -> None:
    """
    Matriz de confusión del modelo principal en dos paneles:
    conteos absolutos y porcentaje por clase real.

    Parámetros
    ----------
    y_true, y_pred     : etiquetas reales y predichas.
    etiquetas          : dict {clase: etiqueta}. Por defecto ETIQUETAS.
    titulo             : título principal.
    subtitulo          : segunda línea del título (modelo, estrategia, n).
    resaltar_distancia : marca en rojo las celdas con distancia ordinal
                         mayor o igual a este valor. None desactiva el resaltado.
    nombre_archivo     : nombre para guardar. None = no guardar.
    """
    from sklearn.metrics import confusion_matrix

    clases = sorted(set(np.asarray(y_true).ravel().tolist()) |
                    set(np.asarray(y_pred).ravel().tolist()))
    clases = [int(c) for c in clases]
    ets    = etiquetas if etiquetas is not None else (
        ETIQUETAS if len(clases) == len(ETIQUETAS) else {})
    # Eje X: etiqueta partida en varias líneas; eje Y: etiqueta en una línea
    etiq_x = ["\n".join(ets[c].split()) if c in ets else f"Clase {c}"
              for c in clases]
    etiq_y = [ets[c] if c in ets else f"Clase {c}" for c in clases]

    cm_n   = confusion_matrix(y_true, y_pred, labels=clases)
    cm_pct = confusion_matrix(y_true, y_pred, labels=clases, normalize="true") * 100

    fig, axes = plt.subplots(1, 2, figsize=(7 + 1.4 * len(clases), 5.5))
    for ax, datos, fmt, etiqueta_cb, sub in [
        (axes[0], cm_n,   ",d",  "Registros",        "Conteos absolutos"),
        (axes[1], cm_pct, ".1f", "% por clase real", "Normalizada por fila"),
    ]:
        sns.heatmap(
            datos, annot=True, fmt=fmt, cmap="Blues", ax=ax,
            xticklabels=etiq_x, yticklabels=etiq_y,
            linewidths=0.4, annot_kws={"size": 9},
            cbar_kws={"label": etiqueta_cb, "shrink": 0.7},
        )
        ax.set_title(sub, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_xticklabels(etiq_x, rotation=0, fontsize=9)
        ax.set_yticklabels(etiq_y, rotation=0, fontsize=9)
        if resaltar_distancia is not None:
            for i in range(len(clases)):
                for j in range(len(clases)):
                    if abs(clases[i] - clases[j]) >= resaltar_distancia:
                        ax.add_patch(plt.Rectangle(
                            (j, i), 1, 1, fill=False,
                            edgecolor=THEME["semantic"]["danger"], lw=2.0))

    titulo_full = titulo if subtitulo is None else f"{titulo}\n{subtitulo}"
    if resaltar_distancia is not None:
        titulo_full += (f"\nContorno rojo: error ordinal ≥ {resaltar_distancia} clases")
    fig.suptitle(titulo_full, fontweight="bold", fontsize=12)

    if nombre_archivo:
        save_figure(nombre_archivo)
    plt.show()


def plot_metricas_por_clase(
    df_clases: pd.DataFrame,
    titulo: str = "Precision, recall y F1 por categoría",
    subtitulo: Optional[str] = None,
    nombre_archivo: Optional[str] = None,
) -> None:
    """
    Barras agrupadas de precision, recall y F1 para cada categoría del target.

    Parámetros
    ----------
    df_clases      : DataFrame devuelto por `utils.metrics.metricas_por_clase`.
                     Las filas de promedio (clase nula) se ignoran.
    titulo         : título principal.
    subtitulo      : segunda línea del título (modelo, estrategia, split).
    nombre_archivo : nombre para guardar. None = no guardar.
    """
    df_p = df_clases[df_clases["clase"].notna()].copy()
    if df_p.empty:
        print("  ⚠ Sin categorías para graficar.")
        return

    etiquetas = df_p["etiqueta"].tolist()
    metricas  = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    colores   = ["#4C78A8", "#F58518", "#54A24B"]

    x     = np.arange(len(df_p))
    ancho = 0.8 / len(metricas)

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(df_p)), 5))
    for i, ((col, etiq), color) in enumerate(zip(metricas, colores)):
        vals = df_p[col].values
        barras = ax.bar(x + i * ancho, vals, ancho * 0.92,
                        label=etiq, color=color, edgecolor="white")
        for b, v in zip(barras, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012,
                    f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x + ancho * (len(metricas) - 1) / 2)
    ax.set_xticklabels(
        [f"{e}\n(n={int(s):,})" for e, s in zip(etiquetas, df_p["soporte"])],
        fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Valor de la métrica")
    titulo_full = titulo if subtitulo is None else f"{titulo}\n{subtitulo}"
    ax.set_title(titulo_full, fontweight="bold")
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    if nombre_archivo:
        save_figure(nombre_archivo)
    plt.show()


def plot_rendimiento_por_pais(
    df_mae: pd.DataFrame,

    nombre_archivo: Optional[str] = None,
) -> None:
    """
    Gráfico de barras horizontales del MAE ordinal por país.

    Parámetros
    ----------
    df_mae        : DataFrame con columnas 'pais', 'modelo', 'mae_ordinal'.

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
    ax.set_title("MAE Ordinal por país — conjunto de prueba 2023+2024", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Umbral 0.5")
    ax.grid(True, axis="x", alpha=0.3)

    if nombre_archivo:
        save_figure(nombre_archivo)
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — EXPLICABILIDAD XAI (notebook 04)
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


def plot_importancia_con_ic(
    df_boot: pd.DataFrame,
    top_n: int = 15,
    titulo: str = "Importancia global con intervalo de confianza",
    subtitulo: Optional[str] = None,
    columna_valor: str = "importancia",
    nombre_archivo: Optional[str] = None,
) -> None:
    """
    Barras horizontales de importancia con barras de error del bootstrap.

    Parámetros
    ----------
    df_boot       : salida de `utils.xai.bootstrap_importancia_shap`, con las
                    columnas etiqueta, importancia, ic_inf, ic_sup y rango.
    top_n         : número de variables a mostrar.
    titulo        : título principal.
    subtitulo     : segunda línea del título (modelo, clúster, réplicas).
    columna_valor : columna con la estimación puntual.
    nombre_archivo: nombre para guardar. None = no guardar.
    """
    top = df_boot.head(top_n).iloc[::-1]
    etiquetas = [
        f"{e}  [{int(a)}–{int(b)}]"
        for e, a, b in zip(top["etiqueta"], top["rango_ic_inf"], top["rango_ic_sup"])
    ] if {"rango_ic_inf", "rango_ic_sup"}.issubset(top.columns) else list(top["etiqueta"])

    colores = [THEME.get("blocks", {}).get(b, "#AAAAAA")
               for b in top.get("bloque", ["" ] * len(top))]
    err_inf = (top[columna_valor] - top["ic_inf"]).clip(lower=0).values
    err_sup = (top["ic_sup"] - top[columna_valor]).clip(lower=0).values

    fig, ax = plt.subplots(figsize=(11, max(5, top_n * 0.42)))
    ax.barh(etiquetas, top[columna_valor].values, color=colores,
            edgecolor="white", linewidth=0.4)
    ax.errorbar(top[columna_valor].values, np.arange(len(top)),
                xerr=[err_inf, err_sup], fmt="none",
                ecolor=THEME["semantic"]["text"], elinewidth=1.2, capsize=3)
    ax.set_xlabel("|SHAP| medio (con IC 95% por bootstrap de clústeres)")
    titulo_full = titulo if subtitulo is None else f"{titulo}\n{subtitulo}"
    ax.set_title(titulo_full, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=9)

    vistos = {}
    for b in top.get("bloque", []):
        if b and b not in vistos:
            vistos[b] = THEME.get("blocks", {}).get(b, "#AAAAAA")
    if vistos:
        patches = [mpatches.Patch(color=c, label=b) for b, c in vistos.items()]
        ax.legend(handles=patches, fontsize=8, loc="lower right",
                  title="Bloque temático", title_fontsize=8)

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
    # La curva describe el desplazamiento de la PREDICCIÓN del modelo respecto
    # de su promedio, no un efecto sobre la satisfacción observada.
    ax.set_ylabel("Cambio en la predicción del modelo (ALE)")
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
# SECCIÓN 5 — ESTABILIDAD TEMPORAL Y REGIONAL (notebook 05)
# ═════════════════════════════════════════════════════════════════════════════

def plot_spearman_estabilidad(
    df_correlaciones: pd.DataFrame,
    titulo: str = "Correlación Spearman entre rankings SHAP",
    nombre_archivo: Optional[str] = None,
) -> None:
    """
    Heatmap de correlación de Spearman entre rankings de importancia SHAP
    de distintas configuraciones o modelos (triangular superior).

    Parámetros
    ----------
    df_correlaciones : DataFrame cuadrado (modelos × modelos o configs × configs)
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
# SECCIÓN 6 — CONTRASTE TEÓRICO (notebook 06)
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
