"""
utils/io.py
===========
Funciones de entrada/salida para el proyecto de tesis.

Proporciona acceso centralizado a:
- Pipelines entrenados (artefactos completos por modelo y subperiodo)
- Resultados de métricas (CSV y Parquet)
- Splits procesados (Parquet por subperiodo)
- Valores SHAP calculados (Parquet por modelo y subperiodo)

Todas las rutas se resuelven desde PATHS definido en config.py,
de modo que el proyecto sea portable entre entornos.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict

from .config import PATHS


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINES
# ─────────────────────────────────────────────────────────────────────────────

def cargar_pipeline(nombre_modelo: str, subperiodo: str) -> Dict:
    """
    Carga el artefacto completo de pipeline para un modelo y subperiodo.

    El artefacto incluye el modelo entrenado, los preprocesadores ajustados
    (imputer, scaler), las features esperadas, los bloques temáticos, las
    etiquetas de features y la metadata de entrenamiento.

    Parámetros
    ----------
    nombre_modelo : 'OLO', 'XGBoost', 'CatBoost', 'LightGBM' o 'TabNet'
    subperiodo    : 'SP1', 'SP2' o 'SP3'

    Retorna
    -------
    dict con todos los componentes del artefacto.

    Lanza
    -----
    FileNotFoundError si el archivo no existe.
    """
    ruta = PATHS["FOLDER_MODELS"] / f"pipeline_{nombre_modelo}_{subperiodo}.pkl"
    if not ruta.exists():
        raise FileNotFoundError(
            f"Pipeline no encontrado: {ruta}\n"
            f"Asegúrate de haber ejecutado el notebook 02 de entrenamiento."
        )
    return joblib.load(ruta)


def listar_pipelines_disponibles() -> pd.DataFrame:
    """
    Devuelve un DataFrame con todos los pipelines disponibles en FOLDER_MODELS.

    Retorna
    -------
    pd.DataFrame con columnas: modelo, subperiodo, ruta, tamaño_kb.
    """
    patron = PATHS["FOLDER_MODELS"].glob("pipeline_*.pkl")
    filas = []
    for ruta in sorted(patron):
        partes = ruta.stem.replace("pipeline_", "").rsplit("_", 1)
        if len(partes) == 2:
            filas.append({
                "modelo"     : partes[0],
                "subperiodo" : partes[1],
                "ruta"       : str(ruta),
                "tamaño_kb"  : round(ruta.stat().st_size / 1024, 1),
            })
    return pd.DataFrame(filas) if filas else pd.DataFrame(
        columns=["modelo", "subperiodo", "ruta", "tamaño_kb"])


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADOS DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def cargar_resultados(split: str = "test") -> pd.DataFrame:
    """
    Carga el DataFrame de resultados de métricas generado en el notebook 02.

    Parámetros
    ----------
    split : 'test', 'val' o None (retorna todos los splits).

    Retorna
    -------
    pd.DataFrame con columnas: modelo, subperiodo, split, balanced_accuracy,
    f1_macro, f1_weighted, kappa_lineal, kappa_cuadratico, mae_ordinal,
    auroc_macro.

    Lanza
    -----
    FileNotFoundError si el archivo Parquet no existe.
    """
    ruta = PATHS["FOLDER_RESULTS"] / "resultados_modelos.parquet"
    if not ruta.exists():
        # Intentar fallback a CSV
        ruta_csv = PATHS["FOLDER_RESULTS"] / "resultados_modelos.csv"
        if ruta_csv.exists():
            df = pd.read_csv(ruta_csv)
        else:
            raise FileNotFoundError(
                f"Archivo de resultados no encontrado.\n"
                f"Rutas probadas:\n  {ruta}\n  {ruta_csv}\n"
                f"Asegúrate de haber ejecutado el notebook 02."
            )
    else:
        df = pd.read_parquet(ruta)

    if split is not None and "split" in df.columns:
        df = df[df["split"] == split].copy()
    return df


def cargar_mejor_modelo(subperiodo: str = "SP3",
                        metrica: str = "kappa_cuadratico") -> str:
    """
    Retorna el nombre del modelo con mayor valor en la métrica indicada
    para el subperiodo y split de test especificados.

    Parámetros
    ----------
    subperiodo : 'SP1', 'SP2' o 'SP3'.
    metrica    : columna de métrica a maximizar (default: 'kappa_cuadratico').

    Retorna
    -------
    str con el nombre del modelo ganador.
    """
    df = cargar_resultados(split="test")
    sub = df[df["subperiodo"] == subperiodo]
    if sub.empty:
        raise ValueError(f"No hay resultados para subperiodo '{subperiodo}'.")
    return sub.loc[sub[metrica].idxmax(), "modelo"]


# ─────────────────────────────────────────────────────────────────────────────
# SPLITS PROCESADOS (Parquet)
# ─────────────────────────────────────────────────────────────────────────────

def cargar_split_parquet(subperiodo: str,
                         split: str = "test") -> pd.DataFrame:
    """
    Carga un conjunto procesado (train o test) desde Parquet.

    Los archivos Parquet son generados por el notebook 02 y contienen
    las features transformadas más la columna 'target'.

    Parámetros
    ----------
    subperiodo : 'SP1', 'SP2' o 'SP3'.
    split      : 'train' o 'test'.

    Retorna
    -------
    pd.DataFrame con features + columna 'target'.
    """
    ruta = PATHS["FOLDER_PROCS"] / f"{subperiodo}_{split}.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"Split Parquet no encontrado: {ruta}\n"
            f"Ejecuta el notebook 02 para generarlo."
        )
    return pd.read_parquet(ruta)


def cargar_pesos_train(subperiodo: str) -> np.ndarray:
    """
    Carga los sample_weights del conjunto de entrenamiento para un subperiodo.

    Parámetros
    ----------
    subperiodo : 'SP1', 'SP2' o 'SP3'.

    Retorna
    -------
    np.ndarray con los pesos de cada registro.
    """
    ruta = PATHS["FOLDER_PROCS"] / f"{subperiodo}_train_weights.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"Pesos no encontrados: {ruta}")
    return pd.read_parquet(ruta)["sample_weight"].values


# ─────────────────────────────────────────────────────────────────────────────
# VALORES SHAP
# ─────────────────────────────────────────────────────────────────────────────

def guardar_shap_values(shap_array: np.ndarray,
                        feature_names: list,
                        nombre_modelo: str,
                        subperiodo: str,
                        clase: Optional[int] = None) -> None:
    """
    Persiste los valores SHAP en formato Parquet para reutilización
    entre notebooks sin necesidad de recalcularlos.

    Los valores SHAP pueden ser bidimensionales (n_muestras × n_features)
    para un análisis por clase específica, o tridimensionales
    (n_muestras × n_features × n_clases) para el análisis global.
    En el caso tridimensional se guarda la media absoluta entre clases.

    Parámetros
    ----------
    shap_array    : array NumPy de valores SHAP.
    feature_names : lista de nombres de features en el mismo orden que las
                    columnas de shap_array.
    nombre_modelo : nombre del modelo.
    subperiodo    : 'SP1', 'SP2' o 'SP3'.
    clase         : índice de clase (0–3) si se guardan valores por clase;
                    None para guardar la media absoluta entre clases.
    """
    PATHS["FOLDER_RESULTS_SHAP"].mkdir(parents=True, exist_ok=True)

    # Reducir a 2D si el array es 3D (n_muestras x n_features x n_clases)
    if shap_array.ndim == 3:
        arr_2d = np.abs(shap_array).mean(axis=2)
    else:
        arr_2d = shap_array

    sufijo = f"_clase{clase}" if clase is not None else ""
    nombre = f"shap_{nombre_modelo}_{subperiodo}{sufijo}.parquet"
    ruta   = PATHS["FOLDER_RESULTS_SHAP"] / nombre

    df = pd.DataFrame(arr_2d, columns=feature_names)
    df.to_parquet(ruta, index=False)
    print(f"  ✓ SHAP guardado: {nombre} ({ruta.stat().st_size / 1024:.0f} KB)")


def cargar_shap_values(nombre_modelo: str,
                       subperiodo: str,
                       clase: Optional[int] = None) -> pd.DataFrame:
    """
    Carga los valores SHAP previamente calculados y guardados.

    Parámetros
    ----------
    nombre_modelo : nombre del modelo.
    subperiodo    : 'SP1', 'SP2' o 'SP3'.
    clase         : índice de clase o None para la media absoluta.

    Retorna
    -------
    pd.DataFrame con forma (n_muestras × n_features).

    Lanza
    -----
    FileNotFoundError si los valores no han sido calculados aún.
    """
    sufijo = f"_clase{clase}" if clase is not None else ""
    nombre = f"shap_{nombre_modelo}_{subperiodo}{sufijo}.parquet"
    ruta   = PATHS["FOLDER_RESULTS_SHAP"] / nombre
    if not ruta.exists():
        raise FileNotFoundError(
            f"Valores SHAP no encontrados: {ruta}\n"
            f"Ejecuta la celda de cálculo SHAP en el notebook 04."
        )
    return pd.read_parquet(ruta)


def shap_disponible(nombre_modelo: str, subperiodo: str,
                    clase: Optional[int] = None) -> bool:
    """
    Verifica si los valores SHAP ya están calculados para un modelo
    y subperiodo sin lanzar excepción.
    """
    sufijo = f"_clase{clase}" if clase is not None else ""
    nombre = f"shap_{nombre_modelo}_{subperiodo}{sufijo}.parquet"
    return (PATHS["FOLDER_RESULTS_SHAP"] / nombre).exists()
