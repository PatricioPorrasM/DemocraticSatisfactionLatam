"""
utils/io.py
===========
Funciones de entrada/salida para el proyecto de tesis.

Con el rediseño a split único, los artefactos ya no llevan
sufijo de subperiodo (SP1/SP2/SP3). Los nombres de archivo son:
  - pipeline_{modelo}.pkl
  - train.parquet, val.parquet, test.parquet
  - shap_{modelo}.parquet
  - resultados_modelos.parquet / .csv
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

def cargar_pipeline(nombre_modelo: str) -> Dict:
    """
    Carga el artefacto completo de pipeline para un modelo.

    Parámetros
    ----------
    nombre_modelo : 'OLO', 'XGBoost', 'CatBoost', 'LightGBM' o 'TabNet'

    Retorna
    -------
    dict con todos los componentes del artefacto.
    """
    ruta = PATHS["FOLDER_MODELS"] / f"pipeline_{nombre_modelo}.pkl"
    if not ruta.exists():
        raise FileNotFoundError(
            f"Pipeline no encontrado: {ruta}\n"
            f"Asegúrate de haber ejecutado el notebook 02."
        )
    try:
        import torch
        if not torch.cuda.is_available():
            _orig_load = torch.load
            torch.load = lambda f, *a, **kw: _orig_load(
                f, *a, **{**kw, "map_location": kw.get("map_location", "cpu")}
            )
            try:
                return joblib.load(ruta)
            finally:
                torch.load = _orig_load
    except ImportError:
        pass
    return joblib.load(ruta)


def listar_pipelines_disponibles() -> pd.DataFrame:
    """Devuelve un DataFrame con todos los pipelines disponibles."""
    patron = PATHS["FOLDER_MODELS"].glob("pipeline_*.pkl")
    filas = []
    for ruta in sorted(patron):
        nombre_modelo = ruta.stem.replace("pipeline_", "")
        filas.append({
            "modelo":     nombre_modelo,
            "ruta":       str(ruta),
            "tamaño_kb":  round(ruta.stat().st_size / 1024, 1),
        })
    return pd.DataFrame(filas) if filas else pd.DataFrame(
        columns=["modelo", "ruta", "tamaño_kb"])


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADOS DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def cargar_resultados(split: str = "test") -> pd.DataFrame:
    """
    Carga el DataFrame de resultados generado en el notebook 02.

    Parámetros
    ----------
    split : 'test', 'val' o None (retorna todos los splits).

    Retorna
    -------
    pd.DataFrame con columnas: modelo, estrategia_balanceo, variante_target,
    split, accuracy, balanced_accuracy, f1_macro, f1_weighted,
    kappa_lineal, kappa_cuadratico, mae_ordinal, auroc_macro.
    """
    ruta = PATHS["FILE_RESULTS_MODEL_PARQUET"]
    if not ruta.exists():
        ruta_csv = PATHS["FILE_RESULTS_MODEL_CSV"]
        if ruta_csv.exists():
            df = pd.read_csv(ruta_csv)
        else:
            raise FileNotFoundError(
                f"Archivo de resultados no encontrado.\n"
                f"Rutas probadas:\n  {ruta}\n  {ruta_csv}\n"
                f"Ejecuta el notebook 02."
            )
    else:
        df = pd.read_parquet(ruta)

    if split is not None and "split" in df.columns:
        df = df[df["split"] == split].copy()
    return df


def cargar_mejor_modelo(metrica: str = "kappa_cuadratico",
                        estrategia: str = None,
                        variante: str = None) -> str:
    """
    Retorna el nombre del modelo con mayor valor en la métrica indicada.

    Parámetros
    ----------
    metrica    : columna a maximizar (default: 'kappa_cuadratico').
    estrategia : filtrar por estrategia_balanceo (opcional).
    variante   : filtrar por variante_target (opcional).
    """
    df = cargar_resultados(split="test")
    if estrategia and "estrategia_balanceo" in df.columns:
        df = df[df["estrategia_balanceo"] == estrategia]
    if variante and "variante_target" in df.columns:
        df = df[df["variante_target"] == variante]
    if df.empty:
        raise ValueError("No hay resultados con los filtros indicados.")
    return df.loc[df[metrica].idxmax(), "modelo"]


# ─────────────────────────────────────────────────────────────────────────────
# SPLITS PROCESADOS
# ─────────────────────────────────────────────────────────────────────────────

def cargar_split_parquet(split: str = "test") -> pd.DataFrame:
    """
    Carga un conjunto procesado desde Parquet.

    Parámetros
    ----------
    split : 'train', 'val' o 'test'.
    """
    ruta = PATHS["FOLDER_PROCS"] / f"{split}.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"Split Parquet no encontrado: {ruta}\n"
            f"Ejecuta el notebook 02."
        )
    return pd.read_parquet(ruta)


def cargar_pesos_train() -> np.ndarray:
    """Carga los sample_weights del conjunto de entrenamiento."""
    ruta = PATHS["FOLDER_PROCS"] / "train_weights.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"Pesos no encontrados: {ruta}")
    return pd.read_parquet(ruta)["sample_weight"].values


# ─────────────────────────────────────────────────────────────────────────────
# VALORES SHAP
# ─────────────────────────────────────────────────────────────────────────────

def guardar_shap_values(shap_array: np.ndarray,
                        feature_names: list,
                        nombre_modelo: str,
                        clase: Optional[int] = None) -> None:
    """
    Persiste los valores SHAP en formato Parquet.

    Parámetros
    ----------
    shap_array    : array NumPy (n_muestras × n_features) o
                    (n_muestras × n_features × n_clases).
    feature_names : lista de nombres de features.
    nombre_modelo : nombre del modelo.
    clase         : índice de clase (0–3) o None para media absoluta.
    """
    PATHS["FOLDER_RESULTS_SHAP"].mkdir(parents=True, exist_ok=True)
    if shap_array.ndim == 3:
        arr_2d = np.abs(shap_array).mean(axis=2)
    else:
        arr_2d = shap_array
    sufijo = f"_clase{clase}" if clase is not None else ""
    nombre = f"shap_{nombre_modelo}{sufijo}.parquet"
    ruta   = PATHS["FOLDER_RESULTS_SHAP"] / nombre
    pd.DataFrame(arr_2d, columns=feature_names).to_parquet(ruta, index=False)
    print(f"  ✓ SHAP guardado: {nombre} ({ruta.stat().st_size / 1024:.0f} KB)")


def cargar_shap_values(nombre_modelo: str,
                       clase: Optional[int] = None) -> pd.DataFrame:
    """Carga los valores SHAP previamente calculados."""
    sufijo = f"_clase{clase}" if clase is not None else ""
    nombre = f"shap_{nombre_modelo}{sufijo}.parquet"
    ruta   = PATHS["FOLDER_RESULTS_SHAP"] / nombre
    if not ruta.exists():
        raise FileNotFoundError(
            f"Valores SHAP no encontrados: {ruta}\n"
            f"Ejecuta la celda de cálculo SHAP en el notebook 04."
        )
    return pd.read_parquet(ruta)


def shap_disponible(nombre_modelo: str, clase: Optional[int] = None) -> bool:
    """Verifica si los valores SHAP ya están calculados."""
    sufijo = f"_clase{clase}" if clase is not None else ""
    nombre = f"shap_{nombre_modelo}{sufijo}.parquet"
    return (PATHS["FOLDER_RESULTS_SHAP"] / nombre).exists()
