"""
utils/io.py
===========
Funciones de entrada/salida para el proyecto de tesis.

Diseño de split único (train 1995-2018, val 2020, test 2023-2024).
Convención de nombres de archivo:
  - pipeline_{modelo}_{estrategia}.pkl              (E1: variante ordinal)
  - pipeline_{modelo}_{estrategia}_binario.pkl      (E2: variante binaria)
  - hp_{modelo}_{estrategia}_{variante}.json        (registro de hiperparámetros)
  - train.parquet, val.parquet, test.parquet
  - shap_{modelo}_{estrategia}[_claseN].parquet
  - resultados_modelos.parquet / .csv

Estrategias de balanceo: sin_balanceo | pesos_clase | smotenc.
Para leer los hiperparámetros usar `utils.models.cargar_hiperparametros()`.

El modelo principal y su estrategia se resuelven con
`modelo_xai_seleccionado()`, que lee la selección escrita por el NB03: como el
nombre de los archivos SHAP incluye la estrategia, los notebooks 04, 05 y 06
deben partir de la misma fuente para no buscar archivos inexistentes.
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple

from .config import PATHS


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINES
# ─────────────────────────────────────────────────────────────────────────────

def cargar_pipeline(nombre_modelo: str, estrategia: str = "pesos_clase",
                    variante: str = None) -> Dict:
    """
    Carga el artefacto completo de pipeline para un modelo y estrategia.

    Parámetros
    ----------
    nombre_modelo : 'OLO', 'XGBoost', 'CatBoost', 'LightGBM' o 'TabNet'
    estrategia    : 'sin_balanceo', 'pesos_clase' o 'smotenc'
    variante      : None (E1, variante ordinal) o 'binario' (E2)
                    Genera el nombre: pipeline_{modelo}_{estrategia}_{variante}.pkl

    Retorna
    -------
    dict con todos los componentes del artefacto.
    """
    sufijo = f"_{variante}" if variante else ""
    ruta = PATHS["FOLDER_MODELS"] / f"pipeline_{nombre_modelo}_{estrategia}{sufijo}.pkl"
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


# ─────────────────────────────────────────────────────────────────────────────
# VALORES SHAP
# ─────────────────────────────────────────────────────────────────────────────

def normalizar_shap_2d(shap_vals,
                       n_features: int,
                       n_muestras: Optional[int] = None) -> np.ndarray:
    """
    Normaliza cualquier salida de `shap` a una matriz (n_muestras × n_features).

    La forma del retorno de `shap` cambia según la versión de la librería, el
    tipo de modelo y el número de clases:

    - lista de `n_clases` arrays (n_muestras × n_features) — shap < 0.45 multiclase
    - array (n_muestras × n_features × n_clases)           — shap ≥ 0.45 multiclase
    - array (n_clases × n_muestras × n_features)           — algunas versiones
    - array (n_muestras × n_features)                      — binario o regresión

    En los casos multiclase la dimensión de clases se agrega con la media del
    valor absoluto, que es la convención de importancia global del proyecto.
    Los ejes se identifican por su longitud, no por su posición, de modo que
    la función es estable ante cambios de versión de `shap`.

    Parámetros
    ----------
    shap_vals  : salida de `explainer.shap_values()`.
    n_features : número de features (obligatorio; desambigua los ejes).
    n_muestras : número de registros explicados. Si es None se asume que es
                 el mayor de los ejes restantes (n_muestras >> n_clases).

    Retorna
    -------
    Array 2-D (n_muestras × n_features).
    """
    if isinstance(shap_vals, (list, tuple)):
        arr = np.stack([np.asarray(v, dtype=float) for v in shap_vals], axis=-1)
    else:
        arr = np.asarray(shap_vals, dtype=float)

    if arr.ndim == 2:
        if arr.shape[1] == n_features:
            return arr
        if arr.shape[0] == n_features:
            return arr.T
        raise ValueError(
            f"Valores SHAP con forma {arr.shape} incompatibles con "
            f"{n_features} features."
        )

    if arr.ndim == 3:
        ejes = list(range(3))
        eje_f = next((i for i in ejes if arr.shape[i] == n_features), None)
        if eje_f is None:
            raise ValueError(
                f"Valores SHAP con forma {arr.shape}: ningún eje coincide con "
                f"{n_features} features."
            )
        restantes = [i for i in ejes if i != eje_f]
        if n_muestras is not None:
            eje_m = next((i for i in restantes if arr.shape[i] == n_muestras), None)
            if eje_m is None:
                raise ValueError(
                    f"Valores SHAP con forma {arr.shape}: ningún eje coincide "
                    f"con {n_muestras} registros."
                )
        else:
            # Sin n_muestras: el eje de clases es el más corto
            eje_m = max(restantes, key=lambda i: arr.shape[i])
        eje_c = next(i for i in restantes if i != eje_m)
        arr = np.transpose(arr, (eje_m, eje_f, eje_c))
        return np.abs(arr).mean(axis=2)

    raise ValueError(f"Valores SHAP con {arr.ndim} dimensiones no soportados.")


def ruta_shap(nombre_modelo: str,
              estrategia: str = "pesos_clase",
              clase: Optional[int] = None) -> Path:
    """
    Ruta del archivo de valores SHAP de un modelo.

    La estrategia de balanceo forma parte del nombre porque los valores SHAP
    de un mismo modelo difieren según cómo se trató el desbalance.

    Naming: ``shap_{modelo}_{estrategia}[_claseN].parquet``
    """
    sufijo = f"_clase{clase}" if clase is not None else ""
    return (PATHS["FOLDER_RESULTS_SHAP"] /
            f"shap_{nombre_modelo}_{estrategia}{sufijo}.parquet")


def shap_vigente(nombre_modelo: str,
                 estrategia: str = "pesos_clase",
                 clase: Optional[int] = None) -> bool:
    """
    Indica si los valores SHAP guardados corresponden al pipeline actual.

    Devuelve ``False`` si el archivo no existe o si el pipeline se reentrenó
    después de calcularlos, comparando la fecha de entrenamiento registrada en
    el artefacto con la fecha de modificación del Parquet. Sin esta
    comprobación, reutilizar el caché tras un reentrenamiento produciría
    explicaciones de un modelo que ya no es el que se reporta.
    """
    ruta = ruta_shap(nombre_modelo, estrategia, clase)
    if not ruta.exists():
        return False
    try:
        art = cargar_pipeline(nombre_modelo, estrategia)
        entrenado = art.get("fecha_entrenamiento")
        if not entrenado:
            return True
        from datetime import datetime
        t_modelo = datetime.fromisoformat(entrenado).timestamp()
        return ruta.stat().st_mtime >= t_modelo
    except Exception:                                            # noqa: BLE001
        # Si el pipeline no se puede inspeccionar, no se puede afirmar que el
        # caché esté obsoleto; se conserva y el llamador decide.
        return True


def guardar_shap_values(shap_array: np.ndarray,
                        feature_names: list,
                        nombre_modelo: str,
                        estrategia: str = "pesos_clase",
                        clase: Optional[int] = None) -> None:
    """
    Persiste los valores SHAP en formato Parquet.

    Parámetros
    ----------
    shap_array    : array NumPy (n_muestras × n_features) o
                    (n_muestras × n_features × n_clases). Se normaliza a 2-D
                    con `normalizar_shap_2d()`.
    feature_names : lista de nombres de features.
    nombre_modelo : nombre del modelo.
    estrategia    : estrategia de balanceo usada (se incluye en el nombre).
    clase         : índice de clase (0–3) o None para media absoluta.

    Naming: shap_{modelo}_{estrategia}[_claseN].parquet
    """
    PATHS["FOLDER_RESULTS_SHAP"].mkdir(parents=True, exist_ok=True)
    arr_2d = normalizar_shap_2d(shap_array, len(feature_names))
    ruta   = ruta_shap(nombre_modelo, estrategia, clase)
    pd.DataFrame(arr_2d, columns=feature_names).to_parquet(ruta, index=False)
    print(f"  ✓ SHAP guardado: {ruta.name} ({ruta.stat().st_size / 1024:.0f} KB)")


def listar_shap_disponibles() -> pd.DataFrame:
    """
    Lista los archivos de valores SHAP presentes en results/shap/.

    Retorna un DataFrame con columnas: modelo, estrategia, clase, archivo.
    Útil para diagnosticar desajustes entre lo que guardó el NB04 y lo que
    buscan los notebooks 05 y 06.
    """
    carpeta = PATHS["FOLDER_RESULTS_SHAP"]
    filas = []
    if carpeta.exists():
        for ruta in sorted(carpeta.glob("shap_*.parquet")):
            partes = ruta.stem.split("_")          # shap, modelo, estrategia…
            clase  = None
            if partes and partes[-1].startswith("clase"):
                clase  = partes[-1].replace("clase", "")
                partes = partes[:-1]
            filas.append({
                "modelo"    : partes[1] if len(partes) > 1 else "",
                "estrategia": "_".join(partes[2:]) if len(partes) > 2 else "",
                "clase"     : clase,
                "archivo"   : ruta.name,
            })
    return pd.DataFrame(filas, columns=["modelo", "estrategia", "clase", "archivo"])


def modelo_xai_seleccionado(fallback_modelo: str = "XGBoost",
                            fallback_estrategia: str = "pesos_clase",
                            verboso: bool = True) -> Tuple[str, str]:
    """
    Resuelve el modelo principal y su estrategia de balanceo.

    Lee `results/modelo_xai_seleccionado.json`, que escribe el NB03 al elegir
    el mejor modelo. Es la fuente única para los notebooks 04, 05 y 06: el
    nombre de los archivos SHAP incluye la estrategia, así que si cada
    notebook la fija por su cuenta los archivos no se encuentran.

    Retorna
    -------
    (nombre_modelo, estrategia_balanceo)
    """
    ruta = PATHS["FOLDER_RESULTS"] / "modelo_xai_seleccionado.json"
    if ruta.exists():
        try:
            sel = json.loads(ruta.read_text())
            modelo     = sel.get("modelo_xai", fallback_modelo)
            estrategia = sel.get("estrategia_balanceo", fallback_estrategia)
            if verboso:
                print(f"Modelo principal (selección del NB03): {modelo} [{estrategia}]")
            return modelo, estrategia
        except Exception as e:                                   # noqa: BLE001
            if verboso:
                print(f"⚠ {ruta.name} ilegible ({e}); se usa el respaldo.")
    if verboso:
        print(f"⚠ Sin {ruta.name} — se usa el respaldo "
              f"{fallback_modelo} [{fallback_estrategia}]. Ejecuta el NB03.")
    return fallback_modelo, fallback_estrategia


def cargar_shap_values(nombre_modelo: str,
                       estrategia: str = "pesos_clase",
                       clase: Optional[int] = None) -> pd.DataFrame:
    """Carga los valores SHAP previamente calculados.

    Parámetros
    ----------
    nombre_modelo : nombre del modelo.
    estrategia    : estrategia de balanceo (debe coincidir con la usada al guardar).
    clase         : índice de clase (0–3) o None para media absoluta.
    """
    ruta   = ruta_shap(nombre_modelo, estrategia, clase)
    nombre = ruta.name
    if not ruta.exists():
        disponibles = listar_shap_disponibles()
        detalle = ("\n  Archivos presentes en results/shap/:\n    " +
                   "\n    ".join(disponibles["archivo"].tolist())
                   if not disponibles.empty
                   else "\n  results/shap/ está vacío.")
        raise FileNotFoundError(
            f"Valores SHAP no encontrados: {nombre}\n"
            f"  Modelo '{nombre_modelo}' con estrategia '{estrategia}'."
            f"{detalle}\n"
            f"  Ejecuta la celda de cálculo SHAP del notebook 04 con la misma "
            f"estrategia, o revisa results/modelo_xai_seleccionado.json."
        )
    return pd.read_parquet(ruta)


def shap_disponible(nombre_modelo: str,
                    estrategia: str = "pesos_clase",
                    clase: Optional[int] = None) -> bool:
    """Verifica si los valores SHAP ya están calculados."""
    return ruta_shap(nombre_modelo, estrategia, clase).exists()
