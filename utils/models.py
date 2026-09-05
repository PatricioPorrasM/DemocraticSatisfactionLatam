"""
utils/models.py
===============
Entrenamiento de los cinco modelos del proyecto con optimización Optuna (TPE)
maximizando el Kappa cuadrático en el conjunto de validación.

Modelos
-------
- **OLO**: regresión logística ordinal acumulativa (``mord.LogisticIT``), un
  único vector de coeficientes y K-1 umbrales. Es la línea base ordinal del
  diseño experimental, por lo que la ausencia de ``mord`` interrumpe la
  ejecución en lugar de sustituirla por un modelo multinomial.
- **XGBoost**, **CatBoost**, **LightGBM**: árboles de gradiente.
- **TabNet**: red de atención tabular.

Tratamiento del desbalance
--------------------------
Las tres estrategias de balanceo (``sin_balanceo``, ``pesos_clase``,
``smotenc``) se traducen a cada familia de modelos así:

- OLO y árboles de gradiente reciben ``sample_weight``, que combina el factor
  de expansión muestral con el peso de clase cuando corresponde.
- TabNet no admite ``sample_weight`` por registro, así que el peso de clase se
  aplica en la función de pérdida (entropía cruzada ponderada) y el muestreo
  se deja uniforme (``weights=0``) en las tres estrategias.

Registro de hiperparámetros
---------------------------
Cada función de entrenamiento guarda en `models/hp_{modelo}_{estrategia}_{variante}.json`
el registro COMPLETO de la configuración del modelo, no solo los parámetros
que optimiza Optuna:

- `hp_optimizados`         : parámetros buscados por Optuna (`study.best_params`).
- `hp_fijos`               : parámetros fijados por diseño (objective, semilla,
                             device, n_jobs, verbosidad, número de clases, …).
- `hp_completos`           : unión de ambos = todo lo que recibe el constructor.
- `espacio_busqueda`       : rango y tipo de cada parámetro optimizado.
- `config_entrenamiento`   : configuración del `.fit()` (early stopping, épocas,
                             batch size, eval_set, uso de sample_weight, …).
- `params_efectivos_modelo`: `get_params()` del estimador ya entrenado, que
                             incluye también los valores por defecto de la
                             librería.

Se accede con `cargar_hiperparametros()` y se resume con `tabla_hiperparametros()`.
"""

import json
import warnings
import numpy as np
import pandas as pd
import torch
import optuna
from optuna.samplers import TPESampler
from scipy.optimize import OptimizeWarning
from sklearn.metrics import cohen_kappa_score
from datetime import datetime
from typing import Dict, Optional, Tuple

from .config import PATHS, PARAMETERS, N_CLASES, VARS_CATEGORICAS
from .metrics import evaluar

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import mord as _mord
    MORD_OK = True
    # mord 0.7 pasa la opción 'disp' a scipy.optimize.minimize, que ya no la
    # reconoce; el aviso no afecta la solución obtenida.
    warnings.filterwarnings("ignore", category=OptimizeWarning,
                            module="mord.threshold_based")
except ImportError:
    MORD_OK = False

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    CatBoostClassifier = Pool = None

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    TabNetClassifier = None


# ═════════════════════════════════════════════════════════════════════════════
# REGISTRO DE HIPERPARÁMETROS
# ═════════════════════════════════════════════════════════════════════════════

def _json_seguro(obj):
    """Convierte cualquier valor a una representación serializable en JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_seguro(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_seguro(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if callable(obj):
        return getattr(obj, "__name__", str(obj))
    texto = str(obj)
    # Evita volcar reprs enormes (p. ej. la red de TabNet) en el JSON
    return texto if len(texto) <= 500 else texto[:500] + " …[truncado]"


def ruta_hiperparametros(nombre_modelo: str, estrategia: str,
                         variante_target: str = "ordinal_4clases",
                         sufijo: str = ""):
    """
    Ruta del archivo JSON de hiperparámetros.

    La variante del target forma parte del nombre para que los modelos de E1
    (ordinal) y de E2 (binario) no se sobrescriban entre sí. El sufijo separa
    los pliegues temporales (``"_fold1"``, …) del corte definitivo (``""``).
    """
    return (PATHS["FOLDER_MODELS"] /
            f"hp_{nombre_modelo}_{estrategia}_{variante_target}{sufijo}.json")


def _params_efectivos(clf) -> Dict:
    """Parámetros efectivos del estimador entrenado, incluidos los por defecto."""
    for extractor in (lambda m: m.get_params(),
                      lambda m: m.get_all_params(),
                      lambda m: vars(m)):
        try:
            params = extractor(clf)
            if isinstance(params, dict):
                return _json_seguro({k: v for k, v in params.items()
                                     if not k.startswith("_")})
        except Exception:
            continue
    return {}


def guardar_hiperparametros(
    nombre_modelo: str,
    estrategia: str,
    variante_target: str,
    hp_optimizados: Dict,
    hp_fijos: Optional[Dict] = None,
    espacio_busqueda: Optional[Dict] = None,
    config_entrenamiento: Optional[Dict] = None,
    clf=None,
    n_trials: Optional[int] = None,
    mejor_kappa_val: Optional[float] = None,
    sufijo: str = "",
) -> Dict:
    """
    Persiste el registro completo de hiperparámetros de un modelo.

    Retorna el diccionario guardado.
    """
    hp_fijos = hp_fijos or {}
    registro = {
        "modelo"                  : nombre_modelo,
        "estrategia_balanceo"     : estrategia,
        "variante_target"         : variante_target,
        "pliegue"                 : sufijo.lstrip("_") or "final",
        # Deja constancia de si el artefacto viene de la corrida definitiva o
        # de una prueba de humo, para que sus cifras no se confundan.
        "modo_ejecucion"          : PARAMETERS["MODO_EJECUCION"],
        "semilla"                 : PARAMETERS["SEED"],
        "n_trials_optuna"         : n_trials,
        "mejor_kappa_val_optuna"  : (round(float(mejor_kappa_val), 6)
                                     if mejor_kappa_val is not None else None),
        "hp_optimizados"          : _json_seguro(hp_optimizados),
        "hp_fijos"                : _json_seguro(hp_fijos),
        "hp_completos"            : _json_seguro({**hp_fijos, **hp_optimizados}),
        "espacio_busqueda"        : _json_seguro(espacio_busqueda or {}),
        "config_entrenamiento"    : _json_seguro(config_entrenamiento or {}),
        "params_efectivos_modelo" : _params_efectivos(clf) if clf is not None else {},
        "fecha_registro"          : datetime.now().isoformat(),
    }
    ruta = ruta_hiperparametros(nombre_modelo, estrategia, variante_target, sufijo)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(registro, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return registro


def cargar_hiperparametros(nombre_modelo: str, estrategia: str,
                           variante_target: str = "ordinal_4clases",
                           sufijo: str = "") -> Dict:
    """Carga el registro de hiperparámetros de un modelo."""
    ruta = ruta_hiperparametros(nombre_modelo, estrategia, variante_target, sufijo)
    if not ruta.exists():
        raise FileNotFoundError(
            f"Registro de hiperparámetros no encontrado: {ruta}\n"
            f"Ejecuta el notebook 02."
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


def tabla_hiperparametros(carpeta=None) -> pd.DataFrame:
    """
    Resume todos los registros de hiperparámetros disponibles en models/.

    Retorna un DataFrame con una fila por modelo × estrategia × variante y los
    hiperparámetros completos serializados, apto para exportar a CSV.
    """
    carpeta = carpeta or PATHS["FOLDER_MODELS"]
    filas = []
    for ruta in sorted(carpeta.glob("hp_*.json")):
        try:
            d = json.loads(ruta.read_text(encoding="utf-8"))
        except Exception as e:                                  # noqa: BLE001
            print(f"  ⚠ No se pudo leer {ruta.name}: {e}")
            continue
        hp_completos = d.get("hp_completos", {})
        filas.append({
            "modelo"              : d.get("modelo"),
            "estrategia_balanceo" : d.get("estrategia_balanceo"),
            "variante_target"     : d.get("variante_target"),
            "pliegue"             : d.get("pliegue", "final"),
            "n_hp_optimizados"    : len(d.get("hp_optimizados", {})),
            "n_hp_completos"      : len(hp_completos),
            "n_trials_optuna"     : d.get("n_trials_optuna"),
            "kappa_val_optuna"    : d.get("mejor_kappa_val_optuna"),
            "hp_optimizados"      : json.dumps(d.get("hp_optimizados", {}),
                                               ensure_ascii=False),
            "hp_completos"        : json.dumps(hp_completos, ensure_ascii=False),
            "config_entrenamiento": json.dumps(d.get("config_entrenamiento", {}),
                                               ensure_ascii=False),
            "archivo"             : ruta.name,
        })
    if not filas:
        return pd.DataFrame(columns=[
            "modelo", "estrategia_balanceo", "variante_target", "pliegue",
            "n_hp_optimizados", "n_hp_completos", "n_trials_optuna",
            "kappa_val_optuna", "hp_optimizados", "hp_completos",
            "config_entrenamiento", "archivo"])
    return pd.DataFrame(filas).sort_values(
        ["pliegue", "variante_target", "modelo", "estrategia_balanceo"]
    ).reset_index(drop=True)


def _hp_previos(nombre: str, estrategia: str, variante: str, cfg: dict):
    """
    Devuelve (hp_optimizados, kappa_val, n_trials) si se deben reutilizar los
    HPs ya guardados; (None, None, None) si hay que ejecutar la búsqueda.

    El kappa y el número de trials provienen del registro original, no de la
    configuración actual, para no falsear la procedencia de los valores.
    """
    if cfg["ejecutar_hp"]:
        return None, None, None
    sufijo = cfg.get("sufijo_hp", "")
    ruta = ruta_hiperparametros(nombre, estrategia, variante, sufijo)
    if not ruta.exists():
        return None, None, None
    registro = cargar_hiperparametros(nombre, estrategia, variante, sufijo)
    hp = registro.get("hp_optimizados", {})
    print(f"  HPs cargados desde {ruta.name}: {hp}")
    return (hp, registro.get("mejor_kappa_val_optuna"),
            registro.get("n_trials_optuna"))


# ═════════════════════════════════════════════════════════════════════════════
# ESPACIOS DE BÚSQUEDA (documentación del registro de hiperparámetros)
# Deben mantenerse sincronizados con las llamadas trial.suggest_* de cada modelo.
# ═════════════════════════════════════════════════════════════════════════════

ESPACIO_OLO = {
    "alpha": {"tipo": "float", "min": 1e-4, "max": 10.0, "log": True},
}

ESPACIO_XGBOOST = {
    "n_estimators"    : {"tipo": "int",   "min": 200,  "max": 1000, "paso": 100},
    "max_depth"       : {"tipo": "int",   "min": 3,    "max": 8},
    "learning_rate"   : {"tipo": "float", "min": 0.01, "max": 0.3, "log": True},
    "subsample"       : {"tipo": "float", "min": 0.6,  "max": 1.0},
    "colsample_bytree": {"tipo": "float", "min": 0.6,  "max": 1.0},
    "min_child_weight": {"tipo": "int",   "min": 1,    "max": 10},
    "reg_alpha"       : {"tipo": "float", "min": 1e-8, "max": 10.0, "log": True},
    "reg_lambda"      : {"tipo": "float", "min": 1e-8, "max": 10.0, "log": True},
}

ESPACIO_CATBOOST = {
    "iterations"         : {"tipo": "int",   "min": 300,  "max": 1000, "paso": 100},
    "depth"              : {"tipo": "int",   "min": 4,    "max": 8},
    "learning_rate"      : {"tipo": "float", "min": 0.01, "max": 0.3, "log": True},
    "l2_leaf_reg"        : {"tipo": "float", "min": 1.0,  "max": 10.0},
    "bagging_temperature": {"tipo": "float", "min": 0.0,  "max": 1.0},
    "border_count"       : {"tipo": "int",   "min": 32,   "max": 128},
    "random_strength"    : {"tipo": "float", "min": 0.0,  "max": 10.0},
}

ESPACIO_LIGHTGBM = {
    "n_estimators"     : {"tipo": "int",   "min": 200,  "max": 1000, "paso": 100},
    "num_leaves"       : {"tipo": "int",   "min": 20,   "max": 150},
    "max_depth"        : {"tipo": "int",   "min": 3,    "max": 8},
    "learning_rate"    : {"tipo": "float", "min": 0.01, "max": 0.3, "log": True},
    "subsample"        : {"tipo": "float", "min": 0.6,  "max": 1.0},
    "colsample_bytree" : {"tipo": "float", "min": 0.6,  "max": 1.0},
    "reg_alpha"        : {"tipo": "float", "min": 1e-8, "max": 10.0, "log": True},
    "reg_lambda"       : {"tipo": "float", "min": 1e-8, "max": 10.0, "log": True},
    "min_child_samples": {"tipo": "int",   "min": 20,   "max": 100},
}

ESPACIO_TABNET = {
    "n_d"          : {"tipo": "int",         "min": 8,    "max": 64, "paso": 8},
    "n_a"          : {"tipo": "int",         "min": 8,    "max": 64, "paso": 8},
    "n_steps"      : {"tipo": "int",         "min": 3,    "max": 7},
    "gamma"        : {"tipo": "float",       "min": 1.0,  "max": 2.0},
    "lambda_sparse": {"tipo": "float",       "min": 1e-6, "max": 1e-3, "log": True},
    "momentum"     : {"tipo": "float",       "min": 0.01, "max": 0.4},
    "mask_type"    : {"tipo": "categorical", "opciones": ["sparsemax", "entmax"]},
    "lr"           : {"tipo": "float",       "min": 1e-4, "max": 1e-2, "log": True},
}


# ═════════════════════════════════════════════════════════════════════════════
# MODELOS
# ═════════════════════════════════════════════════════════════════════════════

def entrenar_olo(
    X_tr, y_tr, X_val, y_val, X_te, y_te, w_tr, w_val,
    estrategia: str, variante_target: str = "ordinal_4clases",
    cfg: dict = None,
) -> Tuple:
    nombre = "OLO"
    seed   = PARAMETERS["SEED"]
    # mord.LogisticIT resuelve el problema con L-BFGS-B sobre un objetivo
    # escrito en Python puro, así que cada ensayo cuesta bastante más que uno
    # de los modelos arbóreos y se le asigna un presupuesto propio.
    n_trials_efectivos = cfg.get("n_trials_olo", cfg["n_trials"])

    print(f"{'='*52}  Entrenando {nombre} — {estrategia}  {'='*52}")
    if not MORD_OK:
        raise ImportError(
            "mord no está instalado y la línea base ordinal es parte del diseño "
            "experimental. Instalar con: pip install mord>=0.7"
        )

    best_hp, kappa_val, n_trials_reg = _hp_previos(
        nombre, estrategia, variante_target, cfg)
    if best_hp is None:
        X_tr_np, y_tr_np   = np.array(X_tr), np.array(y_tr)
        X_val_np, y_val_np = np.array(X_val), np.array(y_val)

        def obj(trial):
            alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
            m = _mord.LogisticIT(alpha=alpha, max_iter=500)
            m.fit(X_tr_np, y_tr_np, sample_weight=w_tr)
            return cohen_kappa_score(y_val_np, m.predict(X_val_np), weights="quadratic")

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(obj, n_trials=n_trials_efectivos, show_progress_bar=False)
        best_hp, kappa_val = study.best_params, study.best_value
        n_trials_reg = n_trials_efectivos
        print(f"  Mejor Kappa Val: {kappa_val:.4f} | {best_hp}")
        guardar_hiperparametros(nombre, estrategia, variante_target, best_hp,
                                espacio_busqueda=ESPACIO_OLO,
                                n_trials=n_trials_reg, mejor_kappa_val=kappa_val,
                                sufijo=cfg.get("sufijo_hp", ""))

    alpha = best_hp.get("alpha", 1.0)
    hp_fijos = {"max_iter": 500, "implementacion": "mord.LogisticIT",
                "formulacion": "logit acumulativo con umbrales de umbral "
                               "inmediato (immediate-threshold)"}
    clf = _mord.LogisticIT(alpha=alpha, max_iter=500)
    clf.fit(np.array(X_tr), np.array(y_tr), sample_weight=w_tr)

    guardar_hiperparametros(
        nombre, estrategia, variante_target, best_hp,
        hp_fijos=hp_fijos, espacio_busqueda=ESPACIO_OLO,
        config_entrenamiento={
            "usa_sample_weight": True,
            "entrada": "X normalizada con StandardScaler",
            "n_clases": int(len(np.unique(np.array(y_tr)))),
            # Evidencia de que el modelo es ordinal y no multinomial: un único
            # vector de coeficientes compartido y K-1 umbrales estimados.
            "n_coeficientes": int(np.size(clf.coef_)),
            "umbrales_theta": [round(float(t), 6) for t in np.atleast_1d(clf.theta_)],
        },
        clf=clf, n_trials=n_trials_reg, mejor_kappa_val=kappa_val,
        sufijo=cfg.get("sufijo_hp", ""),
    )

    y_pred_val = clf.predict(np.array(X_val))
    y_prob_val = clf.predict_proba(np.array(X_val))
    y_pred_te  = clf.predict(np.array(X_te))
    y_prob_te  = clf.predict_proba(np.array(X_te))

    m_val = evaluar(y_val, y_pred_val, y_prob_val, nombre, estrategia_balanceo=estrategia, variante_target=variante_target, split="val")
    m_te  = evaluar(y_te, y_pred_te, y_prob_te, nombre, estrategia_balanceo=estrategia, variante_target=variante_target, split="test")
    return clf, m_val, m_te


def entrenar_xgboost(
    X_tr, y_tr, X_val, y_val, X_te, y_te, w_tr, w_val,
    estrategia: str, variante_target: str = "ordinal_4clases",
    cfg: dict = None,
) -> Tuple:
    nombre = "XGBoost"
    seed   = PARAMETERS["SEED"]
    n_trials_efectivos = cfg["n_trials"]

    print(f"\n{'='*52}\n  Entrenando {nombre} — {estrategia}\n{'='*52}")

    _xgb_obj   = "binary:logistic" if variante_target == "binario" else "multi:softprob"
    _xgb_extra = {} if variante_target == "binario" else {"num_class": N_CLASES}
    hp_fijos = {
        "objective"   : _xgb_obj, **_xgb_extra,
        "tree_method" : "hist",
        "device"      : cfg["device_cuda"] if cfg["usar_gpu"] else "cpu",
        "random_state": seed, "n_jobs": cfg["n_jobs"], "verbosity": 0,
    }

    best_hp, kappa_val, n_trials_reg = _hp_previos(
        nombre, estrategia, variante_target, cfg)
    if best_hp is None:
        def obj(trial):
            p = {
                "n_estimators"    : trial.suggest_int("n_estimators", 200, 1000, step=100),
                "max_depth"       : trial.suggest_int("max_depth", 3, 8),
                "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                **hp_fijos,
            }
            m = xgb.XGBClassifier(**p)
            m.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_val, y_val)], verbose=False)
            y_p = m.predict(X_val)
            if hasattr(y_p, "ndim") and y_p.ndim > 1:
                y_p = y_p.argmax(axis=1)
            return cohen_kappa_score(y_val, y_p, weights="quadratic")

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(obj, n_trials=n_trials_efectivos, show_progress_bar=False)
        best_hp, kappa_val = study.best_params, study.best_value
        n_trials_reg = n_trials_efectivos
        print(f"  Mejor Kappa Val: {kappa_val:.4f} | {best_hp}")
        guardar_hiperparametros(nombre, estrategia, variante_target, best_hp,
                                hp_fijos=hp_fijos, espacio_busqueda=ESPACIO_XGBOOST,
                                n_trials=n_trials_reg, mejor_kappa_val=kappa_val,
                                sufijo=cfg.get("sufijo_hp", ""))

    clf = xgb.XGBClassifier(**best_hp, **hp_fijos)
    clf.fit(X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_val, y_val)], verbose=False)

    guardar_hiperparametros(
        nombre, estrategia, variante_target, best_hp,
        hp_fijos=hp_fijos, espacio_busqueda=ESPACIO_XGBOOST,
        config_entrenamiento={
            "usa_sample_weight" : True,
            "eval_set"          : "conjunto de validación",
            "early_stopping"    : "no (n_estimators lo fija Optuna)",
            "n_features"        : int(X_tr.shape[1]),
            "n_registros_train" : int(len(y_tr)),
        },
        clf=clf, n_trials=n_trials_reg, mejor_kappa_val=kappa_val,
        sufijo=cfg.get("sufijo_hp", ""),
    )

    y_pred_val = clf.predict(X_val)
    y_prob_val = clf.predict_proba(X_val)
    y_pred_te  = clf.predict(X_te)
    y_prob_te  = clf.predict_proba(X_te)

    m_val = evaluar(y_val, y_pred_val, y_prob_val, nombre, estrategia_balanceo=estrategia, variante_target=variante_target, split="val")
    m_te  = evaluar(y_te, y_pred_te, y_prob_te, nombre, estrategia_balanceo=estrategia, variante_target=variante_target, split="test")
    return clf, m_val, m_te


def entrenar_catboost(
    X_tr, y_tr, X_val, y_val, X_te, y_te, w_tr, w_val,
    estrategia: str, variante_target: str = "ordinal_4clases",
    cfg: dict = None,
) -> Tuple:
    nombre = "CatBoost"
    seed   = PARAMETERS["SEED"]
    n_trials_efectivos = cfg["n_trials"]

    print(f"\n{'='*52}\n  Entrenando {nombre} — {estrategia}\n{'='*52}")

    def prep_cat(X):
        X = X.copy()
        for col in VARS_CATEGORICAS:
            if col in X.columns:
                X[col] = X[col].fillna(-999).astype(int).astype(str)
        return X

    X_tr_c  = prep_cat(X_tr)
    X_val_c = prep_cat(X_val)
    X_te_c  = prep_cat(X_te)
    cat_idx = [i for i, c in enumerate(X_tr.columns) if c in VARS_CATEGORICAS]

    _cb_loss = "Logloss" if variante_target == "binario" else "MultiClass"
    hp_fijos = {
        "loss_function": _cb_loss,
        "random_seed"  : seed,
        "verbose"      : False,
        "task_type"    : "GPU" if cfg["usar_gpu"] else "CPU",
    }

    best_hp, kappa_val, n_trials_reg = _hp_previos(
        nombre, estrategia, variante_target, cfg)
    if best_hp is None:
        def obj(trial):
            p = {
                "iterations"         : trial.suggest_int("iterations", 300, 1000, step=100),
                "depth"              : trial.suggest_int("depth", 4, 8),
                "learning_rate"      : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg"        : trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "border_count"       : trial.suggest_int("border_count", 32, 128),
                "random_strength"    : trial.suggest_float("random_strength", 0.0, 10.0),
            }
            pool_tr  = Pool(X_tr_c, label=y_tr.values, weight=w_tr, cat_features=cat_idx)
            pool_val = Pool(X_val_c, label=y_val.values, cat_features=cat_idx)
            m = CatBoostClassifier(**p, **hp_fijos)
            m.fit(pool_tr, eval_set=pool_val)
            return cohen_kappa_score(y_val, m.predict(X_val_c).flatten(), weights="quadratic")

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(obj, n_trials=n_trials_efectivos, show_progress_bar=False)
        best_hp, kappa_val = study.best_params, study.best_value
        n_trials_reg = n_trials_efectivos
        print(f"  Mejor Kappa Val: {kappa_val:.4f} | {best_hp}")
        guardar_hiperparametros(nombre, estrategia, variante_target, best_hp,
                                hp_fijos=hp_fijos, espacio_busqueda=ESPACIO_CATBOOST,
                                n_trials=n_trials_reg, mejor_kappa_val=kappa_val,
                                sufijo=cfg.get("sufijo_hp", ""))

    pool_tr  = Pool(X_tr_c, label=y_tr.values, weight=w_tr, cat_features=cat_idx)
    pool_val = Pool(X_val_c, label=y_val.values, cat_features=cat_idx)
    clf = CatBoostClassifier(**best_hp, **hp_fijos)
    clf.fit(pool_tr, eval_set=pool_val)

    guardar_hiperparametros(
        nombre, estrategia, variante_target, best_hp,
        hp_fijos=hp_fijos, espacio_busqueda=ESPACIO_CATBOOST,
        config_entrenamiento={
            "usa_sample_weight"  : True,
            "eval_set"           : "conjunto de validación",
            "cat_features_idx"   : cat_idx,
            "cat_features_nombre": [c for c in VARS_CATEGORICAS if c in X_tr.columns],
            "nan_categoricas"    : "-999 como categoría explícita",
            "n_features"         : int(X_tr.shape[1]),
            "n_registros_train"  : int(len(y_tr)),
            "arboles_usados"     : int(getattr(clf, "tree_count_", 0) or 0),
        },
        clf=clf, n_trials=n_trials_reg, mejor_kappa_val=kappa_val,
        sufijo=cfg.get("sufijo_hp", ""),
    )

    y_pred_val = clf.predict(X_val_c).flatten()
    y_prob_val = clf.predict_proba(X_val_c)
    y_pred_te  = clf.predict(X_te_c).flatten()
    y_prob_te  = clf.predict_proba(X_te_c)

    m_val = evaluar(y_val, y_pred_val, y_prob_val, nombre, estrategia_balanceo=estrategia, variante_target=variante_target, split="val")
    m_te  = evaluar(y_te, y_pred_te, y_prob_te, nombre, estrategia_balanceo=estrategia, variante_target=variante_target, split="test")
    return clf, m_val, m_te


def entrenar_lightgbm(
    X_tr, y_tr, X_val, y_val, X_te, y_te, w_tr, w_val,
    pesos_clase: dict, estrategia: str,
    variante_target: str = "ordinal_4clases", cfg: dict = None,
) -> Tuple:
    nombre = "LightGBM"
    seed   = PARAMETERS["SEED"]
    n_trials_efectivos = cfg["n_trials"]

    print(f"\n{'='*52}\n  Entrenando {nombre} — {estrategia}\n{'='*52}")

    def prep_lgb(Xa, Xb, Xc):
        Xa, Xb, Xc = Xa.copy(), Xb.copy(), Xc.copy()
        for col in VARS_CATEGORICAS:
            if col not in Xa.columns:
                continue
            cats = sorted(set(Xa[col].dropna().tolist() +
                              Xb[col].dropna().tolist() +
                              Xc[col].dropna().tolist()))
            ct = pd.CategoricalDtype(categories=cats, ordered=False)
            Xa[col] = Xa[col].astype(ct)
            Xb[col] = Xb[col].astype(ct)
            Xc[col] = Xc[col].astype(ct)
        return Xa, Xb, Xc

    X_tr_l, X_val_l, X_te_l = prep_lgb(X_tr, X_val, X_te)

    _lgb_obj   = "binary" if variante_target == "binario" else "multiclass"
    _lgb_extra = {} if variante_target == "binario" else {"num_class": N_CLASES}
    hp_fijos = {
        "objective"   : _lgb_obj, **_lgb_extra,
        "random_state": seed,
        "n_jobs"      : cfg["n_jobs"], "verbose": -1,
        "device"      : cfg["device_cuda"] if cfg["usar_gpu"] else "cpu",
    }
    CONFIG_FIT = {
        "usa_sample_weight"    : True,
        "eval_set"             : "conjunto de validación",
        "early_stopping_rounds": 50,
        "log_evaluation"       : -1,
    }

    best_hp, kappa_val, n_trials_reg = _hp_previos(
        nombre, estrategia, variante_target, cfg)
    if best_hp is None:
        def obj(trial):
            p = {
                "n_estimators"     : trial.suggest_int("n_estimators", 200, 1000, step=100),
                "num_leaves"       : trial.suggest_int("num_leaves", 20, 150),
                "max_depth"        : trial.suggest_int("max_depth", 3, 8),
                "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            }
            m = lgb.LGBMClassifier(**p, **hp_fijos)
            try:
                m.fit(X_tr_l, y_tr, sample_weight=w_tr,
                      eval_set=[(X_val_l, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
            except lgb.basic.LightGBMError:
                raise optuna.exceptions.TrialPruned()
            return cohen_kappa_score(y_val, m.predict(X_val_l), weights="quadratic")

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(obj, n_trials=n_trials_efectivos, show_progress_bar=False)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            raise ValueError(
                f"LightGBM ({estrategia}): todos los trials de Optuna fallaron. "
                "Revisar datos o aumentar min_child_samples."
            )
        best_hp, kappa_val = study.best_params, study.best_value
        n_trials_reg = n_trials_efectivos
        print(f"  Mejor Kappa Val: {kappa_val:.4f} | {best_hp}")
        guardar_hiperparametros(nombre, estrategia, variante_target, best_hp,
                                hp_fijos=hp_fijos, espacio_busqueda=ESPACIO_LIGHTGBM,
                                config_entrenamiento=CONFIG_FIT,
                                n_trials=n_trials_reg, mejor_kappa_val=kappa_val,
                                sufijo=cfg.get("sufijo_hp", ""))

    clf = lgb.LGBMClassifier(**best_hp, **hp_fijos)
    clf.fit(X_tr_l, y_tr, sample_weight=w_tr,
            eval_set=[(X_val_l, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)])

    guardar_hiperparametros(
        nombre, estrategia, variante_target, best_hp,
        hp_fijos=hp_fijos, espacio_busqueda=ESPACIO_LIGHTGBM,
        config_entrenamiento={
            **CONFIG_FIT,
            "cat_features_nombre": [c for c in VARS_CATEGORICAS if c in X_tr.columns],
            "cat_features_dtype" : "pandas.CategoricalDtype compartido entre splits",
            "n_features"         : int(X_tr.shape[1]),
            "n_registros_train"  : int(len(y_tr)),
            "best_iteration"     : int(getattr(clf, "best_iteration_", 0) or 0),
            "n_arboles_ajustados": int(getattr(clf, "n_estimators_", 0) or 0),
        },
        clf=clf, n_trials=n_trials_reg, mejor_kappa_val=kappa_val,
        sufijo=cfg.get("sufijo_hp", ""),
    )

    y_pred_val = clf.predict(X_val_l)
    y_prob_val = clf.predict_proba(X_val_l)
    y_pred_te  = clf.predict(X_te_l)
    y_prob_te  = clf.predict_proba(X_te_l)

    m_val = evaluar(y_val, y_pred_val, y_prob_val, nombre, estrategia_balanceo=estrategia, variante_target=variante_target, split="val")
    m_te  = evaluar(y_te, y_pred_te, y_prob_te, nombre, estrategia_balanceo=estrategia, variante_target=variante_target, split="test")
    return clf, m_val, m_te


def _pesos_clase_tabnet(y_tr, estrategia: str, pesos_clase: Optional[Dict],
                        dispositivo: str):
    """
    Función de pérdida de TabNet para la estrategia de balanceo indicada.

    En la estrategia ``pesos_clase`` el desbalance se corrige dentro de la
    pérdida, con una entropía cruzada ponderada por la frecuencia inversa de
    cada clase, y no reponderando el muestreo. Así el brazo con pesos difiere
    realmente del baseline: el parámetro ``weights`` de ``fit`` solo controla
    el ``WeightedRandomSampler``, de modo que dejarlo activo en las tres
    estrategias las volvería equivalentes.

    Retorna ``(loss_fn, vector_de_pesos)``; ``(None, None)`` cuando no se
    aplican pesos y corresponde la pérdida por defecto.
    """
    if estrategia != "pesos_clase":
        return None, None

    clases = np.unique(y_tr)
    n_clases = len(clases)
    # Se reutiliza el vector calculado sobre el conjunto de entrenamiento para
    # el resto de los modelos; si no corresponde al número de clases de esta
    # variante del target (por ejemplo en la variante binaria), se recalcula
    # con la misma fórmula de frecuencia inversa.
    if pesos_clase and len(pesos_clase) == n_clases:
        vector = [float(pesos_clase[c]) for c in sorted(pesos_clase)]
    else:
        vector = [len(y_tr) / (n_clases * int((y_tr == c).sum())) for c in clases]

    tensor = torch.tensor(vector, dtype=torch.float32, device=dispositivo)
    return torch.nn.CrossEntropyLoss(weight=tensor), vector


def entrenar_tabnet(
    X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_te,
    estrategia: str, cat_idxs: list, cat_dims: list,
    pesos_clase: Optional[Dict] = None,
    variante_target: str = "ordinal_4clases", cfg: dict = None,
) -> Tuple:
    nombre = "TabNet"
    seed   = PARAMETERS["SEED"]

    print(f"\n{'='*52}\n  Entrenando {nombre} — {estrategia}\n{'='*52}")
    print(f"  Dispositivo: {cfg['dispositivo_tn']}")

    loss_fn, vector_pesos = _pesos_clase_tabnet(
        y_tr, estrategia, pesos_clase, cfg["dispositivo_tn"])
    if vector_pesos is not None:
        print("  Pérdida ponderada por clase: " +
              ", ".join(f"{w:.4f}" for w in vector_pesos))

    n_trials_efectivos = cfg.get("n_trials_tabnet", cfg["n_trials"])
    hp_fijos = {
        "optimizer_fn"    : "torch.optim.Adam",
        "scheduler_fn"    : "torch.optim.lr_scheduler.StepLR",
        "scheduler_params": {"step_size": 10, "gamma": 0.9},
        "cat_idxs"        : cat_idxs,
        "cat_dims"        : cat_dims,
        "cat_emb_dim"     : 3,
        "verbose"         : 0,
        "device_name"     : cfg["dispositivo_tn"],
        "seed"            : seed,
    }
    CONFIG_FIT = {
        "max_epochs"        : PARAMETERS["EPOCAS_TABNET"],
        "patience"          : PARAMETERS["PACIENCIA_TABNET"],
        "batch_size"        : 1024,
        "virtual_batch_size": 128,
        "eval_metric"       : ["balanced_accuracy"],
        # weights=0 desactiva el WeightedRandomSampler: el muestreo es uniforme
        # en las tres estrategias y el balanceo, cuando corresponde, se aplica
        # en la función de pérdida.
        "weights"           : 0,
        "eval_set"          : "conjunto de validación",
        "usa_sample_weight" : False,
        "loss_fn"           : ("CrossEntropyLoss ponderada por frecuencia "
                               "inversa de clase" if loss_fn is not None
                               else "cross_entropy (por defecto)"),
        "pesos_clase"       : vector_pesos,
        "max_epochs_optuna" : PARAMETERS["EPOCAS_TABNET_OPTUNA"],
        "patience_optuna"   : PARAMETERS["PACIENCIA_TABNET_OPTUNA"],
    }

    best_hp, kappa_val, n_trials_reg = _hp_previos(
        nombre, estrategia, variante_target, cfg)
    if best_hp is None:
        def obj(trial):
            p = {
                "n_d"          : trial.suggest_int("n_d", 8, 64, step=8),
                "n_a"          : trial.suggest_int("n_a", 8, 64, step=8),
                "n_steps"      : trial.suggest_int("n_steps", 3, 7),
                "gamma"        : trial.suggest_float("gamma", 1.0, 2.0),
                "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
                "momentum"     : trial.suggest_float("momentum", 0.01, 0.4),
                "mask_type"    : trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
            }
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            m = TabNetClassifier(
                **p,
                optimizer_fn=torch.optim.Adam,
                optimizer_params={"lr": lr},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                scheduler_params={"step_size": 10, "gamma": 0.9},
                cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=3,
                verbose=0, device_name=cfg["dispositivo_tn"], seed=seed,
            )
            m.fit(
                X_tr_sc.astype(np.float32), y_tr,
                eval_set=[(X_val_sc.astype(np.float32), y_val)],
                eval_metric=["balanced_accuracy"],
                max_epochs=PARAMETERS["EPOCAS_TABNET_OPTUNA"],
                patience=PARAMETERS["PACIENCIA_TABNET_OPTUNA"],
                batch_size=1024, virtual_batch_size=128,
                weights=0, loss_fn=loss_fn,
            )
            return cohen_kappa_score(y_val,
                                     m.predict(X_val_sc.astype(np.float32)),
                                     weights="quadratic")

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(obj, n_trials=n_trials_efectivos, show_progress_bar=False)
        best_hp, kappa_val = study.best_params, study.best_value
        n_trials_reg = n_trials_efectivos
        print(f"  Mejor Kappa Val: {kappa_val:.4f} | {best_hp}")
        guardar_hiperparametros(nombre, estrategia, variante_target, best_hp,
                                hp_fijos=hp_fijos, espacio_busqueda=ESPACIO_TABNET,
                                config_entrenamiento=CONFIG_FIT,
                                n_trials=n_trials_reg, mejor_kappa_val=kappa_val,
                                sufijo=cfg.get("sufijo_hp", ""))

    best_hp = dict(best_hp)
    lr_opt  = best_hp.pop("lr", 1e-3)
    clf = TabNetClassifier(
        **best_hp,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": lr_opt},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=3,
        verbose=0, device_name=cfg["dispositivo_tn"], seed=seed,
    )
    clf.fit(
        X_tr_sc.astype(np.float32), y_tr,
        eval_set=[(X_val_sc.astype(np.float32), y_val)],
        eval_metric=["balanced_accuracy"],
        max_epochs=PARAMETERS["EPOCAS_TABNET"],
        patience=PARAMETERS["PACIENCIA_TABNET"],
        batch_size=1024, virtual_batch_size=128,
        weights=0, loss_fn=loss_fn,
    )
    best_hp["lr"] = lr_opt

    try:
        epocas = len(clf.history["loss"])
    except Exception:                                            # noqa: BLE001
        epocas = None
    guardar_hiperparametros(
        nombre, estrategia, variante_target, best_hp,
        hp_fijos={**hp_fijos, "optimizer_params": {"lr": lr_opt}},
        espacio_busqueda=ESPACIO_TABNET,
        config_entrenamiento={
            **CONFIG_FIT,
            "n_features"       : int(X_tr_sc.shape[1]),
            "n_registros_train": int(len(y_tr)),
            "epocas_entrenadas": epocas,
            "entrada"          : "X normalizada con MinMaxScaler",
        },
        clf=clf, n_trials=n_trials_reg, mejor_kappa_val=kappa_val,
        sufijo=cfg.get("sufijo_hp", ""),
    )

    y_pred_val = clf.predict(X_val_sc.astype(np.float32))
    y_prob_val = clf.predict_proba(X_val_sc.astype(np.float32))
    y_pred_te  = clf.predict(X_te_sc.astype(np.float32))
    y_prob_te  = clf.predict_proba(X_te_sc.astype(np.float32))

    # TabNet no admite sample_weight por registro: el factor de expansión
    # muestral X_020 no interviene en su entrenamiento, a diferencia del resto
    # de los modelos. Queda registrado como limitación del diseño.
    print("  Nota: TabNet no admite sample_weight por registro; el factor de "
          "expansión muestral no interviene en su ajuste.")
    m_val = evaluar(y_val, y_pred_val, y_prob_val, nombre, estrategia_balanceo=estrategia, variante_target=variante_target, split="val")
    m_te  = evaluar(y_te, y_pred_te, y_prob_te, nombre, estrategia_balanceo=estrategia, variante_target=variante_target, split="test")
    return clf, m_val, m_te
