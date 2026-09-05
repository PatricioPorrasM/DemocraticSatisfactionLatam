import os
import glob
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

# ====================================================
# MODO DE EJECUCIÓN — el único interruptor que hay que mover
#
#   "real" : corrida definitiva. Dataset completo, presupuestos de Optuna y de
#            bootstrap completos, pliegues temporales activos. Es la que produce
#            las cifras que van al documento. Coste: horas en GPU.
#   "humo" : prueba de humo. Recorre TODO el flujo de los seis notebooks —
#            incluidos los pliegues temporales, el bootstrap, SHAP, LIME y ALE —
#            con la muestra reducida y presupuestos mínimos, para verificar que
#            no hay errores antes de lanzar la corrida real. Sus cifras NO son
#            publicables.
#            El NB01 lee y armoniza las olas de Latinobarómetro en los dos
#            modos, así que su duración no cambia; el ahorro está en los
#            notebooks 02 a 06, que pasan de horas a minutos.
#
# Cómo se cambia, según cómo se ejecute el proyecto:
#
#   - Notebook a notebook (Jupyter Lab, VS Code): editar MODO abajo.
#     Después de editarlo hay que REINICIAR EL KERNEL, porque Python conserva
#     en memoria el módulo ya importado y no volvería a leer el archivo.
#
#   - Corrida completa con papermill: no hace falta tocar el archivo, la
#     variable de entorno tiene prioridad sobre MODO.
#         MODO_EJECUCION=humo bash run_all.sh
#
# Todo lo que depende del modo está en PERFILES_EJECUCION; lo que no depende
# está en _PARAMETERS_COMUNES. Ningún notebook define banderas propias.
# ====================================================

# ┌───────────────────────────────────────────────────────────────────────┐
# │  ESTA ES LA LÍNEA QUE SE EDITA:  "real"  o  "humo"                    │
MODO = "humo"
# └───────────────────────────────────────────────────────────────────────┘

# La variable de entorno, si está definida, tiene prioridad sobre MODO: es la
# que usa run_all.sh para no tener que editar el archivo en el servidor.
MODO_EJECUCION = os.environ.get("MODO_EJECUCION", MODO).strip().lower()


# ── Parámetros que NO cambian entre modos ───────────────────────────────────
_PARAMETERS_COMUNES = {
    # Semilla única del proyecto: numpy, torch, Optuna, SMOTE-NC y bootstrap.
    "SEED": 42,
    # Rango de años admitido al armonizar Latinobarómetro con V-Dem.
    "YEAR_START": 1995,
    "YEAR_END": 2024,

    # ── Hardware y paralelismo (NB02) ────────────────────────────────────────
    # USAR_GPU: True activa 'device=gpu' en XGBoost y LightGBM, 'task_type=GPU'
    # en CatBoost y 'cuda' en TabNet. Si no hay GPU visible, hw_cfg() degrada a
    # CPU automáticamente, así que dejarlo en True es seguro en cualquier
    # máquina.
    "USAR_GPU": True,
    # N_JOBS: hilos para los modelos que corren en CPU. -1 usa todos los núcleos.
    "N_JOBS": -1,

    # ── Búsqueda de hiperparámetros con Optuna (NB02 §14–§15) ────────────────
    # EJECUTAR_BUSQUEDA_HP: True corre la búsqueda TPE; False reutiliza los
    # hiperparámetros ya registrados en models/hp_*.json (útil para repetir
    # solo la evaluación sin volver a optimizar). Se deja en True en los dos
    # modos: la prueba de humo también debe ejercitar la búsqueda.
    "EJECUTAR_BUSQUEDA_HP": True,

    # ── Validación temporal en pliegues históricos (NB02 §19) ───────────────
    # Estrategia de balanceo con la que se corren los pliegues. Se usa una sola
    # porque el objetivo es medir estabilidad temporal, no repetir E1.
    "ESTRATEGIA_FOLDS": "pesos_clase",

    # ── Evaluación comparativa (NB03) ────────────────────────────────────────
    # Métrica que ordena los modelos y guía la selección. El kappa cuadrático
    # penaliza los errores en proporción a la distancia ordinal.
    "METRICA_PRINCIPAL": "kappa_cuadratico",
    # Conjunto donde se ELIGE la configuración principal. Debe ser 'val': el
    # conjunto de prueba se reserva para reportar, no para seleccionar.
    "CONJUNTO_SELECCION": "val",
    # True dibuja una matriz de confusión por cada modelo × estrategia (15);
    # False solo la del modelo principal.
    "CONFUSION_TODOS_MODELOS": True,
    # True añade, junto a cada métrica, su versión ponderada por el factor de
    # expansión muestral X_020.
    "REPORTAR_PONDERADAS": True,
    "NIVEL_CONFIANZA": 0.95,
    # Unidad que se remuestrea: los registros de un mismo país-año comparten
    # los indicadores de V-Dem, así que se remuestrean clústeres completos.
    # 'pais_anio' (32 clústeres en test) o 'pais' (16, más conservador).
    "NIVEL_CLUSTER": "pais_anio",
    # Lista y orden de los modelos en todas las tablas y figuras comparativas.
    "MODELOS": ["OLO", "XGBoost", "CatBoost", "LightGBM", "TabNet"],

    # ── Explicabilidad (NB04) ────────────────────────────────────────────────
    # MODELO_XAI: 'auto' toma la configuración principal que seleccionó el NB03
    # (results/modelo_xai_seleccionado.json); un nombre concreto la fija a mano.
    "MODELO_XAI": "auto",
    # Conjunto sobre el que se explican las predicciones.
    "SPLIT_REFERENCIA_XAI": "test",
    # SHAP para TabNet exige KernelExplainer sobre la red completa (horas).
    # False usa en su lugar las máscaras de atención propias del modelo.
    "SHAP_PARA_TABNET": False,
    # True recalcula SHAP incluso si existe el parquet guardado. El notebook
    # además invalida el caché por sí solo cuando el pipeline es más nuevo que
    # el parquet, así que basta dejarlo en False.
    "FORZAR_RECALCULO_SHAP": False,
    # Variables mostradas en los gráficos de importancia.
    "TOP_N_SHAP": 20,
    # True añade el gráfico de las variables que más pesan en los errores graves.
    "LIME_SOBRE_ERRORES": True,
    "NIVEL_CLUSTER_XAI": "pais_anio",
    # Modelos con rendimiento estadísticamente indistinguible cuyos rankings se
    # comparan entre sí (diagnóstico de identificabilidad del ranking).
    "MODELOS_CONCORDANCIA": ["CatBoost", "XGBoost", "LightGBM"],
    # Variables por bloque temático para las que se calcula la curva ALE.
    "VARS_ALE_POR_BLOQUE": 2,

    # ── Contraste teórico (NB06) ─────────────────────────────────────────────
    # Variables del ranking empírico sobre las que se calcula el porcentaje de
    # convergencia con el bloque que cada teoría predice como dominante.
    "TOP_N_CONTRASTE": 10,
    # Variables incluidas en la tabla detallada de convergencias y divergencias.
    "TOP_N_TABLA_CONVERGENCIAS": 20,
    # Variables consideradas en el mapa de calor bloque × teoría.
    "TOP_N_HEATMAP": 10,
}


# ── Parámetros que SÍ cambian entre modos ───────────────────────────────────
#
# Las dos columnas tienen exactamente las mismas claves: el perfil de humo
# recorta el volumen de cada etapa, nunca desactiva una etapa entera, para que
# la prueba ejercite todas las rutas de código que usará la corrida real.
PERFILES_EJECUCION = {

    "real": {
        # Carga de datos (NB01/NB02): las olas completas.
        "LOAD_SAMPLE": False,
        "MIN_NUMBER_RECORDS": 8,     # solo afectan a la muestra de inspección
        "MAX_NUMBER_RECORDS": 12,    # que genera el NB01; sin efecto aquí
        # Optuna: los presupuestos difieren porque el costo por ensayo cambia
        # mucho entre familias de modelos.
        "N_TRIALS_OPTUNA": 50,       # árboles de gradiente
        "N_TRIALS_OLO": 20,          # mord.LogisticIT: L-BFGS-B en Python puro
        "N_TRIALS_TABNET": 20,       # cada ensayo entrena una red completa
        # Pliegues temporales históricos.
        "EJECUTAR_FOLDS_TEMPORALES": True,
        "N_TRIALS_FOLDS": 15,
        # Bootstrap de clústeres.
        "N_BOOTSTRAP": 1000,
        "N_BOOTSTRAP_SHAP": 1000,
        # TabNet: épocas y paciencia del early stopping. Es el modelo más
        # costoso por ajuste, así que su presupuesto de épocas también depende
        # del modo.
        "EPOCAS_TABNET": 200,
        "PACIENCIA_TABNET": 20,
        "EPOCAS_TABNET_OPTUNA": 100,     # por ensayo de la búsqueda
        "PACIENCIA_TABNET_OPTUNA": 15,
        # SHAP y LIME.
        "N_MUESTRAS_SHAP_OLO": 500,
        "CASOS_LIME_REPRESENTATIVOS": 100,  # estratificados por clase × subregión
        "CASOS_LIME_ERRORES": 50,           # mayor distancia ordinal |ŷ - y|
        "CASOS_LIME_DISCORDANTES": 50,      # poliarquía alta y satisfacción baja
    },

    "humo": {
        # Muestra reducida: el NB01 la genera con MIN/MAX registros por ola y
        # país. Con 60–80 por celda quedan ~30.000 registros, suficientes para
        # que cada ola de validación tenga las cuatro clases representadas y el
        # kappa y el AUROC se puedan calcular sin degenerar.
        "LOAD_SAMPLE": True,
        "MIN_NUMBER_RECORDS": 60,
        "MAX_NUMBER_RECORDS": 80,
        # Optuna: dos ensayos bastan para recorrer el bucle de búsqueda, el
        # guardado del registro y el reajuste final.
        "N_TRIALS_OPTUNA": 2,
        "N_TRIALS_OLO": 2,
        "N_TRIALS_TABNET": 2,
        # Los pliegues se ejecutan también en humo: es justo el código nuevo
        # que más conviene probar antes de la corrida real.
        "EJECUTAR_FOLDS_TEMPORALES": True,
        "N_TRIALS_FOLDS": 1,
        # Bootstrap: 50 réplicas dan intervalos inservibles pero recorren todo
        # el camino de remuestreo, agregación y guardado de tablas.
        "N_BOOTSTRAP": 50,
        "N_BOOTSTRAP_SHAP": 50,
        # TabNet: sin recortar las épocas, el ajuste de la red domina el
        # tiempo de la prueba y la deja en horas en lugar de minutos. Con este
        # presupuesto la red no converge —no es el objetivo— pero se ejercitan
        # el bucle de entrenamiento, el early stopping y la pérdida ponderada.
        "EPOCAS_TABNET": 12,
        "PACIENCIA_TABNET": 4,
        "EPOCAS_TABNET_OPTUNA": 8,
        "PACIENCIA_TABNET_OPTUNA": 3,
        # SHAP y LIME: lo justo para que cada grupo tenga casos.
        "N_MUESTRAS_SHAP_OLO": 50,
        "CASOS_LIME_REPRESENTATIVOS": 8,
        "CASOS_LIME_ERRORES": 4,
        "CASOS_LIME_DISCORDANTES": 4,
    },
}

if MODO_EJECUCION not in PERFILES_EJECUCION:
    raise ValueError(
        f"MODO_EJECUCION={MODO_EJECUCION!r} no reconocido. "
        f"Valores admitidos: {sorted(PERFILES_EJECUCION)}."
    )

# Las dos columnas deben cubrir las mismas claves: si una se añade en un perfil
# y se olvida en el otro, el modo correspondiente fallaría a mitad de la corrida.
_claves = {m: set(p) for m, p in PERFILES_EJECUCION.items()}
if _claves["real"] != _claves["humo"]:
    raise ValueError(
        "Los perfiles de ejecución no tienen las mismas claves. "
        f"Solo en 'real': {sorted(_claves['real'] - _claves['humo'])}. "
        f"Solo en 'humo': {sorted(_claves['humo'] - _claves['real'])}."
    )


# ====================================================
# PARAMETERS — punto único de control de la ejecución
#
# Unión de los parámetros comunes y del perfil activo. Es el diccionario que
# leen los notebooks; no se edita a mano, se edita MODO_EJECUCION o el perfil
# correspondiente.
# ====================================================

PARAMETERS = {
    "MODO_EJECUCION": MODO_EJECUCION,
    **_PARAMETERS_COMUNES,
    **PERFILES_EJECUCION[MODO_EJECUCION],
}


def es_prueba_de_humo() -> bool:
    """True si la corrida activa es una prueba de humo y no la definitiva."""
    return PARAMETERS["MODO_EJECUCION"] == "humo"


def resumen_modo() -> None:
    """
    Imprime el modo activo y los parámetros que dependen de él.

    Se llama al inicio de cada notebook. En modo de humo el aviso es
    deliberadamente llamativo, para que ninguna cifra de una prueba termine
    citada como resultado.
    """
    perfil = PERFILES_EJECUCION[PARAMETERS["MODO_EJECUCION"]]
    if es_prueba_de_humo():
        print("#" * 72)
        print("#  MODO PRUEBA DE HUMO — muestra reducida y presupuestos mínimos")
        print("#  Las cifras de esta corrida NO son publicables.")
        print("#  Para la corrida definitiva: MODO_EJECUCION = \"real\" en")
        print("#  utils/config.py, o MODO_EJECUCION=real en el entorno.")
        print("#" * 72)
    else:
        print("=" * 72)
        print("  MODO CORRIDA REAL — dataset completo y presupuestos completos")
        print("=" * 72)
    print("Parámetros que dependen del modo:")
    for clave in sorted(perfil):
        print(f"  {clave:<28}: {perfil[clave]}")


def hw_cfg(n_trials=None, sufijo_hp=""):
    """Configuración de hardware y presupuesto que reciben las funciones
    ``entrenar_*`` de :mod:`utils.models`.

    Degrada a CPU si ``PARAMETERS["USAR_GPU"]`` es True pero no hay GPU
    visible, de modo que el mismo notebook corre en servidor y en portátil.

    Parameters
    ----------
    n_trials : int, optional
        Sobrescribe el presupuesto de Optuna de los árboles de gradiente. Se
        usa en los pliegues temporales, que corren con menos ensayos.
    sufijo_hp : str
        Sufijo de los archivos ``models/hp_*.json``. Vacío para el split
        principal; ``"_foldN"`` para los pliegues históricos, de modo que sus
        registros no sobrescriban los del modelo final.
    """
    try:
        import torch
        gpu_disponible = torch.cuda.is_available()
    except Exception:
        gpu_disponible = False

    usar_gpu = bool(PARAMETERS["USAR_GPU"]) and gpu_disponible
    trials_arboles = PARAMETERS["N_TRIALS_OPTUNA"] if n_trials is None else n_trials

    return {
        "usar_gpu"      : usar_gpu,
        # 'gpu' es el alias que aceptan tanto XGBoost (parámetro device) como
        # LightGBM (device_type) para "el dispositivo GPU por defecto".
        "device_cuda"   : "gpu" if usar_gpu else "cpu",
        "dispositivo_tn": "cuda" if usar_gpu else "cpu",
        "n_jobs"        : PARAMETERS["N_JOBS"],
        "ejecutar_hp"   : PARAMETERS["EJECUTAR_BUSQUEDA_HP"],
        "n_trials"      : trials_arboles,
        "n_trials_olo"  : PARAMETERS["N_TRIALS_OLO"] if n_trials is None else n_trials,
        "n_trials_tabnet": PARAMETERS["N_TRIALS_TABNET"] if n_trials is None else n_trials,
        "sufijo_hp"     : sufijo_hp,
    }

# ====================================================
# PATHS
# ====================================================

PATHS = {
    "FOLDER_RAW_ZIP":           Path("..") / "data" / "raw_zip",
    "FOLDER_RAW_LB":            Path("..") / "data" / "raw_latinobarometro",
    "FOLDER_RAW_VDEM":          Path("..") / "data" / "raw_v-dem",
    "FOLDER_BASE":              Path("..") / "data" / "base",
    "FOLDER_PROCS":             Path("..") / "data" / "processed",
    "FOLDER_MODELS":            Path("..") / "models",
    "FOLDER_RESULTS":           Path("..") / "results",
    "FOLDER_RESULTS_FIGURES":   Path("..") / "results" / "figures",
    "FOLDER_RESULTS_METRICS":   Path("..") / "results" / "metrics",
    "FOLDER_RESULTS_TABLES":    Path("..") / "results" / "tables",
    "FOLDER_RESULTS_SHAP":      Path("..") / "results" / "shap",
    "FILE_RAW_VDEM":            Path("..") / "data" / "raw_v-dem" / "V-Dem-CY-Core-v16.csv",
    "FILE_BASE_VDEM":           Path("..") / "data" / "base" / "v-dem.csv",
    "FILE_BASE_LB":             Path("..") / "data" / "base" / "latinobarometro.csv",
    "FILE_LB_VAR_MAPPING":      Path("..") / "data" / "variables" / "latinobarometro_variable_mapping.csv",
    "FILE_VAR_SELECTION":       Path("..") / "data" / "variables" / "variables_selection.csv",
    "FILE_FREQUENCY_TABLE":     Path("..") / "data" / "base" / "lb_frecuencia_valores_por_ola.csv",
    "FILE_BASE_LB_SAMPLE":      Path("..") / "data" / "base" / "latinobarometro_muestra.csv",
    "FILE_RESULTS_MODEL_CSV":    Path("..") / "results" / "resultados_modelos.csv",
    "FILE_RESULTS_MODEL_PARQUET":Path("..") / "results" / "resultados_modelos.parquet",
}

# ====================================================
# TEMAS Y PALETAS
# ====================================================

THEME = {
    "models": {
        "OLO":      "#4C78A8",
        "CatBoost": "#F58518",
        "XGBoost":  "#54A24B",
        "LightGBM": "#E45756",
        "TabNet":   "#72B7B2",
    },
    "target": {
        0: "#DC2626",
        1: "#F58518",
        2: "#54A24B",
        3: "#4C78A8",
    },
    "semantic": {
        "success":    "#2CA02C",
        "warning":    "#FFB000",
        "danger":     "#D62728",
        "grid":       "#E8E8E8",
        "text":       "#303030",
        "background": "#FFFFFF",
    },
    "blocks": {
        "Confianza institucional":           "#1E3A5F",
        "Evaluación económica":              "#0D9488",
        "Percepción política":               "#2E74B5",
        "Corrupción y seguridad":            "#DC2626",
        "Características sociodemográficas": "#78716C",
        "Contexto democrático":              "#7C3AED",
    },
}

PALETTES = {
    "models":      list(THEME["models"].values()),
    "categorical": sns.color_palette("colorblind"),
    "sequential":  sns.color_palette("viridis"),
    "diverging":   sns.color_palette("coolwarm"),
}


def setup_plots():
    sns.set_theme(style="whitegrid", palette=PALETTES["models"])
    mpl.rcParams.update({
        "figure.figsize":  (10, 6),
        "figure.dpi":      150,
        "savefig.dpi":     300,
        "font.family":     "DejaVu Sans",
        "font.size":       12,
        "axes.titlesize":  16,
        "axes.labelsize":  13,
        "axes.edgecolor":  "#555555",
        "axes.prop_cycle": plt.cycler(color=PALETTES["models"]),
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "grid.color":      THEME["semantic"]["grid"],
        "grid.linestyle":  "--",
        "legend.frameon":  False,
        "lines.linewidth": 2.5,
        "lines.markersize":7,
        "savefig.bbox":    "tight",
    })


# ====================================================
# COLUMNAS CLAVE
# ====================================================

COL_TARGET = "target"
COL_AÑO    = "año"
COL_PAIS   = "pais_nombre"
COL_ISO3   = "pais_iso3"
COL_PESO   = "X_020"

NSNR = [-1, -2, -3, -4, -5, -6, -7, -8]

# ====================================================
# SPLIT ÚNICO — Validación temporal
#
# Diseño: un único split temporal que separa entrenamiento,
# validación y prueba en olas de encuesta no superpuestas.
#
# Train  : 1995–2018 (21 olas). Venezuela incluida hasta 2017
#          inclusive (AÑO_CORTE_VEN). Nicaragua incluida hasta 2018.
# Val    : 2020 (1 ola). Venezuela y Nicaragua excluidas
#          (PAISES_EXCLUIR_EVAL) para consistencia con el test.
# Test   : 2023–2024 (2 olas). Venezuela y Nicaragua excluidas
#          (Venezuela: datos no representativos post-2017;
#           Nicaragua: sin cobertura en 2023-2024).
#
# Justificación del año de validación (2020):
#   Test KS entre la distribución del target en 2020 y en
#   el test (2023+2024): estadístico=0.043, p=0.787.
#   Las distribuciones son estadísticamente indistinguibles,
#   lo que garantiza que Optuna calibra hiperparámetros sobre
#   un contexto representativo del test.
# ====================================================

SPLIT = {
    "train": [1995, 1996, 1997, 1998, 2000, 2001, 2002, 2003, 2004, 2005,
              2006, 2007, 2008, 2009, 2010, 2011, 2013, 2015, 2016, 2017, 2018],
    "val":   [2020],
    "test":  [2023, 2024],
}

# ====================================================
# PLIEGUES TEMPORALES HISTÓRICOS — validación de la estabilidad del split
#
# El split anterior es único, así que su métrica no lleva asociada una
# desviación entre periodos: no se sabe si el rendimiento observado en
# 2023-2024 es típico o propio de esa coyuntura. Para estimarlo se replica el
# mismo esquema hacia atrás en tres cortes con ventana de entrenamiento
# expansiva (origen fijo en 1995), cada uno con una ola de validación y dos de
# prueba inmediatamente posteriores, sin solapamiento entre conjuntos.
#
# Los pliegues NO reemplazan al split principal ni cambian el modelo que se
# reporta: sirven para acotar la variabilidad temporal de kappa cuadrático y
# del MAE ordinal, que se informa como media ± desviación estándar.
#
# Las reglas de exclusión son idénticas en los cuatro cortes (corte de
# Venezuela en AÑO_CORTE_VEN y exclusión de PAISES_EXCLUIR_EVAL en validación
# y prueba), porque se aplican sobre el dataframe antes de construir el split.
# Nicaragua sí tiene datos en las olas históricas, pero se excluye igual para
# que los cuatro cortes midan el mismo dominio de países.
# ====================================================

SPLITS_TEMPORALES = {
    "fold1": {
        "train": [1995, 1996, 1997, 1998, 2000, 2001, 2002, 2003, 2004,
                  2005, 2006, 2007],
        "val":   [2008],
        "test":  [2009, 2010],
    },
    "fold2": {
        "train": [1995, 1996, 1997, 1998, 2000, 2001, 2002, 2003, 2004,
                  2005, 2006, 2007, 2008, 2009, 2010],
        "val":   [2011],
        "test":  [2013, 2015],
    },
    "fold3": {
        "train": [1995, 1996, 1997, 1998, 2000, 2001, 2002, 2003, 2004,
                  2005, 2006, 2007, 2008, 2009, 2010, 2011, 2013, 2015],
        "val":   [2016],
        "test":  [2017, 2018],
    },
    # Corte definitivo: el mismo SPLIT que entrena los modelos reportados.
    # Se incluye para que la tabla de estabilidad lo muestre junto a los demás.
    "final": SPLIT,
}

# ====================================================
# TRATAMIENTO DE CASOS ESPECIALES: VENEZUELA Y NICARAGUA
#
# VENEZUELA
#   Situación: el régimen de Maduro instauró en 2017 la Asamblea
#   Nacional Constituyente, que disolvió la separación de poderes
#   y marcó el colapso del Estado de derecho reconocido
#   internacionalmente (V-Dem: poliarquía cae de 0.281 en 2016 a
#   0.233 en 2017, y a 0.196 en 2024).
#   Desde 2018 las encuestas de Latinobarómetro en Venezuela
#   muestran un patrón estadísticamente anómalo: en 2018 el 73.7%
#   declara estar "Muy satisfecho", y en 2024 el 54.5%. Este
#   sesgo de respuesta en regímenes autoritarios está documentado
#   (Guriev y Treisman, 2019; Norris, 2011).
#   Criterio de corte: AÑO_CORTE_VEN = 2017. Los registros de
#   Venezuela posteriores a 2017 se eliminan antes del split.
#   Test KS Venezuela 2017 vs. otros países: p=0.163 (no sig.).
#   Test KS Venezuela 2018 vs. otros países: p<0.001 (anomalía).
#
# NICARAGUA
#   Situación: Nicaragua no tiene datos de Latinobarómetro en
#   los años de prueba (2023 y 2024). Por tanto, queda excluida
#   del test por falta de datos, no por exclusión activa.
#   Para consistencia metodológica, también se excluye de la
#   validación (2020): la validación debe representar el mismo
#   dominio que el test, y Nicaragua no forma parte de ese dominio.
#   Nicaragua se mantiene en entrenamiento (1996–2018).
#   Nicaragua en 2020: v2x_polyarchy=0.215 (vs. promedio 0.610),
#   contexto político anómalo bajo el gobierno de Ortega.
# ====================================================

AÑO_CORTE_VEN       = 2017
PAISES_EXCLUIR_EVAL = ["Venezuela", "Nicaragua"]   # excluidos de val y test

# ====================================================
# SUBREGIONES Y PAÍSES
# ====================================================

SUBREGIONES = {
    "Cono Sur":        ["Argentina", "Chile", "Uruguay", "Paraguay"],
    "Región Andina":   ["Bolivia", "Colombia", "Ecuador", "Perú"],
    "Brasil":          ["Brasil"],
    "Centroamérica":   ["Costa Rica", "El Salvador", "Guatemala", "Honduras", "Panamá"],
    "México y Caribe": ["México", "República Dominicana"],
}

# ====================================================
# MAPEOS
# ====================================================

MAPEO_NUMINVES = {16: 2011, 17: 2013, 18: 2015, 23: 2023, 24: 2024}

MAPEO_PAIS_ISO3 = {
    "Argentina": "ARG", "Bolivia": "BOL", "Brasil": "BRA",
    "Chile": "CHL", "Colombia": "COL", "Costa Rica": "CRI",
    "República Dominicana": "DOM", "Ecuador": "ECU",
    "El Salvador": "SLV", "Guatemala": "GTM", "Honduras": "HND",
    "México": "MEX", "Nicaragua": "NIC", "Panamá": "PAN",
    "Paraguay": "PRY", "Perú": "PER", "Uruguay": "URY",
    "Venezuela": "VEN",
}

# ====================================================
# VARIABLES
# ====================================================

VARS_EXCLUIR_LB = [
    # ── Exclusiones por incompatibilidad técnica ──────────────────────────────
    "C_001_031",      # ruptura de codificación en 2018; incomparable entre olas
    "A_003_021",      # ausente en el conjunto de test (2023, 2024)
    "D_001_061",      # ausente en los tres conjuntos de evaluación
    "D_001_131",      # ausente en el conjunto de test
    "X_004",          # 627 categorías; 94% categorías nuevas en test; sin señal
    "S_700",          # sin señal en ningún período; alta cardinalidad
    # ── Exclusiones por señal predictiva baja (|r_Spearman| < 0.05) ──────────
    "H_002_101",      # Confianza Iglesia Católica: |r|=0.047; sin justificación política
    "C_003_003_011",  # Preocupación desempleo: |r|=0.039; señal baja
    "A_007_071",      # Escala Izquierda-Derecha: |r|=0.021; señal baja
    # ── Exclusiones por decisión del investigador ─────────────────────────────
    "H_001_011",      # Confianza interpersonal: excluida por decisión metodológica
    "S_701",          # Práctica religiosa: sin relevancia política directa
    "X_008",          # Tamaño del municipio: sin señal (|r|=0.042) ni justificación
]

VARS_EXCLUIR_VDEM = [
    # Excluidas antes del split en NB01 (sin cobertura en test o redundancia extrema)
    "v2x_neopat",      # correlación 0.970 con v2x_rule
    "v2xnp_regcorr",   # correlación 0.985 con v2x_execorr
    "v2xpe_exlsocgr",  # sin datos en 2024 (test)
    "v2xpe_exlecon",   # sin datos en 2024 (test)
]

# No se invierten variables: los modelos de árboles son invariantes a
# transformaciones monotónicas. OLO y los gráficos ALE producen
# coeficientes y curvas con el signo correcto interpretado desde la
# escala original. Ver documento metodológico sección 5 para justificación.
VARS_CATEGORICAS = ["S_200"]   # única variable nominal pura del dataset

N_CLASES = 4

# ====================================================
# ETIQUETAS DEL TARGET
# ====================================================

ETIQUETAS = {
    0: "Para nada satisfecho",
    1: "No muy satisfecho",
    2: "Más bien satisfecho",
    3: "Muy satisfecho",
}

# ====================================================
# BLOQUES TEMÁTICOS Y ETIQUETAS DE FEATURES
#
# NOTA: los dos sub-bloques anteriores de V-Dem (High-level y Mid-level)
# se consolidan en un único bloque "Contexto democrático" con 4 variables,
# seleccionadas por cobertura semántica y mínima multicolinealidad:
#   Con 19 vars V-Dem: 44 pares con |r| > 0.85 (máx: 0.990)
#   Con  4 vars V-Dem:  1 par con |r| > 0.85  (máx: 0.862)
# ====================================================

BLOQUES = {
    "Confianza institucional": [
        "H_002_011",  # Confianza Congreso
        "H_002_031",  # Confianza Gobierno
        "H_002_041",  # Confianza Poder Judicial
        "H_002_111",  # Confianza Policía
        "H_002_131",  # Confianza Televisión
        "H_002_161",  # Confianza FF.AA.
        "H_002_241",  # Confianza Partidos Políticos
    ],
    "Evaluación económica": [
        "D_001_001",      # Situación económica país
        "D_001_021",      # Economía país vs. año anterior
        "D_001_041",      # Expectativa económica país
        "D_001_091",      # Expectativa económica personal
        "C_006_003_011",  # Distribución del ingreso justa
    ],
    "Percepción política": [
        "A_001_001",  # Apoyo a la democracia
        "A_007_001",  # Interés en política
        "B_001_101",  # País para todos / poderosos
        "B_006_061",  # Aprobación gobierno
    ],
    "Corrupción y seguridad": [
        "G_002_011",  # Conoce caso de corrupción
        "G_005_001",  # Progreso contra corrupción
        "I_001_001",  # Victimización delictiva (armonizada entre olas)
    ],
    "Características sociodemográficas": [
        "S_001",  # Sexo
        "S_002",  # Edad
        "S_101",  # Nivel educativo
        "S_200",  # Situación ocupacional (categórica nominal)
        "S_301",  # Nivel socioeconómico
    ],
    "Contexto democrático": [
        # 4 variables V-Dem: una por bloque semántico distinto
        # Cubren: democracia electoral, igualdad, integridad institucional,
        # Estado de derecho — los cuatro pilares del marco teórico de la tesis.
        "v2x_polyarchy",  # Democracia electoral (índice global de poliarquía)
        "v2x_egal",       # Componente igualitario (mayor señal: |r|=0.169 ind., 0.364 p-a)
        "v2x_corr",       # Integridad institucional (corrupción; escala: alto=más corrupción)
        "v2xcl_rol",      # Igualdad ante la ley (Estado de derecho y libertades civiles)
    ],
}

ETIQUETAS_FEATURES = {
    # ── Confianza institucional ────────────────────────────────────────────────
    "H_002_011": "Confianza Congreso",
    "H_002_031": "Confianza Gobierno",
    "H_002_041": "Confianza Poder Judicial",
    "H_002_111": "Confianza Policía",
    "H_002_131": "Confianza Televisión",
    "H_002_161": "Confianza FF.AA.",
    "H_002_241": "Confianza Partidos Políticos",
    # ── Evaluación económica ──────────────────────────────────────────────────
    "D_001_001":     "Situación económica país",
    "D_001_021":     "Economía país vs. año anterior",
    "D_001_041":     "Expectativa económica país",
    "D_001_091":     "Expectativa económica personal",
    "C_006_003_011": "Distribución ingreso justa",
    # ── Percepción política ───────────────────────────────────────────────────
    "A_001_001": "Apoyo a la democracia",
    "A_007_001": "Interés en política",
    "B_001_101": "País para todos / poderosos",
    "B_006_061": "Aprobación gobierno",
    # ── Corrupción y seguridad ────────────────────────────────────────────────
    "G_002_011": "Conoce caso de corrupción",
    "G_005_001": "Progreso contra corrupción",
    "I_001_001": "Victimización delictiva",
    # ── Características sociodemográficas ─────────────────────────────────────
    "S_001": "Sexo",
    "S_002": "Edad",
    "S_101": "Nivel educativo",
    "S_200": "Situación ocupacional",
    "S_301": "Nivel socioeconómico",
    # ── Contexto democrático (V-Dem) ──────────────────────────────────────────
    "v2x_polyarchy": "Democracia electoral",
    "v2x_egal":      "Componente igualitario",
    "v2x_corr":      "Integridad institucional (corrupción)",
    "v2xcl_rol":     "Igualdad ante la ley",
}


def bloque_de(var: str) -> str:
    for bloque, variables in BLOQUES.items():
        if var in variables:
            return bloque
    return "Sin clasificar"


# ====================================================
# UTILIDADES DE LIMPIEZA
# ====================================================

def delete_files(patron, descripcion):
    archivos = glob.glob(patron)
    if not archivos:
        print(f"  [vacío] {descripcion}")
        return
    for f in archivos:
        os.remove(f)
    print(f"  [ok] {descripcion}: {len(archivos)} archivo(s) eliminado(s)")


def empty_folders(ruta, descripcion):
    if not os.path.isdir(ruta):
        print(f"  [no existe] {ruta}")
        return
    items = list(os.scandir(ruta))
    if not items:
        print(f"  [vacío] {descripcion}")
        return
    for item in items:
        if item.is_dir():
            shutil.rmtree(item.path)
        else:
            os.remove(item.path)
    print(f"  [ok] {descripcion}: {len(items)} elemento(s) eliminado(s)")


def clean_process_folders():
    BASE = os.path.dirname(os.getcwd())
    print("=== Limpieza de carpetas ===\n")
    delete_files(os.path.join(BASE, "data/base/*"),              "data/base")
    delete_files(os.path.join(BASE, "data/processed/*"),         "data/processed")
    delete_files(os.path.join(BASE, "data/raw_latinobarometro/*.dta"), "data/raw_latinobarometro (*.dta)")
    delete_files(os.path.join(BASE, "data/raw_v-dem/*.csv"),     "data/raw_v-dem (*.csv)")
    delete_files(os.path.join(BASE, "models/*"),                  "models")
    empty_folders(os.path.join(BASE, "notebooks/catboost_info"), "notebooks/catboost_info")
    empty_folders(os.path.join(BASE, "notebooks/output"),        "notebooks/output")
    empty_folders(os.path.join(BASE, "results"),                  "results")
    print("\nListo.")
