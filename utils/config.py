import os
import glob
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

# ====================================================
# PARAMETERS
# ====================================================

PARAMETERS = {
    "LOAD_SAMPLE": True,
    "SEED": 42,
    "YEAR_START": 1995,
    "YEAR_END": 2024,
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
    "Cono Sur":        ["Argentina", "Chile", "Uruguay", "Paraguay", "Perú"],
    "Región Andina":   ["Bolivia", "Colombia", "Ecuador"],
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


def emty_folders(ruta, descripcion):
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
    delete_files(os.path.join(BASE, "data/raw_c-dem/*.csv"),     "data/raw_c-dem (*.csv)")
    delete_files(os.path.join(BASE, "models/*"),                  "models")
    emty_folders(os.path.join(BASE, "notebooks/catboost_info"), "notebooks/catboost_info")
    emty_folders(os.path.join(BASE, "notebooks/output"),        "notebooks/output")
    emty_folders(os.path.join(BASE, "results"),                  "results")
    print("\nListo.")
