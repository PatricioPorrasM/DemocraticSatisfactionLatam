import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

# ====================================================
# PATHS
# ====================================================

PARAMETERS = {
    "SEED": 42,
    "YEAR_START": 1995,
    "YEAR_END": 2024,
}

# ====================================================
# PATHS
# ====================================================

PATHS = {
    # --- Configuración de rutas ----------------------------------------------
    "FOLDER_RAW_ZIP": Path("..") / "data"/ "raw_zip",
    "FOLDER_RAW_LB": Path("..") / "data" / "raw_latinobarometro",
    "FOLDER_RAW_VDEM": Path("..") / "data" / "raw_v-dem",
    "FOLDER_BASE": Path("..") / "data" / "base",
    "FOLDER_PROCS": Path("..") / "data" / "processed",
    "FOLDER_MODELS": Path("..") / "models",
    "FOLDER_RESULTS": Path("..") / "results",
    "FOLDER_RESULTS_FIGURES": Path("..") / "results" / "figures",
    "FOLDER_RESULTS_METRICS": Path("..") / "results" / "metrics",
    "FOLDER_RESULTS_TABLES": Path("..") / "results" / "tables",
    "FOLDER_RESULTS_SHAP": Path("..") / "results" / "shap",
    "FILE_RAW_VDEM": Path("..") / "data" / "raw_v-dem" / "V-Dem-CY-Core-v16.csv",
    "FILE_BASE_VDEM": Path("..") / "data" / "base" / "v-dem.csv",
    "FILE_BASE_LB": Path("..") / "data" / "base" / "latinobarometro.csv",
    "FILE_LB_VAR_MAPPING": Path("..") / "data" / "variables" / "latinobarometro_variable_mapping.csv",
    "FILE_VAR_SELECTION": Path("..") / "data" / "variables" / "variables_selection.csv",
    "FILE_FREQUENCY_TABLE": Path("..") / "data" / "base" / "lb_frecuencia_valores_por_ola.csv",
    "FILE_SAMPLE": Path("..") / "data" / "base" / "latinobarometro_muestra.csv",
    "FILE_RESULTS_MODEL_CSV": Path("..") / "data" / "results" / "resultados_modelos.csv",
    "FILE_RESULTS_MODEL_PARQUET": Path("..") / "data" / "results" / "resultados_modelos.parquet",
}

# ====================================================
# TEMAS
# ====================================================

THEME = {
    "models": {
        "OLO": "#4C78A8",        # Azul
        "CatBoost": "#F58518",   # Naranja
        "XGBoost": "#54A24B",    # Verde
        "LightGBM": "#E45756",   # Rojo
        "TabNet": "#72B7B2"      # Turquesa
    },
    "target": {
        0: "#DC2626",            # Para nada satisfecho
        1: "#F58518",            # No muy satisfecho
        2: "#54A24B",            # Más bien satisfecho
        3: "#4C78A8",            # Muy satisfecho
    },
    "semantic": {
        "success": "#2CA02C",
        "warning": "#FFB000",
        "danger": "#D62728",
        "grid": "#E8E8E8",
        "text": "#303030",
        "background": "#FFFFFF"
    },
    "blocks": {
        "Confianza institucional"        : "#1E3A5F",
        "Evaluación económica"           : "#0D9488",
        "Percepción política"            : "#2E74B5",
        "Corrupción y seguridad"         : "#DC2626",
        "Características sociodemográficas": "#78716C",
    }
}


# ====================================================
# PALETAS
# ====================================================

PALETTES = {
    "models": list(THEME["models"].values()),
    "categorical": sns.color_palette("colorblind"),
    "sequential": sns.color_palette("viridis"),
    "diverging": sns.color_palette("coolwarm")
}


# ====================================================
# CONFIGURACIÓN GENERAL
# ====================================================

def setup_plots():

    sns.set_theme(
        style="whitegrid",
        palette=PALETTES["models"]
    )

    mpl.rcParams.update({
        "figure.figsize": (10,6),
        "figure.dpi": 150,
        "savefig.dpi":300,
        "font.family":"DejaVu Sans",
        "font.size":12,
        "axes.titlesize":16,
        "axes.labelsize":13,
        "axes.edgecolor":"#555555",
        "axes.prop_cycle":plt.cycler(color=PALETTES["models"]),
        "xtick.labelsize":11,
        "ytick.labelsize":11,
        "grid.color":THEME["semantic"]["grid"],
        "grid.linestyle":"--",
        "legend.frameon":False,
        "lines.linewidth":2.5,
        "lines.markersize":7,
        "savefig.bbox":"tight"
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
# SUBPERIODOS — Expanding Window Walk-Forward
# ====================================================

SUBPERIODOS = {
    "SP1": {
        "train_olas"  : [1995,1996,1997,1998,2000,2001,2002,2003,2004,2005],
        "validate_ola": [2006],
        "test_ola"    : [2007],
        "descripcion" : "Consolidación democrática (1995–2007)",
    },
    "SP2": {
        "train_olas"  : [1995,1996,1997,1998,2000,2001,2002,2003,2004,2005,2006,
                         2007,2008,2009,2010,2011,2013,2015,2016],
        "validate_ola": [2017],
        "test_ola"    : [2018],
        "descripcion" : "Crisis y reconfiguración (2008–2018)",
    },
    "SP3": {
        "train_olas"  : [1995,1996,1997,1998,2000,2001,2002,2003,2004,2005,2006,
                         2007,2008,2009,2010,2011,2013,2015,2016,2017,2018,2020],
        "validate_ola": [2023],
        "test_ola"    : [2024],
        "descripcion" : "Pandemia y recuperación (2020–2024)",
    },
}

# ====================================================
# SUBREGIONES Y PAÍSES
# ====================================================

SUBREGIONES = {
    "Cono Sur"       : ["Argentina","Chile","Uruguay","Paraguay","Perú"],
    "Región Andina"  : ["Bolivia","Colombia","Ecuador","Venezuela"],
    "Brasil"         : ["Brasil"],
    "Centroamérica"  : ["Costa Rica","El Salvador","Guatemala","Honduras","Nicaragua","Panamá"],
    "México y Caribe": ["México","República Dominicana"],
}

PAISES_EXCLUIR_TEST = ["Venezuela", "Nicaragua"]
AÑO_CORTE_VEN       = 2017

# ====================================================
# MAPEOS DE IDENTIFICACIÓN
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
    "C_001_031",  # ruptura de codificación en 2018
    "A_003_021",  # ausente en SP2_test y SP3_test
    "D_001_061",  # ausente en los 3 conjuntos de prueba
    "D_001_131",  # ausente en SP2_test y SP3_test
    "X_004",      # 627 categorías; sin señal predictiva
    "S_700",      # sin señal en ningún subperiodo; alta cardinalidad
]

VARS_EXCLUIR_VDEM = [
    "v2x_neopat",     # correlación 0.970 con v2x_rule
    "v2xnp_regcorr",  # correlación 0.985 con v2x_execorr
    "v2xpe_exlsocgr", # sin datos en 2024 (SP3_test)
    "v2xpe_exlecon",  # sin datos en 2024 (SP3_test)
]

VARS_LIKERT4_INVERTIR = [
    "H_002_011", "H_002_031", "H_002_041", "H_002_101",
    "H_002_111", "H_002_131", "H_002_161", "H_002_241",
    "G_005_001",
]

VARS_LIKERT4_INTERES = ["A_007_001"]

VARS_VDEM_INVERTIR = ["v2x_corr", "v2x_execorr", "v2x_pubcorr"]

VARS_CATEGORICAS = ["S_200"]

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
# ====================================================

BLOQUES = {
    "Confianza institucional": [
        "H_002_011", "H_002_031", "H_002_041", "H_002_101",
        "H_002_111", "H_002_131", "H_002_161", "H_002_241",
        "H_001_011",
    ],
    "Evaluación económica": [
        "D_001_001", "D_001_021", "D_001_041", "D_001_091",
        "C_003_003_011", "C_006_003_011",
    ],
    "Percepción política": [
        "A_001_001", "A_007_071", "A_007_001",
        "B_001_101", "B_006_061",
    ],
    "Corrupción y seguridad": [
        "G_002_011", "G_005_001", "I_001_001",
    ],
    "Características sociodemográficas": [
        "S_001", "S_002", "S_101", "S_200", "S_301", "S_701", "X_008",
    ],
    "Contexto democrático · High-level": [
        "v2x_polyarchy", "v2x_libdem", "v2x_partipdem",
        "v2x_delibdem", "v2x_egaldem",
    ],
    "Contexto democrático · Mid-level": [
        "v2xcl_rol", "v2x_jucon", "v2xlg_legcon", "v2x_freexp_altinf",
        "v2x_cspart", "v2xcs_ccsi",
        "v2x_egal", "v2xeg_eqdr",
        "v2x_accountability_osp", "v2x_rule",
        "v2x_corr", "v2x_execorr", "v2x_pubcorr",
        "v2x_gender", "v2x_polyarchy",
    ],
}

ETIQUETAS_FEATURES = {
    # ── Confianza institucional ────────────────────────────────────────────────
    "H_002_011": "Confianza Congreso",
    "H_002_031": "Confianza Gobierno",
    "H_002_041": "Confianza Poder Judicial",
    "H_002_101": "Confianza Iglesia Católica",
    "H_002_111": "Confianza Policía",
    "H_002_131": "Confianza Televisión",
    "H_002_161": "Confianza FF.AA.",
    "H_002_241": "Confianza Partidos Políticos",
    "H_001_011": "Confianza interpersonal",
    # ── Evaluación económica ──────────────────────────────────────────────────
    "D_001_001": "Situación económica país",
    "D_001_021": "Economía país vs. año anterior",
    "D_001_041": "Expectativa económica país",
    "D_001_091": "Expectativa económica personal",
    "C_003_003_011": "Preocupación desempleo",
    "C_006_003_011": "Distribución ingreso justa",
    # ── Percepción política ───────────────────────────────────────────────────
    "A_001_001": "Apoyo a la democracia",
    "A_007_071": "Escala Izquierda-Derecha",
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
    "S_701": "Práctica religiosa",
    "X_008": "Tamaño municipio (urbano-rural)",
    # ── Contexto democrático · High-level ────────────────────────────────────
    "v2x_polyarchy" : "Democracia electoral",
    "v2x_libdem"    : "Democracia liberal",
    "v2x_partipdem" : "Democracia participativa",
    "v2x_delibdem"  : "Democracia deliberativa",
    "v2x_egaldem"   : "Democracia igualitaria",
    # ── Contexto democrático · Mid-level ─────────────────────────────────────
    "v2xcl_rol"             : "Igualdad ante la ley",
    "v2x_jucon"             : "Independencia judicial",
    "v2xlg_legcon"          : "Control legislativo",
    "v2x_freexp_altinf"     : "Libertad de expresión",
    "v2x_cspart"            : "Participación soc. civil",
    "v2xcs_ccsi"            : "Índice soc. civil (CSI)",
    "v2x_egal"              : "Componente igualitario",
    "v2xeg_eqdr"            : "Distribución igualitaria",
    "v2x_accountability_osp": "Rendición de cuentas",
    "v2x_rule"              : "Estado de derecho",
    "v2x_corr"              : "Integridad institucional",
    "v2x_execorr"           : "Integridad ejecutiva",
    "v2x_pubcorr"           : "Integridad sector público",
    "v2x_gender"            : "Empoderamiento político mujeres",
}


def bloque_de(var: str) -> str:
    for bloque, variables in BLOQUES.items():
        if var in variables:
            return bloque
    return "Sin clasificar"
