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
