import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


SEED = 42

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