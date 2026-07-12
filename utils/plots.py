from .config import THEME
import matplotlib.pyplot as plt

def model_color(model):

    return THEME["models"][model]

def save_figure(nombre, dpi=150, bbox_inches="tight"):

    plt.tight_layout()

    plt.savefig(
        f"../results/figures/{nombre}.png",
        dpi=dpi,
        bbox_inches=bbox_inches
    )