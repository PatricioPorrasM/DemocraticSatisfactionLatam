from .config import THEME
import matplotlib.pyplot as plt

def model_color(model):

    return THEME["models"][model]

def save_figure(nombre):

    plt.tight_layout()

    plt.savefig(
        f"../results/figures/{nombre}.png",
        dpi=300
    )