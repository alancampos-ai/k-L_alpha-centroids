import json
import numpy as np
import matplotlib.pyplot as plt

ARQ_MAX = "dice_alpha_max.json"
ARQ_MIN = "dice_alpha_min.json"

OUT_MAX = "dice_alpha_max.png"
OUT_MIN = "dice_alpha_min.png"

CORES = {
    "EUC": "#332288",
    "LOG": "#117733",
    "AIRM": "#44AA99"
}


def carregar_json(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def ajustar_curva(valores_alpha, valores_dice, n_pontos=200):
    valores_alpha = np.asarray(valores_alpha, dtype=float)
    valores_dice = np.asarray(valores_dice, dtype=float)

    indice = np.argsort(valores_alpha)
    valores_alpha = valores_alpha[indice]
    valores_dice = valores_dice[indice]

    grau = min(3, len(valores_alpha) - 1)
    coeficientes = np.polyfit(valores_alpha, valores_dice, deg=grau)
    polinomio = np.poly1d(coeficientes)

    x_suave = np.linspace(valores_alpha[0], valores_alpha[-1], n_pontos)
    y_suave = polinomio(x_suave)
    return x_suave, y_suave


def grafico_dice_alpha(dados, titulo, nome_arquivo, usar_xticks_originais=False):
    plt.figure(figsize=(8, 6))

    todos_alpha = []

    for metodo in ["EUC", "LOG", "AIRM"]:
        if metodo not in dados:
            continue

        valores_alpha = [p["alpha"] for p in dados[metodo]]
        valores_dice = [p["dice"] for p in dados[metodo]]
        todos_alpha.extend(valores_alpha)

        x_suave, y_suave = ajustar_curva(valores_alpha, valores_dice)

        plt.plot(
            x_suave,
            y_suave,
            linewidth=2,
            label=metodo,
            color=CORES[metodo]
        )

    plt.xlabel("alpha")
    plt.ylabel("Dice")
    plt.title(titulo)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    if usar_xticks_originais:
        xticks = [1.0, 1.25, 1.5, 1.75, 2.0]
        plt.xlim(xticks[0], xticks[-1])
        plt.xticks(xticks)
    else:
        if todos_alpha:
            xmin, xmax = min(todos_alpha), max(todos_alpha)
            plt.xlim(xmin, xmax)

    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=300)
    plt.close()


def main():
    dados_max = carregar_json(ARQ_MAX)
    dados_min = carregar_json(ARQ_MIN)

    grafico_dice_alpha(
        dados_max,
        "Dice vs alpha",
        OUT_MAX,
        usar_xticks_originais=True
    )

    grafico_dice_alpha(
        dados_min,
        "Dice vs alpha",
        OUT_MIN,
        usar_xticks_originais=False
    )


if __name__ == "__main__":
    main()
