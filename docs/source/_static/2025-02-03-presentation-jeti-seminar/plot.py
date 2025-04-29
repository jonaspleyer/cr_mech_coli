import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["lines.color"] = "white"
mpl.rcParams["patch.edgecolor"] = "white"
mpl.rcParams["text.color"] = "white"
mpl.rcParams["axes.facecolor"] = (0, 0, 0, 0)
mpl.rcParams["axes.edgecolor"] = "white"
mpl.rcParams["axes.labelcolor"] = "white"
mpl.rcParams["xtick.color"] = "white"
mpl.rcParams["ytick.color"] = "white"
mpl.rcParams["grid.color"] = "white"
mpl.rcParams["figure.facecolor"] = (0, 0, 0, 0)
mpl.rcParams["figure.edgecolor"] = (0, 0, 0, 0)
mpl.rcParams["savefig.facecolor"] = (0, 0, 0, 0)
mpl.rcParams["savefig.edgecolor"] = (0, 0, 0, 0)
mpl.rcParams["legend.facecolor"] = (0, 0, 0, 0)
mpl.rcParams["legend.edgecolor"] = "white"
mpl.rcParams["legend.framealpha"] = None

figsize = (10, 6)


def plot_morse_potential():
    x = np.linspace(0, 3)

    def morse(r, lam):
        return (1 - np.exp(-lam * (r - 1))) ** 2

    y1 = morse(x, 0.5)
    y2 = morse(x, 1)
    y3 = morse(x, 2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y1, label="$\\lambda=0.5$", linestyle=":", color="w")
    ax.plot(x, y2, label="$\\lambda=1$", linestyle="-.", color="w")
    ax.plot(x, y3, label="$\\lambda=2$", linestyle="-", color="w")
    ax.set_ylim(-0.1, 3)
    ax.legend()
    ax.set_title("Morse Potential")
    # ax.set_xlabel("x [R]")
    x_ticks = np.linspace(min(x), max(x), 7)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["${}R$".format(x) for x in x_ticks])
    y_ticks = np.linspace(0, 3, 7)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(["${}V_0$".format(y) for y in y_ticks])
    fig.savefig("morse_potential.png")


def plot_mie_potential():
    x = np.linspace(0, 3)

    def sigma(r, n, m):
        return (m / n) ** (1 / (n - m)) * r

    def C(n, m):
        return n / (n - m) * (n / m) ** (n / (n - m))

    def mie(t, n, m):
        return C(n, m) * ((sigma(1, n, m) / t) ** n - (sigma(1, n, m) / t) ** m)

    y1 = mie(x, 3, 1)
    y11 = mie(x, 3, 2)
    y2 = mie(x, 4, 2)
    y3 = mie(x, 5, 3)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y1, label="$n=3,m=1$", linestyle=":", color="w")
    ax.plot(x, y11, label="$n=3,m=2$", linestyle="-.", color="w")
    ax.plot(x, y2, label="$n=4,m=2$", linestyle="--", color="w")
    ax.plot(x, y3, label="$n=5,m=3$", linestyle="-", color="w")
    ax.set_ylim(-3.1, 3)
    ax.legend()
    ax.set_title("Mie Potential")
    x_ticks = np.linspace(min(x), max(x), 7)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["${}R$".format(x) for x in x_ticks])
    y_ticks = np.linspace(-3, 3, 7)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(["${}V_0$".format(y) for y in y_ticks])
    fig.savefig("mie_potential.png")


if __name__ == "__main__":
    plot_morse_potential()
    plot_mie_potential()
