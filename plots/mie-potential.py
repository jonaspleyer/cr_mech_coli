import numpy as np
import matplotlib.pyplot as plt

import cr_mech_coli as crm


def plot_mie_potential():
    def sigma(r, n, m):
        return (m / n) ** (1 / (n - m)) * r

    def C(n, m):
        return n / (n - m) * (n / m) ** (n / (n - m))

    def mie(t, n, m, bound, cutoff):
        strength = C(n, m) * ((sigma(1, n, m) / t) ** n - (sigma(1, n, m) / t) ** m)
        return np.minimum(strength, bound) * (t <= cutoff)

    bound = 2.5
    cutoff = 3.0

    x = np.linspace(0.2, 3.3, 100)

    data = [
        (2, 1, ":"),
        (3, 2, "-."),
        (4, 2, "--"),
        (5, 3, "-"),
    ]

    crm.plotting.set_mpl_rc_params()
    fig, ax = plt.subplots(figsize=(8, 8))
    crm.plotting.configure_ax(ax)

    for en, em, ls in data:
        y = mie(x, en, em, bound, cutoff)
        ax.plot(x, y, label=f"n={en},m={em}", linestyle=ls, color=crm.plotting.COLOR3)

    ax.set_ylim(-2.5, 3)
    ax.legend()
    ax.set_title("Mie Potential")

    ax.set_xlabel("Distance [R$_1$+R$_2$]")
    ax.set_ylabel("Interaction Strength [U$_0$]")

    fig.savefig("docs/source/_static/mie-potential-shapes.png")
    fig.savefig("docs/source/_static/mie-potential-shapes.pdf")


if __name__ == "__main__":
    plot_mie_potential()
