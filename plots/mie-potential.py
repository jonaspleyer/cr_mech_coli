import numpy as np

from cr_mech_coli import crm_fit

if __name__ == "__main__":
    bound = 6.0
    cutoff = 3.0

    x = np.linspace(0.01, 3.3, 100)

    data = [
        (2, 1, ":"),
        (3, 2, "-."),
        (4, 2, "--"),
        (5, 3, "-"),
    ]

    fig_ax = None
    for en, em, ls in data:
        f = crm_fit.plot_mie_potential(x, 1, en, em, 1, bound, cutoff, fig_ax, ls)
        fig_ax = (f[0], f[1])
    fig, ax = fig_ax

    ax.set_xlim(0, 3.3)
    ax.set_ylim(-2.5, 3)
    ax.legend()
    ax.set_title("Mie Potential")

    ax.set_xlabel("Distance [R$_1$+R$_2$]")
    ax.set_ylabel("Interaction Strength [U$_0$]")

    fig.savefig("docs/source/_static/mie-potential-shapes.png")
    fig.savefig("docs/source/_static/mie-potential-shapes.pdf")
