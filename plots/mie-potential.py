import numpy as np
import matplotlib.pyplot as plt
import cr_mech_coli as crm
from cr_mech_coli import crm_fit

if __name__ == "__main__":
    bound = 6.0
    cutoff = 3.5

    x = np.linspace(0.01, 3.9, 200)

    data = [
        (2, 1, ":"),
        (3, 2, "-."),
        (4, 2, "--"),
        (5, 3, "-"),
    ]

    crm.set_mpl_rc_params()
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    crm.configure_ax(axs[0])
    crm.configure_ax(axs[1])

    for ax, label in zip(axs, ("C", "D")):
        ax.text(
            0.03,
            0.97,
            label,
            fontsize=40,
            fontweight="semibold",
            fontfamily="serif",
            va="top",
            horizontalalignment="left",
            transform=ax.transAxes,
        )

    for en, em, ls in data:
        crm_fit.plot_mie_potential(x, 1, en, em, 1, bound, cutoff, (fig, axs[0]), ls)

    axs[0].set_xlim(0, np.max(x))
    axs[0].set_ylim(-2.5, 3)
    axs[0].legend()
    axs[0].set_title("Mie Potential")
    axs[0].set_xlabel("Distance [R$_1$+R$_2$]")
    axs[0].set_ylabel("Interaction Strength [V$_0$]")

    data2 = [(1.0, ":"), (1.5, "-."), (3.0, "--")]
    for sti, ls in data2:
        label = f"ω={sti:3.1f}/(R$_1$+R$_2$)"
        crm_fit.plot_morse_potential(
            x,
            1,
            sti,
            1,
            cutoff,
            (fig, axs[1]),
            ls,
            label=label,
            yoffset=-1,
        )

    axs[1].set_xlim(0, np.max(x))
    xmin = -1
    xmax = 4
    dx = 0.05 * (xmax - xmin)
    axs[1].set_ylim(xmin - dx, xmax + dx)
    axs[1].legend()
    axs[1].set_title("Morse Potential")
    axs[1].set_xlabel("Distance [R$_1$+R$_2$]")
    axs[1].set_ylabel("Interaction Strength [V$_0$]")

    fig.tight_layout()
    fig.savefig("docs/source/_static/interaction-potentials.png")
    fig.savefig("docs/source/_static/interaction-potentials.pdf")
