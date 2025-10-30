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
        ds = np.abs(
            C(n, m)
            / sigma(1, n, m)
            * (
                n * (sigma(1, n, m) / t) ** (n + 1)
                - m * (sigma(1, n, m) / t) ** (m + 1)
            )
        )
        co = t <= cutoff
        n = np.argmin(co)
        bo = ds <= bound
        m = np.argmax(bo)
        s = strength * bo + (1 - bo) * (bound * (t[m] - t) + strength[m])
        return s * co + (1 - co) * strength[n], m

    bound = 6.0
    cutoff = 3.0

    x = np.linspace(0.01, 3.3, 100)

    data = [
        (2, 1, ":"),
        (3, 2, "-."),
        (4, 2, "--"),
        (5, 3, "-"),
    ]

    crm.plotting.set_mpl_rc_params()
    fig, ax = plt.subplots(figsize=(8, 8))
    crm.plotting.configure_ax(ax)

    yfinmax = -10
    for en, em, ls in data:
        y, m_bound = mie(x, en, em, bound, cutoff)
        yfinmax = max(yfinmax, y[-1])
        ax.plot(
            x[: m_bound + 1], y[: m_bound + 1], linestyle=ls, color=crm.plotting.COLOR2
        )
        ax.plot(
            x[m_bound:],
            y[m_bound:],
            label=f"n={en},m={em}",
            linestyle=ls,
            color=crm.plotting.COLOR3,
        )

    ax.vlines(cutoff, -2.5, yfinmax, color=crm.plotting.COLOR5)

    ax.set_xlim(0, 3.3)
    ax.set_ylim(-2.5, 3)
    ax.legend()
    ax.set_title("Mie Potential")

    ax.set_xlabel("Distance [R$_1$+R$_2$]")
    ax.set_ylabel("Interaction Strength [U$_0$]")

    fig.savefig("docs/source/_static/mie-potential-shapes.png")
    fig.savefig("docs/source/_static/mie-potential-shapes.pdf")


if __name__ == "__main__":
    plot_mie_potential()
