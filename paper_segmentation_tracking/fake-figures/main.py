import matplotlib.pyplot as plt
import numpy as np
import cr_mech_coli as crm


def configure_ax(ax):
    crm.configure_ax(ax)
    ax.text(
        0.5,
        0.5,
        "FAKE DATA",
        transform=ax.transAxes,
        fontsize=60,
        color="red",
        alpha=0.5,
        ha="center",
        va="center",
        rotation=30,
    )


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 8))
    configure_ax(ax)

    fake_data = {
        # Name, (performance before, performance after)
        "Cellpose": (0.8, 0.85),
        "Omnipose": (0.9, 0.92),
        "Other ..": (0.5, 0.6),
    }

    width = 0.33
    padding = 0.02
    x = np.arange(2)
    colors = ["black", "gray"]
    labels = ["Before", "After"]

    for n, (name, measurements) in enumerate(fake_data.items()):
        rects = ax.bar(
            [n - width / 2 - padding / 2, n + width / 2 + padding / 2],
            measurements,
            width,
            label=labels,
            color=colors,
        )
        ax.bar_label(rects, padding=3)

    ax.set_xticks(np.arange(len(fake_data)), fake_data.keys())
    ax.set_ylim(0, 1.1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[: len(colors)], labels[: len(colors)])

    fig.savefig("paper_segmentation_tracking/fake-figures/segmentation-benchmark.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    configure_ax(ax)

    width = 0.25
    fake_data = {
        "Cellpose": (5.1, 15.34, 60.3),
        "Omnipose": (3.1, 16.34, 34.3),
        "Other ..": (7.8, 26.1, 87.7),
    }

    colors = ["green", "orange", "blue"]
    labels = ["Sensor Noise", "Smudges", "Shaking"]
    x = np.arange(len(labels))

    for n, (name, measurements) in enumerate(fake_data.items()):
        rects = ax.bar(
            n + x * width - width / 2,
            measurements,
            width,
            label=labels,
            color=colors,
        )
        ax.bar_label(rects, padding=3)

    ax.set_xticks(np.arange(len(fake_data)) + width / 2, fake_data.keys())
    ax.set_ylabel("Relative Impact [%]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[: len(colors)], labels[: len(colors)])

    fig.savefig("paper_segmentation_tracking/fake-figures/tracking-defects.pdf")
    plt.close(fig)
