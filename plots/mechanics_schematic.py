import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Rectangle
from matplotlib import rc, rc_context
import pyvista as pv
import cr_mech_coli as crm
from copy import deepcopy

# rcParams['path.sketch'] = (3, 10, 1)
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
rc("font", size="20")


def __plot_cell_envelope(ax, points, radius, cell_color=[0.9, 0.9, 0.9]):
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        c = p2 - p1
        angle = np.arccos(c[0] / np.linalg.norm(c))
        direction = np.array([np.sin(angle), -np.cos(angle)])
        rect = Rectangle(
            points[i] + radius * direction,
            width=np.linalg.norm(c),
            height=2 * radius,
            angle=angle / 2 / np.pi * 360,
            rotation_point="xy",
            color=cell_color,
        )
        ax.add_patch(rect)
        circle = Circle(points[i], radius, color=cell_color)
        ax.add_patch(circle)
    circle = Circle(points[-1], radius, color=cell_color)
    ax.add_patch(circle)


def __plot_cell_springs(ax, points, color=[0.5, 0.5, 0.5]):
    with rc_context({"path.sketch": (4, 7, 1)}):
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1)


def plot_polygon_with_arrows(points, radius, angle_circle_size=0.8):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")

    __plot_cell_envelope(ax, points, radius)

    ax.scatter(points[:, 0], points[:, 1], s=80, marker="+", color="k")
    ax.text(*(points[0] + np.array([-0.2, 0.2])), "$\\vec x_{}$".format(0))
    ax.text(
        *(points[-1] + np.array([-0.2, 0.2])), "$\\vec x_{}$".format(len(points) - 1)
    )

    __plot_cell_springs(ax, points)

    for i in range(1, len(points) - 1):
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[i + 1]
        c1 = p1 - p2
        c2 = p3 - p2

        perp = c1[0] * c2[1] - c1[1] * c2[0]
        alpha = (
            np.arccos(np.dot(c2, np.array([1, 0])) / np.linalg.norm(c2))
            / 2
            / np.pi
            * 360
        )
        t = np.dot(c2, c1) / np.linalg.norm(c1) / np.linalg.norm(c2)
        theta2 = np.arccos(t) / 2 / np.pi * 360

        if perp > 0.0:
            alpha = 360 - theta2 + alpha

        a = (alpha + 0.5 * theta2) * 2 * np.pi / 360
        text_pos = np.array(
            [
                0.25 * angle_circle_size * np.cos(np.pi + a),
                0.25 * angle_circle_size * np.sin(np.pi + a),
            ]
        )
        ax.text(
            *(p2 + text_pos),
            "$\\vec x_{}$".format(i),
            verticalalignment="center",
            horizontalalignment="center",
        )
        ax.text(
            *(p2 - text_pos),
            "$\\alpha_{}$".format(i),
            verticalalignment="center",
            horizontalalignment="center",
        )

        arc = Arc(p2, angle_circle_size, angle_circle_size, angle=alpha, theta2=theta2)
        ax.add_patch(arc)
    ax.set_xlim(np.min(points[:, 0]) - radius, np.max(points[:, 0]) + radius)
    ax.set_ylim(np.min(points[:, 1]) - radius, np.max(points[:, 1]) + radius)
    fig.tight_layout()
    return fig, ax


def _closest_point_on_segment(p, a, b):
    ap = p - a
    ab = b - a

    ab_squared = np.dot(ab, ab)
    if ab_squared == 0:
        return a

    t = np.dot(ap, ab) / ab_squared
    t = max(0, min(1, t))

    closest = a + t * ab

    dist = np.linalg.norm(p - closest)
    return dist, closest


def _closest_point_on_polygon(p, polygon):
    dists = []
    points = []
    for pi1, pi2 in zip(polygon[1:], polygon[:-1]):
        dist, closest = _closest_point_on_segment(p, pi1, pi2)
        dists.append(dist)
        points.append(closest)

    n = np.argmin(dists)
    return points[n]


def plot_cells_interacting(points1, points2, radius):
    points = np.vstack([points1, points2])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")

    __plot_cell_envelope(ax, points1, radius)
    __plot_cell_envelope(ax, points2, radius)

    __plot_cell_springs(ax, points1)
    __plot_cell_springs(ax, points2)

    for p in points1:
        closest = _closest_point_on_polygon(p, points2)
        x = [p[0], closest[0]]
        y = [p[1], closest[1]]
        ax.plot(x, y, c=crm.plotting.COLOR5, linestyle="--")

    ax.scatter(points[:, 0], points[:, 1], s=80, marker="+", color="k")

    xmin = np.min(points[:, 0]) - radius
    xmax = np.max(points[:, 0]) + radius
    ymin = np.min(points[:, 1]) - radius
    ymax = np.max(points[:, 1]) + radius
    dx = xmax - xmin
    dy = ymax - ymin
    dd = max(dx, dy)
    ddx = (dd - dx) / 2
    ddy = (dd - dy) / 2

    ax.set_xlim(xmin - ddx, xmin + dd - ddx)
    ax.set_ylim(ymin - ddy, ymin + dd - ddy)

    fig.tight_layout()
    return fig, ax


def plot_3d_rod(points, radius):
    pos = np.zeros((points.shape[0], 3), dtype=np.float32)
    pos[:, :2] = points
    vel = 0 * pos
    interaction = crm.PhysicalInteraction(
        crm.MiePotentialF32(radius, 0.1, 0.1, 0.1, 1.0, 2.0)
    )
    agent = crm.RodAgent(pos, vel, interaction)

    domain_size = np.max(points) + radius
    plotter = pv.Plotter(
        off_screen=True,
        window_size=[int(np.ceil(domain_size)) * 300] * 2,
    )
    p1 = np.array([-radius] * 3)
    p2 = np.array([domain_size, domain_size, -radius])
    p3 = np.array([-radius, domain_size, -radius])
    rect = pv.Rectangle(np.array([p1, p2, p3]))
    plotter.add_mesh(rect)
    bounds = (-radius, domain_size, -radius, domain_size, -radius, radius)
    pv.Plotter.view_xy(plotter, bounds=bounds)
    pv.Plotter.enable_parallel_projection(plotter)
    plotter.camera.tight(padding=0)
    plotter.camera.position = (*plotter.camera.position[:2], 100 * domain_size)
    camera = plotter.camera.copy()
    plotter.clear_actors()

    # Prepare perlin noise
    freq1 = [0.689, 0.562, 0.683]
    freq2 = [4.97, 5.2, 5.18]
    noise1 = pv.perlin_noise(0.80, freq1, (0, 0, 0))
    noise2 = pv.perlin_noise(0.40, freq2, (0, 0, 0))

    def noise(p):
        n1 = noise1.EvaluateFunction(p)
        n2 = noise2.EvaluateFunction(p)
        return n1 + n2

    z_resolution = 50
    theta_resolution = 100
    mesh = pv.Sphere(
        center=agent.pos[0],
        radius=float(agent.radius),
        theta_resolution=theta_resolution,
    )
    meshes = [mesh]
    for pos1, pos2 in zip(agent.pos[:-1], agent.pos[1:]):
        center = 0.5 * (pos1 + pos2)
        direction = pos2 - center
        height = float(np.linalg.norm(pos1 - pos2))
        cylinder = pv.CylinderStructured(
            center=center,
            direction=direction,
            radius=float(agent.radius),
            height=height,
            theta_resolution=theta_resolution,
            z_resolution=z_resolution,
        )
        meshes.append(cylinder)
        sphere = pv.Sphere(
            center=pos2,
            radius=float(agent.radius),
            theta_resolution=theta_resolution,
        )
        meshes.append(sphere)

    mesh = pv.merge(meshes)
    mesh = mesh.triangulate()
    mesh = mesh.extract_surface(nonlinear_subdivision=3)

    import pyacvd

    clus = pyacvd.Clustering(mesh)
    clus.subdivide(3)
    clus.cluster(20000, iso_try=20)
    mesh = clus.create_mesh()

    mesh["scalars1"] = [noise1.EvaluateFunction(p) for p in mesh.points]
    mesh = mesh.warp_by_scalar("scalars1")

    # mesh.smooth(n_iter=100, inplace=True, relaxation_factor=0.7)
    mesh.smooth(n_iter=20, inplace=True)
    mesh.subdivide(2)

    mesh["scalars2"] = [noise2.EvaluateFunction(p) for p in mesh.points]
    mesh = mesh.warp_by_scalar("scalars2")
    mesh.smooth(n_iter=5, inplace=True)

    actor = plotter.add_mesh(
        mesh,
        # show_edges=True,
        show_scalar_bar=False,
        scalars=None,
        color=[150, 150, 150],
        smooth_shading=True,
        metallic=0,
        roughness=1,
    )
    actor.UseBoundsOff()

    plotter.disable_anti_aliasing()
    img = np.array(plotter.screenshot())
    plotter.close()

    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    points = np.array(
        [
            [0, 0],
            [1, 0.7],
            [2, 1.9],
            [3, 2.6],
            [4, 3.8],
            [5, 4.7],
        ]
    )

    radius = 0.8
    fig, ax = plot_polygon_with_arrows(points, radius)
    fig.savefig("docs/source/_static/mechanics.png", transparent=True)
    fig.savefig("docs/source/_static/mechanics.pdf", transparent=True)
    fig.savefig("docs/source/_static/mechanics.svg", transparent=True)
    plt.close(fig)

    points2 = np.array(points)
    points2 += np.array([3, -2])
    points2[0] += np.array([0.1, -0.2])
    points2[2] += np.array([0.3, -0.2])
    points2[4] += np.array([0.1, -0.1])
    fig, ax = plot_cells_interacting(points, points2, radius)
    fig.savefig("docs/source/_static/interaction.png", transparent=True)
    fig.savefig("docs/source/_static/interaction.pdf", transparent=True)
    fig.savefig("docs/source/_static/interaction.svg", transparent=True)

    # plot_3d_rod(points, radius)
