from .cr_mech_coli_rs import counter_to_color, color_to_counter, assign_colors_to_cells
from .simulation import Configuration, sort_cellular_identifiers, CellIdentifier
import pyvista as pv
import numpy as np
import cv2 as cv
import dataclasses
from pathlib import Path
from tqdm import tqdm


@dataclasses.dataclass
class RenderSettings:
    """
    This class contains all settings to render images with
    `pyvista <https://docs.pyvista.org/>_ and `opencv <https://opencv.org>_ to create
    near-realistic microscopic images.
    Many of the values will be given directly to pyvistas :meth:`pyvista.plotter.add_mesh` function.
    Others are used by :mod:`cv2`.
    """
    resolution: int = 1280#: Resolution of the generated image
    diffuse: float = 0.5#: Value Between 0 and 1.
    ambient: float = 0.5#: Value between 0 and 1.
    specular: float = 0.5#: Value between 0 and 1.
    specular_power: float = 10.0#: Value between 0 and 250.
    metallic: float = 1.0#: Value between 0 and 1
    noise: int = 50#: RGB values per pixel
    bg_brightness: int = 100#: Background brightness
    cell_brightness: int = 30#: Brightness of the individual cells
    ssao_radius: int = 50#: Radius for ssao scattering
    kernel_size: int = 30#: Smoothing kernel size
    pbr: bool = True#: Enable physics-based rendering
    lighting: bool = True# Enable lighting in 3D render
    render_mask: bool = False# If enabled, notifies rendering engine to disable effects

    def prepare_for_masks(self):
        """
        Prepares all render settings such that the image generated is a mask.
        """
        rs = dataclasses.replace(self)
        rs.diffuse = 0
        rs.ambient = 1
        rs.specular = 0
        rs.specular_power = 0
        rs.metallic = 0
        rs.noise = 0
        rs.bg_brightness = 0
        rs.ssao_radius = 0
        rs.kernel_size = 1
        rs.pbr = False
        rs.lighting = False
        rs.render_mask = True
        return rs


def __create_cell_surfaces(cells: dict) -> list:
    cell_surfaces = []
    for ident in cells:
        meshes = []
        p = cells[ident][1].pos.T
        r = cells[ident][1].radius

        meshes.append(pv.Sphere(center=p[:,0], radius=r))
        for j in range(max(p.shape[1]-1,0)):
            # Add sphere at junction
            meshes.append(pv.Sphere(center=p[:,j+1], radius=r))

            # Otherwise add cylinders
            pos1 = p[:,j]
            pos2 = p[:,j+1]
            center = 0.5 * (pos1 + pos2)
            direction = pos2 - center
            height = float(np.linalg.norm(pos1 - pos2))
            cylinder = pv.Cylinder(center, direction, r, height)
            meshes.append(cylinder)
        # Combine all together
        mesh = pv.MultiBlock(meshes).extract_geometry()
        cell_surfaces.append((ident, mesh))
    return cell_surfaces


def render_pv_image(
        config: Configuration,
        cells,
        render_settings: RenderSettings,
        colors: dict | None = None,
        filename: str | Path | None = None,
    ) -> np.ndarray:
    """
    Creates a 3D render of the given cells.

    Args:
        config (Configuration): The configuration used to run the simulation.
        cells: A iterable which contains all cells at a specific iteration.
        render_settings (RenderSettings): Contains all settings to specify how to render image.
        colors (dict): A dictionary mapping a :class:`CellIdentifier` to a color.
            If not given use color from `render_settings`.
        filename: Name of the file in which to save the image. If not specified, do
            not save.

    Returns:
        np.ndarray: An array of shape `(resolution, resolution, 3)` which contains the rendered
            pixels.
    """
    plotter = pv.Plotter(off_screen=True, window_size=[render_settings.resolution]*2)
    plotter.enable_parallel_projection()
    plotter.set_background([render_settings.bg_brightness]*3)

    cell_surfaces = __create_cell_surfaces(cells)

    for ident, cell in cell_surfaces:
        color = [render_settings.cell_brightness]*3
        if colors is not None:
            color = colors[ident]
        plotter.add_mesh(
            cell,
            show_edges=False,
            color=color,
            diffuse=render_settings.diffuse,
            ambient=render_settings.ambient,
            specular=render_settings.specular,
            specular_power=render_settings.specular_power,
            metallic=render_settings.metallic,
            pbr=render_settings.pbr,
            lighting=render_settings.lighting,
        )
    dx = config.domain_size
    plotter.camera.position = (0.5*dx, -0.5*dx, 5*dx)
    plotter.camera.focal_point = (0.5*dx, 0.5*dx, 0)

    if not render_settings.render_mask:
        plotter.enable_ssao(radius=render_settings.ssao_radius)
        plotter.enable_anti_aliasing()
    else:
        plotter.disable_anti_aliasing()
    img = np.array(plotter.screenshot())
    plotter.close()

    if filename is not None:
        # Check that folder exist and if not create them
        odir = Path(filename).parents[0]
        odir.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(filename), img)

    return img

def render_mask(
        config: Configuration,
        cells: dict,
        colors: dict,
        render_settings: RenderSettings | None = None,
        filename: str | Path | None = None,
    ) -> np.ndarray:
    """
    Creates an image containing masks of the given cells.
    This function internally uses the :func:`render_pv_image` function and
    :meth:`RenderSettings.prepare_for_masks` method.

    Args:
        config (Configuration): See :func:`render_pv_image`.
        cells: See :func:`render_pv_image`.
        render_settings (RenderSettings): See :func:`render_pv_image`.
        colors (dict): See :func:`render_pv_image`.
        filename: See :func:`render_pv_image`.

    Returns:
        np.ndarray: See :func:`render_pv_image`.
    """
    if render_settings is None:
        render_settings = RenderSettings()
    rs = render_settings.prepare_for_masks()
    img = render_pv_image(config, cells, rs, colors, filename=filename)
    return img


def render_image(
        config: Configuration,
        cells: dict,
        render_settings: RenderSettings | None = None,
        filename: str | Path | None = None
    ) -> np.ndarray:
    """
    Aims to create a near-realistic microscopic image with the given cells.
    This function internally uses the :func:`render_pv_image` function but changes some of the 

    Args:
        config (Configuration): See :func:`render_pv_image`.
        cells: See :func:`render_pv_image`.
        render_settings (RenderSettings): See :func:`render_pv_image`.
        colors (dict): See :func:`render_pv_image`.
        filename: See :func:`render_pv_image`.

    Returns:
        np.ndarray: See :func:`render_pv_image`.
    """
    if render_settings is None:
        render_settings = RenderSettings()
    img = render_pv_image(config, cells, render_settings)

    # Smoothen it out
    kernel = np.ones([render_settings.kernel_size]*2, np.float32) / render_settings.kernel_size**2
    img = cv.filter2D(img, -1, kernel)

    # Make noise on image
    rng = np.random.default_rng()
    noise = rng.integers(0, render_settings.noise, img.shape[:2], dtype=np.uint8, endpoint=True)
    for j in range(3):
        img[:,:,j] = cv.add(img[:,:,j], noise)

    if filename is not None:
        # Check that folder exist and if not create them
        odir = Path(filename).parents[0]
        odir.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(filename), img)

    return img


def store_all_images(
        config: Configuration,
        sim_result: dict,
        render_settings: RenderSettings | None = None,
        save_dir: str | Path = "out",
        use_hash: bool = True,
        render_raw_pv: bool = False,
        show_progressbar: bool | int = False,
        store_config: bool = True,
    ):
    """
    Combines multiple functions and renders images to files for a complete simulation result.
    This function calls the :func:`render_image`, :func:`render_pv_image` and
    :func:`render_mask` functions to create multiple images.

    Args:
        config (Configuration): See :func:`render_pv_image`.
        sim_result: See :func:`cr_mech_coli.simulation.run_simulation`.
        render_settings (RenderSettings): See :func:`render_pv_image`.
        save_dir: Path of the directory where to save all images.
        use_hash (bool): Use a hash generated from the :class:`Configuration` class as subfolder of
            `save_dir` to store results in.
        render_raw_pv (bool): Additionaly render the intermediate image before applying effects
            from :mod:`cv2`.
        show_progressbar (bool): Shows a progressbar of how many iterations have been rendered.
        store_config (bool): Store config as json string in directory.

    Returns:
        np.ndarray: See :func:`render_pv_image`.
    """
    if render_settings is None:
        render_settings = RenderSettings()
    colors = assign_colors_to_cells(sim_result)

    colors = assign_colors_to_cells(sim_result)
    iterations = sorted(sim_result.keys())

    if use_hash:
        sim_hash = config.to_hash()
        save_dir = Path(save_dir) / "{:020}/".format(sim_hash)

    if store_config:
        config_string = config.to_json()
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "config.json", "w") as f:
            f.write(config_string)

    if show_progressbar is True:
        iterations = tqdm(iterations, total=len(iterations))
    for iteration in iterations:
        cells = sim_result[iteration]
        render_image(
            config,
            cells,
            render_settings,
            Path(save_dir) / "images/{:09}.png".format(iteration),
        )
        render_mask(
            config,
            cells,
            colors,
            render_settings,
            Path(save_dir) / "masks/{:09}.png".format(iteration),
        )
        if render_raw_pv:
            render_pv_image(
                config,
                cells,
                render_settings,
                colors = None,
                filename=Path(save_dir) / "raw_pv/{:09}.png".format(iteration),
            )
