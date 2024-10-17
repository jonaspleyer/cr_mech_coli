from .cr_mech_coli_rs import Configuration, sort_cellular_identifiers
import pyvista as pv
import numpy as np
import cv2 as cv
import dataclasses
from pathlib import Path
from tqdm import tqdm


@dataclasses.dataclass
class RenderSettings:
    resolution: int = 1280
    # Between 0 and 1
    diffuse: float = 0.5
    # Between 0 and 1
    ambient: float = 0.5
    # Between 0 and 1
    specular: float = 0.5
    # Between 0 and 250. See pyvista add_mesh.
    specular_power: float = 10.0
    # Between 0 and 1
    metallic: float = 1.0
    # RGB values per pixel
    noise: int = 25
    # Background brightness
    bg_brightness: int = 100
    cell_brightness: int = 30
    ssao_radius: int = 12
    # For smoothing of image
    kernel_size: int = 10
    # Enable physics-based rendering
    pbr: bool = True

    def prepare_for_masks(self):
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
        return rs


def counter_to_color(counter: int, artistic: bool = False) -> list[int]:
    if artistic:
        counter = (counter * 157 * 163 * 173) % 255**3
    color = [0, 0, 0]
    q, mod = divmod(counter, 255**2)
    color[0] = q
    q, mod = divmod(mod, 255)
    color[1] = q
    color[2] = mod
    return color


def assign_colors_to_cells(sim_result: dict, artistic: bool = False) -> dict:
    """This functions assigns unique colors to given cellular identifiers. Note that this
    """
    iterations = sorted(sim_result.keys())
    color_index = 1
    colors = {}
    for i in iterations:
        for ident in sort_cellular_identifiers(list(sim_result[i].keys())):
            if ident not in colors:
                if color_index > 255**3:
                    raise ValueError("The maximum supported number of cells is 255^3")
                color = counter_to_color(color_index, artistic)
                color_index += 1
                colors[ident] = color
    return colors


def __create_cell_surfaces(cells: dict) -> list:
    cell_surfaces = []
    for ident in cells:
        meshes = []
        p = cells[ident][0].pos.T
        r = cells[ident][0].radius

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
        )
    dx = config.domain_size
    plotter.camera.position = (0.5*dx, -0.5*dx, 5*dx)
    plotter.camera.focal_point = (0.5*dx, 0.5*dx, 0)

    plotter.enable_ssao(radius=render_settings.ssao_radius)
    plotter.enable_anti_aliasing()
    img = plotter.screenshot()
    plotter.close()
    return np.array(img)

def render_mask(
        config: Configuration,
        cells: dict,
        colors: dict,
        render_settings: RenderSettings | None = None,
        filename: str | Path | None = None,
    ) -> np.ndarray:
    if render_settings is None:
        render_settings = RenderSettings()
    rs = render_settings.prepare_for_masks()
    img = render_pv_image(config, cells, rs, colors)
    if filename is not None:
        cv.imwrite(filename, img)
    return img

def render_image(
        config: Configuration,
        cells: dict,
        render_settings: RenderSettings | None = None,
        filename: str | Path | None = None
    ) -> np.ndarray:
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
        cv.imwrite(filename, img)

    return img
