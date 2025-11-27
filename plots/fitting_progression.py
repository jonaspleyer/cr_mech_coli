import cr_mech_coli as crm
import cv2 as cv
from pathlib import Path
from PIL import Image
import numpy as np

from fitting_extract_positions import create_simulation_result


def store_image(mask, path, name):
    image_rgb = cv.cvtColor(mask.astype(np.uint8), cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    pil_img.save(str(path / name) + ".pdf")
    cv.imwrite(filename=str(path / name) + ".png", img=mask)


if __name__ == "__main__":
    config, cell_container = create_simulation_result(8)

    all_cells = cell_container.get_cells()
    iterations = cell_container.get_all_iterations()
    colors = cell_container.cell_to_color
    i1 = iterations[25]
    i2 = iterations[35]

    color_to_cell = cell_container.color_to_cell
    parent_map = cell_container.parent_map

    rs = crm.RenderSettings()
    mask1 = crm.render_mask(all_cells[i1], colors, config.domain_size, rs)
    mask2 = crm.render_mask(all_cells[i2], colors, config.domain_size, rs)
    mask3 = crm.area_diff_mask(mask1, mask2)
    mask4 = crm.parents_diff_mask(mask1, mask2, color_to_cell, parent_map, 0.5)

    # Save first mask
    path = Path("docs/source/_static/fitting-methods/")
    path.mkdir(parents=True, exist_ok=True)

    store_image(mask1, path, "progressions-1")
    store_image(mask2, path, "progressions-2")
    store_image(mask3 * 255.0, path, "progressions-3")
    store_image(mask4 * 255.0, path, "progressions-4")
