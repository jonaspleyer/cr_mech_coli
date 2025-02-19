import cr_mech_coli as crm
import cv2 as cv
from pathlib import Path
import numpy as np

from fitting_extract_positions import create_simulation_result

if __name__ == "__main__":
    config, cell_container = create_simulation_result(8)

    all_cells = cell_container.get_cells()
    iterations = cell_container.get_all_iterations()
    colors = cell_container.cell_to_color
    i1 = iterations[1]
    i2 = iterations[2]

    rs = crm.RenderSettings(resolution=800)
    mask1 = crm.render_mask(all_cells[i1], colors, config.domain_size, render_settings=rs)
    mask2 = crm.render_mask(all_cells[i2], colors, config.domain_size, render_settings=rs)
    mask3 = crm.area_diff_mask(mask1, mask2)
    mask4 = crm.parents_diff_mask(mask1, mask2, cell_container)

    # Save first mask
    path = Path("docs/source/_static/fitting-methods/")
    path.mkdir(parents=True, exist_ok=True)
    cv.imwrite(filename=str(path / "progressions-1.png"), img=mask1[100:-100,100:-100])
    cv.imwrite(filename=str(path / "progressions-2.png"), img=mask2[100:-100,100:-100])
    cv.imwrite(filename=str(path / "progressions-3.png"), img=mask3[100:-100,100:-100]*255.0)
    cv.imwrite(filename=str(path / "progressions-4.png"), img=mask4[100:-100,100:-100]*255.0)

