import cr_mech_coli as crm
import cv2 as cv
from pathlib import Path

if __name__ == "__main__":
    config = crm.Configuration()
    config.agent_settings.growth_rate = 0.05
    config.t0 = 0.0
    config.dt = 0.1
    config.t_max = 100.0
    config.save_interval = 20.0
    config.n_agents = 4

    cell_container = crm.run_simulation(config)

    all_cells = cell_container.get_cells()
    iterations = cell_container.get_all_iterations()
    colors = cell_container.cell_to_color
    i1 = iterations[1]
    i2 = iterations[2]

    rs = crm.RenderSettings(resolution=800)
    mask1 = crm.render_mask(config, all_cells[i1], colors, render_settings=rs)
    mask2 = crm.render_mask(config, all_cells[i2], colors, render_settings=rs)
    mask3 = crm.area_diff_mask(mask1, mask2)
    mask4 = crm.parents_diff_mask(mask1, mask2, cell_container)

    # Save first mask
    path = Path("docs/source/_static/fitting/")
    path.mkdir(parents=True, exist_ok=True)
    cv.imwrite(filename=str(path / "progressions-1.png"), img=mask1)
    cv.imwrite(filename=str(path / "progressions-2.png"), img=mask2)
    cv.imwrite(filename=str(path / "progressions-3.png"), img=mask3*255.0)
    cv.imwrite(filename=str(path / "progressions-4.png"), img=mask4*255.0)
