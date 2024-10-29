import cr_mech_coli as crm
import numpy as np
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == "__main__":
    config = crm.Configuration(
        growth_rate = 0.05,
        t0=0.0,
        dt=0.1,
        t_max=100.0,
        save_interval=20.0,
        n_agents=4,
        domain_size=100,
    )

    cell_container = crm.run_simulation(config)

    all_cells = cell_container.get_cells()
    iterations = cell_container.get_all_iterations()
    colors = cell_container.cell_to_color

    # Pick one iteration to plot results
    iter_masks = [
        (iterations[1], crm.render_mask(config, all_cells[iterations[1]], colors)),
        (iterations[2], crm.render_mask(config, all_cells[iterations[2]], colors)),
    ]
    for iteration, mask in iter_masks:
        positions = crm.extract_positions(mask)
        positions = np.round(np.array(positions))
        positions = np.roll(positions, 1, axis=2)
        positions = np.array(positions, dtype=int).reshape((len(positions), -1, 1, 2))

        dl = 2**0.5 * config.domain_size
        domain_pixels = np.array(mask.shape[:2], dtype=float)
        pixel_per_length = domain_pixels / dl

        # Calculate differences in positions
        fig, ax = plt.subplots(figsize=(6, 6))
        pos_exact = []
        for p1 in positions:
            # Get color
            color = mask[p1[0][0][1], p1[0][0][0]]
            ident = cell_container.get_cell_from_color([*color])
            cell = all_cells[iteration][ident][0]
            p1 = np.array(p1[:,0,:], dtype=float)

            # Shift coordinates to center
            p2 = np.array([0.5 * config.domain_size]*2) - cell.pos[:,:2]
            # Scale with conversion between pixels and length
            p2 *= np.array([-1, 1]) * pixel_per_length
            # Shift coordinate system again
            p2 += 0.5 * domain_pixels
            # Round to plot in image
            p2 = np.array(np.round(p2), dtype=int)

            # Determine if we need to use the reverse order
            d1 = np.sum((p2-p1)**2)
            d2 = np.sum((p2-p1[::-1])**2)
            d = np.sqrt(min(d1, d2)) / len(p2)

            # Compare total length
            l1 = np.sum((p2[1:] - p2[:-1])**2)**0.5
            l2 = np.sum((p1[1:] - p1[:-1])**2)**0.5
            pos_exact.append(p2)

        pos_exact = np.round(np.array(pos_exact)).reshape((len(pos_exact), -1, 1, 2))
        pos_exact = np.array(pos_exact, dtype=int)
        # mask = cv.polylines(mask, positions, False, (50, 50, 50), 2)
        mask = cv.polylines(mask, pos_exact, False, (150, 150, 150), 2)
        mask = cv.polylines(mask, pos_exact, False, (250, 250, 250), 1)
        for p in positions.reshape((-1, 2)):
            mask = cv.drawMarker(mask, p, (50, 50, 50), cv.MARKER_TILTED_CROSS, 14, 2)
        path = Path("docs/source/_static/fitting-methods/")
        cv.imwrite(
            filename=str(path / "extract_positions-{:06}.png".format(iteration)),
            img=mask[200:-200,200:-200],
        )
