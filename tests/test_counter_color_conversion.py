import cr_mech_coli as crm
import numpy as np
import cv2 as cv
from tqdm import tqdm

def test_counter_color_conversion():
    for counter in range(1, 251**3, 13):
        color = crm.counter_to_color(counter)
        counter_new = crm.color_to_counter(color)
        assert counter == counter_new

def test_color_counter_conversion():
    for i in range(1, 251, 2):
        for j in range(1, 251, 3):
            for k in range(1, 251, 5):
                color = [i,j,k]
                counter = crm.color_to_counter(color)
                color_new = crm.counter_to_color(counter)
                assert color == color_new

def test_assign_colors():
    config = crm.Configuration()
    sim_result = crm.run_simulation(config)

    cell_to_color = crm.assign_colors_to_cells(sim_result)
    iterations = sim_result.get_all_iterations()
    cells = sim_result.get_cells()
    cells = cells[iterations[0]]
    img = crm.render_mask(config, cells, cell_to_color)

    all_colors = set()
    all_counters = set()
    for i in tqdm(range(img.shape[0])):
        for j in range(img.shape[1]):
            color = img[i,j,:]
            if np.sum(color) > 0:
                color = (color[0], color[1], color[2])
                if color not in all_colors:
                    all_colors.add(color)
                    counter = crm.color_to_counter(list(color))
                    all_counters.add(counter)
    for i, cell in enumerate(cells):
        color_expected = tuple(crm.counter_to_color(i+1))
        assert color_expected in all_colors
        cell_to_color[cell]
