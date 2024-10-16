import matplotlib.pyplot as plt

def extract_all_identifiers(cells_at_iterations: dict) -> set:
    return set(cells_at_iterations.keys())

def assign_colors_to_cells(cells_at_iterations: dict) -> dict:
    iterations = sorted(cells_at_iterations.keys())
    color_index = 0
    colors = {}
    for i in iterations:
        for ident in cells_at_iterations[i]:
            if ident not in colors:
                color_current = [0, 0, 0]
                q, mod = divmod(color_index, 255**2)
                color_current[0] = q
                q, mod = divmod(mod, 255)
                color_current[1] = q
                color_current[2] = mod
                color_index += 1
                colors[ident] = color_current
    return colors


