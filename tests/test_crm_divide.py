from pathlib import Path
import numpy as np

from cr_mech_coli import crm_divide as crd


def test_default_parameters():
    x0, bounds = crd.default_parameters()
    for x0, (low, high) in zip(x0, bounds):
        assert x0 >= low
        assert x0 <= high


def test_basic_predict():
    x0, _ = crd.default_parameters()
    masks_data, positions_all, settings, iterations_data = crd.preprocessing(
        data_dir=Path("data/crm_divide/test")
    )
    container = crd.predict(
        x0,
        positions_all[0],
        settings,
    )

    iterations = container.get_all_iterations()
    cells = container.get_cells_at_iteration(iterations[-1])

    # Ensure that we have 6 final cells
    assert len(cells) == 6
    # Ensure that this matches the number of colors
    assert len(np.unique(masks_data[-1])) - 1 == len(cells)
    # Ensure that the number of save points is correct and matches the provided data
    # Data points
    # 1044 ____ 1046 1047
    #    0    1    2    3
    assert len(iterations) == 4
    assert len(iterations) == max(iterations_data) + 1


def test_basic_objective_function():
    x0, _ = crd.default_parameters()
    masks_data, positions_all, settings, iterations_data = crd.preprocessing(
        data_dir=Path("data/crm_divide/test")
    )
    cost = crd.objective_function(
        x0,
        positions_all,
        settings,
        masks_data,
        iterations_data,
    )
    assert type(cost) is np.float64
    assert cost > 10


def test_objective_function_return_timings():
    x0, _ = crd.default_parameters()
    masks_data, positions_all, settings, iterations_data = crd.preprocessing(
        data_dir=Path("data/crm_divide/test")
    )
    timings = crd.objective_function(
        x0,
        positions_all,
        settings,
        masks_data,
        iterations_data,
        return_timings=True,
    )

    times = np.array([t[0] for t in timings])
    times = times[1:] - times[:-1]
    total = np.sum(times)

    for i, (_, name) in enumerate(timings[1:]):
        n = name.split("\n")[0]
        print(
            f"{n:20} time: {times[i] / 1e6:10.5f} ms percentage: {times[i] / total * 100:7.2f} %"
        )

    assert type(timings) is list
    assert type(timings[0]) is tuple
    assert type(timings[0][0]) is int
    assert type(timings[0][1]) is str
    assert len(timings) == 7
