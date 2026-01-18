import numpy as np
from itertools import repeat
import warnings
import multiprocessing as mp
import scipy as sp
from scipy.stats import qmc
from tqdm import tqdm

from .predict import objective_function


def __lhs_optim_func(args, polish=False):
    ((sample, bounds, oargs), pyargs, callback) = args
    res = sp.optimize.minimize(
        objective_function,
        x0=sample,
        method=pyargs.local_method if not polish else pyargs.polish_method,
        bounds=bounds,
        args=oargs,
        options={
            "disp": False,
            "maxiter": pyargs.local_maxiter if not polish else pyargs.polish_maxiter,
            "maxfev": pyargs.local_maxiter if not polish else pyargs.polish_maxiter,
        },
        callback=callback,
    )
    return res.x, res.fun


def minimize_lhs(params, bounds, args, callback, pyargs):
    bounds = np.array(bounds)
    print(bounds.shape)
    print(len(params))

    sampler = qmc.LatinHypercube(d=len(params))
    sample = sampler.random(pyargs.profiles_lhs_sample_size)
    sample = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
    sample = np.array([*sample, params])

    arglist = [((s, bounds, args), pyargs, callback) for s in sample]

    pool = mp.Pool(pyargs.workers)
    results = list(pool.imap(__lhs_optim_func, arglist))
    ind = np.argmin([r[1] for r in results])

    final_parameters = sample[ind]
    final_cost = results[ind]

    if not pyargs.polish_skip:
        final_parameters, final_cost = __lhs_optim_func(
            ((final_parameters, bounds, args), pyargs, callback)
        )

    return final_parameters, final_cost


def minimize_de(params, bounds, args, callback, pyargs):
    res = sp.optimize.differential_evolution(
        objective_function,
        x0=params,
        bounds=bounds,
        args=args,
        disp=True,
        maxiter=pyargs.maxiter,
        popsize=pyargs.popsize,
        mutation=(pyargs.mutation_lower, pyargs.mutation_upper),
        recombination=pyargs.recombination,
        tol=pyargs.tol,
        workers=pyargs.workers,
        updating="deferred",
        polish=False,
        init="latinhypercube",
        strategy="best1bin",
        callback=callback,
    )
    if not pyargs.skip_polish:
        res = sp.optimize.minimize(
            objective_function,
            x0=res.x,
            method="Nelder-Mead",
            bounds=bounds,
            args=args,
            options={
                "disp": False,
                "maxiter": pyargs.polish_maxiter,
                "maxfev": pyargs.polish_maxiter,
            },
        )
    return res.x, res.fun


def __optimize_around_single(params, param_single, n, args):
    all_params = np.array([*params[:n], param_single, *params[n:]])
    return objective_function(all_params, *args, print_costs=False)


def __calculate_single_cost(optargs):
    n, p, parameters, bounds, args, pyargs = optargs
    index = np.arange(len(parameters)) != n
    x0 = np.array(parameters)[index]
    bounds_reduced = np.array(bounds)[index]

    assert len(x0) + 1 == len(parameters)
    assert len(bounds_reduced) + 1 == len(bounds)

    if pyargs.profiles_optim_method == "differential_evolution":
        res = sp.optimize.differential_evolution(
            __optimize_around_single,
            x0=x0,
            bounds=bounds_reduced,
            args=(p, n, args),
            # popsize=pyargs.popsize,
            # mutation=(pyargs.mutation_lower, pyargs.mutation_upper),
            # recombination=pyargs.recombination,
            # tol=pyargs.tol,
            init="latinhypercube",
            strategy="best1bin",
            maxiter=pyargs.profiles_maxiter,
            workers=1,
            disp=False,
        )
    else:
        res = sp.optimize.minimize(
            __optimize_around_single,
            x0=x0,
            method="Nelder-Mead",
            bounds=bounds_reduced,
            args=(p, n, args),
            options={
                "disp": True,
                "maxiter": pyargs.profiles_maxiter,
                "maxfev": pyargs.profiles_maxiter,
            },
        )
    all_params = np.array([*res.x[:n], p, *res.x[n:]])
    costs = objective_function(
        all_params, *args, print_costs=False, return_split_cost=True
    )
    return res.x, costs


def calculate_profiles(parameters, bounds, n_samples, args, pyargs):
    b_low = np.array(bounds)[:, 0]
    b_high = np.array(bounds)[:, 1]
    n_param = np.repeat([np.arange(len(parameters))], n_samples, axis=0)
    samples = np.linspace(b_low, b_high, n_samples)

    arglist = list(
        zip(
            n_param.flatten(),
            samples.flatten(),
            repeat(parameters),
            repeat(bounds),
            repeat(args),
            repeat(pyargs),
        )
    )

    # Ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pool = mp.Pool(pyargs.workers)
        results = list(
            tqdm(
                pool.imap(__calculate_single_cost, arglist, chunksize=1),
                total=samples.size,
            )
        )

    costs = np.zeros((samples.size, 3))
    for i, (_, split_costs) in zip(range(samples.size), results):
        print(split_costs)
        if type(split_costs) is not tuple:
            costs[i] = np.nan
        else:
            costs[i] = np.array(split_costs)

    costs = costs.reshape((*samples.shape, 3))

    return samples, costs
