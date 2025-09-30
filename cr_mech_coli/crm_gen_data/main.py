import argparse
import multiprocessing as mp

import cr_mech_coli as crm


def crm_gen_data_main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-o", "--output-dir", type=str, default="out/crm_gen_data")
    parser.add_argument("-w", "--workers", type=int, default=-1)

    subparsers = parser.add_subparsers(
        help="Generate image-vertex data-pairs to train extraction algorithm"
    )
    parser_a = subparsers.add_parser(
        "extract",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_a.add_argument(
        "-na",
        "--n-agents",
        type=int,
        default=8,
        help="Number of agents",
    )
    parser_a.add_argument(
        "-nv", "--n-vertices", type=int, default=8, help="Number of vertices"
    )
    parser_a.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for placing initial cells",
    )

    pyargs = parser.parse_args()

    if pyargs.workers <= 0:
        pyargs.workers = mp.cpu_count()

    config = crm.Configuration()
    agent_settings = crm.AgentSettings()
    agent_settings.growth_rate *= 4
    agent_settings.mechanics.spring_length *= 8 / pyargs.n_vertices

    agents = crm.generate_agents(
        pyargs.n_agents,
        agent_settings,
        config,
        rng_seed=pyargs.seed,
        n_vertices=pyargs.n_vertices,
    )
    for i, a in enumerate(agents):
        p = a.pos
        p[:, 2] += (i % 2 - 0.5) * config.domain_height / 2.0
        a.pos = p

    container = crm.run_simulation_with_agents(config, agents)

    crm.store_all_images(
        container,
        config.domain_size,
        save_dir=pyargs.output_dir,
        show_progressbar=True,
        workers=pyargs.workers,
    )
