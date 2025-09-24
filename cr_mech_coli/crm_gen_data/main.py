import argparse

import cr_mech_coli as crm


def crm_gen_data_main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-n",
        type=int,
        default=4,
        help="Number of agents",
    )
    pyargs = parser.parse_args()

    print(f"Simulating {pyargs.n} agents")
