from cr_mech_coli.crm_amir import run_sim


def crm_amir_main():
    agents = run_sim()

    for i, a in agents:
        print(i, a.agent.pos)
