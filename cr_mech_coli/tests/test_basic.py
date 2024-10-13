import cr_mech_coli as crm

if __name__ == "__main__":
    config = crm.Configuration()
    config.show_progressbar = True
    config.n_agents = 2
    config.t0 = 0.0
    config.dt = 0.1
    config.t_max = 10.0
    config.save_interval = 0.25
    config.agent_settings.mechanics.spring_tension = 0.2
    config.agent_settings.mechanics.damping = 0.2

    try:
        cells = crm.run_simulation(config)
        iterations = sorted(cells.keys())
        for i in iterations:
            for ident in cells[i]:
                p = cells[i][ident].pos
                print(i, p)
    except Exception as error:
        print(error)
