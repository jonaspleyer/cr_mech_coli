<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/cr_mech_coli_dark_mode.svg">
        <source media="(prefers-color-scheme: light)" srcset="docs/source_static/cr_mech_coli.svg">
        <img alt="The cellular_raza logo" src="doc/cellular_raza.svg">
    </picture>
</div>

# cr_mech_coli
[![License: GPL 2.0](https://img.shields.io/github/license/jonaspleyer/cr_mech_coli?style=flat-square)](https://opensource.org/license/gpl-2-0/)
[![Test](https://img.shields.io/github/actions/workflow/status/jonaspleyer/cr_mech_coli/test.yml?label=Test&style=flat-square)](https://github.com/jonaspleyer/cr_mech_coli/actions)
[![CI](https://img.shields.io/github/actions/workflow/status/jonaspleyer/cr_mech_coli/CI.yml?label=CI&style=flat-square)](https://github.com/jonaspleyer/cr_mech_coli/actions)
[![Docs](https://img.shields.io/github/actions/workflow/status/jonaspleyer/cr_mech_coli/sphinx_doc.yml?label=Docs&style=flat-square)](https://github.com/jonaspleyer/cr_mech_coli/actions)
[![PyPI - Version](https://img.shields.io/pypi/v/cr_mech_coli?style=flat-square)]()

Find the documentation of this package under
[jonaspleyer.github.io/cr_mech_coli/](https://jonaspleyer.github.io/cr_mech_coli/).

## Example

```python
import cr_mech_coli as crm

# Contains settings regarding simulation domain, time increments etc.
config = crm.Configuration()

# Use predefined values for agents
agent_settings = crm.AgentSettings()

# Automatically generate positions
positions = crm.generate_positions_old(
    4,
    agent_settings,
    config
)

# Create new agents
agents = [
    crm.RodAgent(
        pos=p,
        vel=p*0.0,
        **agent_settings.to_rod_agent_dict()
    )
    for p in positions
]

# Run simulation and return container
cell_container = crm.run_simulation_with_agents(agents, config)

# Plot individual results
crm.store_all_images(cell_container, config.domain_size)
```

The generated images will be stored (by default) in `out`.
```bash
out
├── images
│   ├── 000000100.png
│   ...
│   └── 000001000.png
├── masks
│   ├── 000000100.png
│   ...
│   └── 000001000.png
```
