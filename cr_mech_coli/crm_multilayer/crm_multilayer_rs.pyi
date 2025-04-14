import cr_mech_coli as crm

class MultilayerConfig:
    agent_settings: crm.AgentSettings
    config: crm.Configuration
    rng_seed: int
    dx: tuple[float, float]

    def clone_with_args(self, **kwargs) -> MultilayerConfig: ...
    @staticmethod
    def __new__(cls, **kwargs) -> MultilayerConfig: ...
