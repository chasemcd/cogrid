"""CoGrid: Multi-agent grid-world environments for reinforcement learning."""

__version__ = "0.4.0"


def make(environment_id: str, **kwargs):
    """Create a pre-registered CoGrid environment by name.

    Triggers registration of all built-in environments on first call.
    """
    from cogrid.envs import registry

    return registry.make(environment_id, **kwargs)
