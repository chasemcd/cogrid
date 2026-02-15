def make(environment_id: str, **kwargs):
    """Create a pre-registered CoGrid environment.

    Convenience wrapper around the environment registry. Triggers
    registration of all built-in environments on first call.

    Args:
        environment_id: Registered environment name
            (e.g. ``"Overcooked-CrampedRoom-V0"``).
        **kwargs: Forwarded to the environment constructor
            (e.g. ``backend="jax"``, ``render_mode="human"``).

    Returns:
        A :class:`CoGridEnv` instance.
    """
    import cogrid.envs  # noqa: F401 -- trigger registration
    from cogrid.envs import registry

    return registry.make(environment_id, **kwargs)