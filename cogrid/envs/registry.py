from cogrid import cogrid_env


ENVIRONMENT_REGISTRY: dict[str, cogrid_env.CoGridEnv] = {}


def make(environment_id: str, **kwargs) -> cogrid_env.CoGridEnv:
    return ENVIRONMENT_REGISTRY[environment_id](**kwargs)


def register(environment_id: str, env_class: cogrid_env.CoGridEnv) -> None:
    ENVIRONMENT_REGISTRY[environment_id] = env_class
