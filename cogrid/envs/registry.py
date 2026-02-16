from cogrid import cogrid_env

ENVIRONMENT_REGISTRY: dict[str, cogrid_env.CoGridEnv] = {}


def make(environment_id: str, environment_scope: str = "global", **kwargs) -> cogrid_env.CoGridEnv:
    if environment_id not in ENVIRONMENT_REGISTRY:
        raise ValueError(
            f"Environment ID {environment_id} is not registered. "
            f"Please register it first. \n "
            f"Available environments: {list(ENVIRONMENT_REGISTRY.keys())}"
        )
    return ENVIRONMENT_REGISTRY[environment_id](environment_scope=environment_scope, **kwargs)


def register(environment_id: str, env_class: cogrid_env.CoGridEnv) -> None:
    ENVIRONMENT_REGISTRY[environment_id] = env_class
