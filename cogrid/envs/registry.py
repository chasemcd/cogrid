from cogrid import gridworld_env


ENVIRONMENT_REGISTRY: dict[str, gridworld_env.GridWorld] = {}


def make(environment_id: str, **kwargs) -> gridworld_env.GridWorld:
    return ENVIRONMENT_REGISTRY[environment_id](**kwargs)


def register(environment_id: str, env_class: gridworld_env.GridWorld) -> None:
    ENVIRONMENT_REGISTRY[environment_id] = env_class
