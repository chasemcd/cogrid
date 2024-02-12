from cogrid.envs import registry
from cogrid import gridworld_env


def make(environment_id: str, **kwargs) -> gridworld_env.GridWorld:
    return registry.ENVIRONMENT_REGISTRY[environment_id](**kwargs)


def register(environment_id: str, env_class: gridworld_env.GridWorld) -> None:
    registry.ENVIRONMENT_REGISTRY[environment_id] = env_class
