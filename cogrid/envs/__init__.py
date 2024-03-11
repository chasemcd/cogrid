import functools

from cogrid.envs.search_rescue import search_rescue, search_rescue_grid_objects
from cogrid.envs.overcooked import overcooked, overcooked_grid_objects
from cogrid.core import grid_object
from cogrid.envs import registry


sa_overcooked_config = {
    "name": "overcooked",
    "num_agents": 1,
    "action_set": "cardinal_actions",
    "obs": ["overcooked_features"],
    "grid_gen_kwargs": {"load": "sa_overcooked"},
    "max_steps": 1000,
}

registry.register(
    "SAOvercooked-V0",
    functools.partial(overcooked.Overcooked, config=sa_overcooked_config),
)

overcooked_config = {
    "name": "overcooked",
    "num_agents": 2,
    "action_set": "cardinal_actions",
    "obs": ["overcooked_features"],
    "grid_gen_kwargs": {"load": "overcooked-v0"},
    "max_steps": 1000,
    "common_reward": True,
}


registry.register(
    "Overcooked-V0",
    functools.partial(overcooked.Overcooked, config=overcooked_config),
)

overcooked_config = {
    "name": "overcooked",
    "num_agents": 2,
    "action_set": "cardinal_actions",
    "obs": ["overcooked_features"],
    "grid_gen_kwargs": {"load": "overcooked-crampedroom-v0"},
    "max_steps": 1000,
    "common_reward": True,
}


registry.register(
    "Overcooked-CrampedRoom-V0",
    functools.partial(overcooked.Overcooked, config=overcooked_config),
)
