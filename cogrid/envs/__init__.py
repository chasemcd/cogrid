import functools

from cogrid.core import grid_object
from cogrid.envs.overcooked import overcooked
from cogrid.envs import registry
from cogrid.envs.search_rescue import search_rescue

sa_overcooked_config = {
    "name": "overcooked",
    "num_agents": 1,
    "action_set": "cardinal_actions",
    "features": ["overcooked_features"],
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
    "features": ["overcooked_features"],
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
    "features": ["overcooked_features"],
    "rewards": ["delivery_reward"],
    "grid_gen_kwargs": {"load": "overcooked-crampedroom-v0"},
    "max_steps": 1000,
}


registry.register(
    "Overcooked-CrampedRoom-V0",
    functools.partial(overcooked.Overcooked, config=overcooked_config),
)


sr_config = {
    "name": "search_rescue",
    "num_agents": 2,
    "action_set": "cardinal_actions",
    "obs": ["agent_positions"],
    "grid_gen_kwargs": {"load": "item_map"},
    "max_steps": 1000,
    "common_reward": True,
}


registry.register(
    "SearchRescue-Items-V0",
    functools.partial(search_rescue.SearchRescueEnv, config=sr_config),
)
