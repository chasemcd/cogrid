import functools
import copy

from cogrid.core import grid_object
from cogrid.envs.overcooked import overcooked
from cogrid.envs import registry
from cogrid.envs.search_rescue import search_rescue
from cogrid.core import layouts


layouts.register_layout(
    "overcooked_cramped_room_v0",
    [
        "#######",
        "#CCUCC#",
        "#O   O#",
        "#C   C#",
        "#C=C@C#",
        "#######",
    ],
)

layouts.register_layout(
    "overcooked_asymmetric_advantages_v0",
    [
        "###########",
        "#CCCCCCCCC#",
        "#O C@COC @#",
        "#C   U   C#",
        "#C   U   C#",
        "#CCC=C=CCC#",
        "###########",
    ],
)

layouts.register_layout(
    "overcooked_coordination_ring_v0",
    [
        "#######",
        "#CCCUC#",
        "#C   U#",
        "#= C C#",
        "#O   C#",
        "#CO@CC#",
        "#######",
    ],
)

layouts.register_layout(
    "overcooked_forced_coordination_v0",
    [
        "#######",
        "#CCCUC#",
        "#O+C U#",
        "#O C C#",
        "#= C+C#",
        "#CCC@C#",
        "#######",
    ],
)

layouts.register_layout(
    "overcooked_counter_circuit_v0",
    [
        "##########",
        "#CCCUUCCC#",
        "#C      C#",
        "#= CCCC @#",
        "#C      C#",
        "#CCCOOCCC#",
        "##########",
    ],
)

cramped_room_config = {
    "name": "overcooked",
    "num_agents": 2,
    "action_set": "cardinal_actions",
    "features": ["overcooked_features"],
    "rewards": ["delivery_reward"],
    "grid": {"layout": "overcooked_cramped_room_v0"},
    "max_steps": 1000,
    "scope": "overcooked",
}

registry.register(
    "Overcooked-CrampedRoom-V0",
    functools.partial(overcooked.Overcooked, config=cramped_room_config),
)

asymmetric_adv_config = copy.deepcopy(cramped_room_config)
asymmetric_adv_config["grid"]["layout"] = "overcooked_asymmetric_advantages_v0"

registry.register(
    "Overcooked-AsymmetricAdvantages-V0",
    functools.partial(overcooked.Overcooked, config=asymmetric_adv_config),
)

coordination_ring_config = copy.deepcopy(cramped_room_config)
coordination_ring_config["grid"]["layout"] = "overcooked_coordination_ring_v0"

registry.register(
    "Overcooked-CoordinationRing-V0",
    functools.partial(overcooked.Overcooked, config=coordination_ring_config),
)

forced_coordination_config = copy.deepcopy(cramped_room_config)
forced_coordination_config["grid"][
    "layout"
] = "overcooked_forced_coordination_v0"

registry.register(
    "Overcooked-ForcedCoordination-V0",
    functools.partial(overcooked.Overcooked, config=forced_coordination_config),
)

counter_circuit_config = copy.deepcopy(cramped_room_config)
counter_circuit_config["grid"]["layout"] = "overcooked_counter_circuit_v0"

registry.register(
    "Overcooked-CounterCircuit-V0",
    functools.partial(overcooked.Overcooked, config=counter_circuit_config),
)


def randomized_layout_fn(np_random=None, **kwargs):
    layout_choices = [
        "overcooked_cramped_room_v0",
        "overcooked_asymmetric_advantages_v0",
        "overcooked_coordination_ring_v0",
        "overcooked_forced_coordination_v0",
        "overcooked_counter_circuit_v0",
    ]
    if np_random is None:
        import random as stdlib_random
        layout_name = stdlib_random.choice(layout_choices)  # Fallback for backwards compat
    else:
        layout_name = np_random.choice(layout_choices)
    return layout_name, *layouts.get_layout(layout_name)


overcooked_randomized_config = copy.deepcopy(cramped_room_config)
overcooked_randomized_config["grid"] = {"layout_fn": randomized_layout_fn}

registry.register(
    "Overcooked-RandomizedLayout-V0",
    functools.partial(
        overcooked.Overcooked, config=overcooked_randomized_config
    ),
)


sa_overcooked_config = copy.deepcopy(cramped_room_config)
sa_overcooked_config["num_agents"] = 1
registry.register(
    "Overcooked-CrampedRoom-SingleAgent-V0",
    functools.partial(
        overcooked.Overcooked,
        config=sa_overcooked_config,
        environment_scope="overcooked",
    ),
)


sr_config = {
    "name": "search_rescue",
    "num_agents": 2,
    "action_set": "cardinal_actions",
    "obs": ["agent_positions"],
    "grid": {"layout": "search_rescue_test"},
    "max_steps": 1000,
    "common_reward": True,
    "scope": "search_rescue",
}

layouts.register_layout(
    "search_rescue_test",
    [
        "##########",
        "#++      #",
        "#        #",
        "#        #",
        "#       G#",
        "#        #",
        "#T     K #",
        "#XX M ##D#",
        "#GX Y #GG#",
        "##########",
    ],
)


registry.register(
    "SearchRescue-Test-V0",
    functools.partial(search_rescue.SearchRescueEnv, config=sr_config),
)
