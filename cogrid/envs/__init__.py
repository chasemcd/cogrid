"""Environment definitions and registry."""

import copy
import functools
import random

from cogrid.cogrid_env import CoGridEnv
from cogrid.core import layouts
from cogrid.envs import registry
from cogrid.envs.overcooked.agent import OvercookedAgent
from cogrid.envs.overcooked.config import overcooked_interaction_fn

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
    "features": [
        "agent_dir",
        "overcooked_inventory",
        "next_to_counter",
        "next_to_pot",
        "object_type_masks",
        "ordered_pot_features",
        "dist_to_other_players",
        "agent_position",
        "can_move_direction",
    ],
    "grid": {"layout": "overcooked_cramped_room_v0"},
    "max_steps": 1000,
    "scope": "overcooked",
    "interaction_fn": overcooked_interaction_fn,
    "pickupable_types": ["onion", "onion_soup", "plate", "tomato", "tomato_soup"],
    "recipes": [
        {
            "ingredients": ["onion", "onion", "onion"],
            "result": "onion_soup",
            "cook_time": 30,
            "reward": 1.0,
        },
        {
            "ingredients": ["tomato", "tomato", "tomato"],
            "result": "tomato_soup",
            "cook_time": 30,
            "reward": 1.0,
        },
    ],
}

registry.register(
    "Overcooked-CrampedRoom-V0",
    functools.partial(CoGridEnv, config=cramped_room_config, agent_class=OvercookedAgent),
)

asymmetric_adv_config = copy.deepcopy(cramped_room_config)
asymmetric_adv_config["grid"]["layout"] = "overcooked_asymmetric_advantages_v0"

registry.register(
    "Overcooked-AsymmetricAdvantages-V0",
    functools.partial(CoGridEnv, config=asymmetric_adv_config, agent_class=OvercookedAgent),
)

coordination_ring_config = copy.deepcopy(cramped_room_config)
coordination_ring_config["grid"]["layout"] = "overcooked_coordination_ring_v0"

registry.register(
    "Overcooked-CoordinationRing-V0",
    functools.partial(CoGridEnv, config=coordination_ring_config, agent_class=OvercookedAgent),
)

forced_coordination_config = copy.deepcopy(cramped_room_config)
forced_coordination_config["grid"]["layout"] = "overcooked_forced_coordination_v0"

registry.register(
    "Overcooked-ForcedCoordination-V0",
    functools.partial(
        CoGridEnv,
        config=forced_coordination_config,
        agent_class=OvercookedAgent,
    ),
)

counter_circuit_config = copy.deepcopy(cramped_room_config)
counter_circuit_config["grid"]["layout"] = "overcooked_counter_circuit_v0"

registry.register(
    "Overcooked-CounterCircuit-V0",
    functools.partial(CoGridEnv, config=counter_circuit_config, agent_class=OvercookedAgent),
)


sa_overcooked_config = copy.deepcopy(cramped_room_config)
sa_overcooked_config["num_agents"] = 1
registry.register(
    "Overcooked-CrampedRoom-SingleAgent-V0",
    functools.partial(
        CoGridEnv,
        config=sa_overcooked_config,
        agent_class=OvercookedAgent,
    ),
)


sr_config = {
    "name": "search_rescue",
    "num_agents": 2,
    "action_set": "cardinal_actions",
    "features": [
        "agent_dir",
        "agent_position",
        "can_move_direction",
        "inventory",
    ],
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
    functools.partial(CoGridEnv, config=sr_config),
)
