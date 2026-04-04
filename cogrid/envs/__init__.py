"""Environment definitions and registry."""

import copy
import functools

from cogrid.cogrid_env import CoGridEnv
from cogrid.core.grid import layouts
from cogrid.envs import registry
from cogrid.envs.overcooked.agent import OvercookedAgent
from cogrid.envs.overcooked.config import (
    build_order_extra_state,
    build_order_hud_fn,
    build_order_tick,
)
from cogrid.envs.overcooked.rewards import (
    DeliveryReward,
    OnionInPotReward,
    OrderGatedIngredientInPotReward,
    SoupInDishReward,
)
from cogrid.envs.overcooked.rewards import (
    ExpiredOrderPenalty as ExpiredOrderPenalty,
)
from cogrid.envs.overcooked.rewards import (
    OnionSoupDeliveryReward as OnionSoupDeliveryReward,
)
from cogrid.envs.overcooked.rewards import (
    OrderDeliveryReward as OrderDeliveryReward,
)
from cogrid.envs.overcooked.rewards import (
    OrderGatedSoupInDishReward as OrderGatedSoupInDishReward,
)

layouts.register_layout(
    "overcooked_cramped_room_v0",
    [
        "CCUCC",
        "O   O",
        "C   C",
        "C=C@C",
    ],
)

layouts.register_layout(
    "overcooked_asymmetric_advantages_v0",
    [
        "CCCCCCCCC",
        "O C@COC @",
        "C+  U  +C",
        "C   U   C",
        "CCC=C=CCC",
    ],
)

layouts.register_layout(
    "overcooked_coordination_ring_v0",
    [
        "CCCUC",
        "C   U",
        "= C C",
        "O   C",
        "CO@CC",
    ],
)

layouts.register_layout(
    "overcooked_forced_coordination_v0",
    [
        "CCCUC",
        "O+C U",
        "O C C",
        "= C+C",
    ],
)

layouts.register_layout(
    "overcooked_counter_circuit_v0",
    [
        "CCCUUCCC",
        "C      C",
        "= CCCC @",
        "C      C",
        "CCCOOCCC",
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
    "rewards": [
        DeliveryReward(coefficient=1.0, common_reward=True),
        OnionInPotReward(coefficient=0.1, common_reward=False),
        SoupInDishReward(coefficient=0.3, common_reward=False),
    ],
    "grid": {"layout": "overcooked_cramped_room_v0"},
    "max_steps": 1000,
    "scope": "overcooked",
    "pickupable_types": [
        "onion",
        "onion_soup",
        "plate",
        "tomato",
        "tomato_soup",
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

# -- Partial observability variants --

partial_obs_cramped_room_config = {
    "name": "overcooked",
    "num_agents": 2,
    "n_agents": 2,
    "observable_radius": 1,
    "action_set": "cardinal_actions",
    "features": [
        "local_view",
        "layout_id",
    ],
    "rewards": [
        DeliveryReward(coefficient=1.0, common_reward=True),
        OnionInPotReward(coefficient=0.1, common_reward=False),
        SoupInDishReward(coefficient=0.3, common_reward=False),
    ],
    "grid": {"layout": "overcooked_asymmetric_advantages_v0"},
    "max_steps": 1000,
    "scope": "overcooked",
    "pickupable_types": [
        "onion",
        "onion_soup",
        "plate",
        "tomato",
        "tomato_soup",
    ],
}

registry.register(
    "Overcooked-AsymmetricAdvantages-PartialObs-V0",
    functools.partial(
        CoGridEnv,
        config=partial_obs_cramped_room_config,
        agent_class=OvercookedAgent,
    ),
)

layouts.register_layout(
    "overcooked_mixed_kitchen_v0",
    [
        "CUCCCUC",
        "O+ C +T",
        "=  C  =",
        "C  C  C",
        "CCCCC@C",
    ],
)

_order_cfg = {
    "spawn_probs": {"onion_soup": 0.05, "tomato_soup": 0.05},
    "max_active": 3,
    "time_limit": 100,
}
mixed_kitchen_config = {
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
        "order_observation",
    ],
    "rewards": [
        OrderDeliveryReward(
            coefficient=1.0, common_reward=True, order_time_limit=_order_cfg["time_limit"]
        ),
        OrderGatedIngredientInPotReward(coefficient=0.1, common_reward=False),
        # OrderGatedSoupInDishReward(coefficient=0.3, common_reward=False),
        ExpiredOrderPenalty(penalty=-0.75),
    ],
    "grid": {"layout": "overcooked_mixed_kitchen_v0"},
    "max_steps": 4000,
    "scope": "overcooked",
    "pickupable_types": [
        "onion",
        "onion_soup",
        "plate",
        "tomato",
        "tomato_soup",
    ],
    "orders": _order_cfg,
    "tick_fn": build_order_tick(_order_cfg, recipe_results=["onion_soup", "tomato_soup"]),
    "extra_state_init_fn": functools.partial(build_order_extra_state, _order_cfg),
    "render_hud_fn": build_order_hud_fn(_order_cfg),
}

registry.register(
    "Overcooked-MixedKitchen-V0",
    functools.partial(CoGridEnv, config=mixed_kitchen_config, agent_class=OvercookedAgent),
)

layouts.register_layout(
    "overcooked_cramped_mixed_kitchen_v0",
    [
        "CCUCC",
        "O+ +T",
        "=   =",
        "C   C",
        "CC@CC",
    ],
)
cramped_mixed_kitchen_config = mixed_kitchen_config.copy()
cramped_mixed_kitchen_config["grid"] = {"layout": "overcooked_cramped_mixed_kitchen_v0"}
registry.register(
    "Overcooked-CrampedMixedKitchen-V0",
    functools.partial(
        CoGridEnv,
        config=cramped_mixed_kitchen_config,
        agent_class=OvercookedAgent,
    ),
)

layouts.register_layout(
    "overcooked_order_delivery_v0",
    [
        "CCCCC",
        "C1 2C",
        "C + C",
        "C @ C",
        "CCCCC",
    ],
)

_od_order_cfg = {
    "spawn_probs": {"onion_soup": 0.017, "tomato_soup": 0.017},
    "max_active": 2,
    "time_limit": 200,
}
order_delivery_config = {
    "name": "overcooked",
    "num_agents": 1,
    "action_set": "cardinal_actions",
    "features": [
        "agent_dir",
        "overcooked_inventory",
        "agent_position",
        "can_move_direction",
        "order_observation",
    ],
    "rewards": [
        OrderDeliveryReward(
            coefficient=1.0, common_reward=True, order_time_limit=_od_order_cfg["time_limit"]
        ),
        ExpiredOrderPenalty(penalty=-0.5),
    ],
    "grid": {"layout": "overcooked_order_delivery_v0"},
    "max_steps": 1000,
    "scope": "overcooked",
    "pickupable_types": ["onion_soup", "tomato_soup"],
    "orders": _od_order_cfg,
    "tick_fn": build_order_tick(_od_order_cfg, recipe_results=["onion_soup", "tomato_soup"]),
    "extra_state_init_fn": functools.partial(build_order_extra_state, _od_order_cfg),
    "render_hud_fn": build_order_hud_fn(_od_order_cfg),
}

registry.register(
    "Overcooked-OrderDelivery-V0",
    functools.partial(CoGridEnv, config=order_delivery_config, agent_class=OvercookedAgent),
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
