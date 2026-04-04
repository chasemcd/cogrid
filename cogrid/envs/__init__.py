"""Environment definitions and registry."""

import copy
import functools

from cogrid.cogrid_env import CoGridEnv
from cogrid.core.grid import layouts
from cogrid.envs import registry
from cogrid.envs.overcooked.agent import OvercookedAgent
from cogrid.envs.overcooked.config import (
    build_branch_activate_button,
    build_branch_flag_delivery,
    build_order_extra_state,
    build_order_hud_fn,
    build_order_tick,
    build_target_recipe_extra_state,
    build_target_recipe_tick,
)
from cogrid.envs.overcooked.rewards import (
    ButtonActivationCost,
    DeliveryReward,
    OnionInPotReward,
    OrderGatedIngredientInPotReward,
    SoupInDishReward,
    TargetRecipeDeliveryReward,
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


# ═══════════════════════════════════════════════════════════════════════════
# OvercookedV2 benchmark environments (Gessler et al., 2025)
# ═══════════════════════════════════════════════════════════════════════════
#
# Six environments testing coordination under asymmetric information.
# Pots accept any ingredient; correctness is checked at delivery time.
# Character mapping from JaxMARL:
#   W→C  A→+  X→X(open_delivery_zone)  B(plate)→=  P→u(open_pot)
#   R→R(recipe_indicator)  L→L(button_indicator)
#   0→O(onion_stack)  1→T(tomato_stack)  2→B(broccoli_stack)  3→M(mushroom_stack)

# -- Layouts --

layouts.register_layout(
    "overcooked_v2_grounded_coord_simple",
    [
        "CCBCCCCC",
        "C  C=  O",
        "R +Lu+ X",
        "C  C=  T",
        "CCBCCCCC",
    ],
)

layouts.register_layout(
    "overcooked_v2_grounded_coord_ring",
    [
        "CCCBRBCCC",
        "C       C",
        "C CCLCC C",
        "B O   = B",
        "R+X+u + R",
        "B T   = B",
        "C CCLCC C",
        "C       C",
        "CCCBRBCCC",
    ],
)

layouts.register_layout(
    "overcooked_v2_test_time_simple",
    [
        "CCBCCCCC",
        "C  C=  O",
        "R +Cu+ X",
        "C  C=  T",
        "CCBCCCCC",
    ],
)

layouts.register_layout(
    "overcooked_v2_test_time_wide",
    [
        "CCX=CC",
        "O +  O",
        "T    T",
        "CuCuCC",
        "M +  M",
        "C    C",
        "CCRCCC",
    ],
)

layouts.register_layout(
    "overcooked_v2_demo_cook_simple",
    [
        "CCCCCRBCoCC",
        "O      C  =",
        "C     +u+ X",
        "T      C  =",
        "CCCCCRBCtCC",
    ],
)

layouts.register_layout(
    "overcooked_v2_demo_cook_wide",
    [
        "CCCC=X=CCCC",
        "CCCO + TCCC",
        "CCCCCuCCCCC",
        "C    +    C",
        "O  CMRMC  O",
        "CTCCCCCCCTC",
    ],
)

# -- Shared V2 base config --
# V2 types are registered lazily: the first call to registry.make() for
# a V2 environment triggers the import, which registers OpenPot, indicators,
# and mixed soup types in the overcooked scope.  This avoids polluting
# the standard Pot's autowire.

_v2_types_loaded = False


def _ensure_v2_types():
    global _v2_types_loaded
    if not _v2_types_loaded:
        import cogrid.envs.overcooked.v2_objects  # noqa: F401

        _v2_types_loaded = True


# Defer OPEN_POT_SOUP_NAMES resolution -- compute a static list now
# based on known ingredient combos (must match v2_objects.build_open_pot_recipes).
def _v2_soup_names():
    """Compute all soup result names without importing v2_objects."""
    from itertools import combinations_with_replacement

    names = set()
    ingredients = ["onion", "tomato", "broccoli", "mushroom"]
    for combo in combinations_with_replacement(sorted(ingredients), 3):
        sc = sorted(combo)
        if sc == ["onion", "onion", "onion"]:
            names.add("onion_soup")
        elif sc == ["tomato", "tomato", "tomato"]:
            names.add("tomato_soup")
        else:
            names.add("soup_" + "_".join(sc))
    return sorted(names)


_V2_SOUP_NAMES = _v2_soup_names()


def _make_v2_env(config, **kwargs):
    """Factory for V2 environments that lazily registers V2 types."""
    _ensure_v2_types()
    return CoGridEnv(config=config, agent_class=OvercookedAgent, **kwargs)


_v2_base_config = {
    "name": "overcooked",
    "num_agents": 2,
    "n_agents": 2,
    "action_set": "cardinal_actions",
    "observable_radius": 2,
    "max_steps": 400,
    "scope": "overcooked",
    "features": ["local_view"],
    "pickupable_types": [
        "onion",
        "tomato",
        "broccoli",
        "mushroom",
        "plate",
        *_V2_SOUP_NAMES,
    ],
    "target_recipes": ["onion_soup", "tomato_soup"],
    "resample_on_delivery": True,
}

# -- Grounded Coordination: R + L (button costs -5), incorrect = -20 --

_gc_simple_config = copy.deepcopy(_v2_base_config)
_gc_simple_config["grid"] = {"layout": "overcooked_v2_grounded_coord_simple"}
_gc_simple_config["interactions"] = [
    build_branch_activate_button(activation_time=10),
    build_branch_flag_delivery(),
]
_gc_simple_config["rewards"] = [
    TargetRecipeDeliveryReward(coefficient=20.0, common_reward=True, penalize_incorrect=True),
    ButtonActivationCost(cost=5.0),
]
_gc_simple_config["tick_fn"] = build_target_recipe_tick(_gc_simple_config)
_gc_simple_config["extra_state_init_fn"] = functools.partial(
    build_target_recipe_extra_state,
    _gc_simple_config,
)

registry.register(
    "OvercookedV2-GroundedCoordSimple-V0",
    functools.partial(_make_v2_env, config=_gc_simple_config),
)

_gc_ring_config = copy.deepcopy(_gc_simple_config)
_gc_ring_config["grid"] = {"layout": "overcooked_v2_grounded_coord_ring"}
_gc_ring_config["tick_fn"] = build_target_recipe_tick(_gc_ring_config)
_gc_ring_config["extra_state_init_fn"] = functools.partial(
    build_target_recipe_extra_state,
    _gc_ring_config,
)

registry.register(
    "OvercookedV2-GroundedCoordRing-V0",
    functools.partial(_make_v2_env, config=_gc_ring_config),
)

# -- Test-Time Protocol: R only, incorrect = -20 --

_ttp_simple_config = copy.deepcopy(_v2_base_config)
_ttp_simple_config["grid"] = {"layout": "overcooked_v2_test_time_simple"}
_ttp_simple_config["interactions"] = [build_branch_flag_delivery()]
_ttp_simple_config["rewards"] = [
    TargetRecipeDeliveryReward(coefficient=20.0, common_reward=True, penalize_incorrect=True),
]
_ttp_simple_config["tick_fn"] = build_target_recipe_tick(_ttp_simple_config)
_ttp_simple_config["extra_state_init_fn"] = functools.partial(
    build_target_recipe_extra_state,
    _ttp_simple_config,
)

registry.register(
    "OvercookedV2-TestTimeSimple-V0",
    functools.partial(_make_v2_env, config=_ttp_simple_config),
)

_ttp_wide_config = copy.deepcopy(_ttp_simple_config)
_ttp_wide_config["grid"] = {"layout": "overcooked_v2_test_time_wide"}
_ttp_wide_config["tick_fn"] = build_target_recipe_tick(_ttp_wide_config)
_ttp_wide_config["extra_state_init_fn"] = functools.partial(
    build_target_recipe_extra_state,
    _ttp_wide_config,
)

registry.register(
    "OvercookedV2-TestTimeWide-V0",
    functools.partial(_make_v2_env, config=_ttp_wide_config),
)

# -- Demo Cook: R only, no penalty for incorrect --

_dc_simple_config = copy.deepcopy(_v2_base_config)
_dc_simple_config["grid"] = {"layout": "overcooked_v2_demo_cook_simple"}
_dc_simple_config["interactions"] = [build_branch_flag_delivery()]
_dc_simple_config["rewards"] = [
    TargetRecipeDeliveryReward(coefficient=20.0, common_reward=True, penalize_incorrect=False),
]
_dc_simple_config["tick_fn"] = build_target_recipe_tick(_dc_simple_config)
_dc_simple_config["extra_state_init_fn"] = functools.partial(
    build_target_recipe_extra_state,
    _dc_simple_config,
)

registry.register(
    "OvercookedV2-DemoCookSimple-V0",
    functools.partial(_make_v2_env, config=_dc_simple_config),
)

_dc_wide_config = copy.deepcopy(_dc_simple_config)
_dc_wide_config["grid"] = {"layout": "overcooked_v2_demo_cook_wide"}
_dc_wide_config["tick_fn"] = build_target_recipe_tick(_dc_wide_config)
_dc_wide_config["extra_state_init_fn"] = functools.partial(
    build_target_recipe_extra_state,
    _dc_wide_config,
)

registry.register(
    "OvercookedV2-DemoCookWide-V0",
    functools.partial(_make_v2_env, config=_dc_wide_config),
)


# ═══════════════════════════════════════════════════════════════════════════
# Search & Rescue
# ═══════════════════════════════════════════════════════════════════════════

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
