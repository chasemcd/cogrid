"""Scratch file: API for setting up OvercookedV2 benchmark environments.

Shows how the 6 new environments will look when registered, using existing
cogrid patterns (named recipes, Container, when(), make_ingredient_and_stack).

Key insight from the paper/code: pots accept ANY ingredient. There is no
validation at placement time. Correctness is checked only at delivery.
Ingredients like broccoli/mushroom are active distractors — agents can
cook with them, producing junk dishes that incur a -20 delivery penalty.
"""

import copy
import functools

from cogrid.cogrid_env import CoGridEnv
from cogrid.core.grid import layouts
from cogrid.envs import registry
from cogrid.envs.overcooked.agent import OvercookedAgent
from cogrid.envs.overcooked.config import build_target_recipe_tick
from cogrid.envs.overcooked.overcooked_grid_objects import (
    build_open_pot_recipes,
    make_ingredient_and_stack,
)
from cogrid.envs.overcooked.rewards import (
    ButtonActivationCost,
    TargetRecipeDeliveryReward,
)

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: New ingredients (module-import time registration)
# ═══════════════════════════════════════════════════════════════════════════
#
# Broccoli and mushroom are active distractors: agents CAN put them in
# pots, but the resulting dish never matches a target recipe.

Broccoli, BroccoliStack = make_ingredient_and_stack(
    "broccoli",
    "b",
    (34, 139, 34),
    "broccoli_stack",
    "B",
)

Mushroom, MushroomStack = make_ingredient_and_stack(
    "mushroom",
    "m",
    (139, 90, 43),
    "mushroom_stack",
    "M",
)


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Soup types for all ingredient combos
# ═══════════════════════════════════════════════════════════════════════════
#
# build_open_pot_recipes() enumerates all C(n+2, 3) = 20 three-ingredient
# combos from {onion, tomato, broccoli, mushroom}, creates a named soup
# result type for each (via make_soup), and returns the Recipe list.
#
# onion_soup and tomato_soup already exist. The helper skips those and
# only creates new types for the remaining 18 combos.
#
# Each soup is registered as pickupable + deliverable.

V2_INGREDIENTS = ["onion", "tomato", "broccoli", "mushroom"]
V2_POT_RECIPES = build_open_pot_recipes(V2_INGREDIENTS, cook_time=20)

# The Recipe list looks like:
#   Recipe(["onion", "onion", "onion"], result="onion_soup", cook_time=20)
#   Recipe(["tomato", "tomato", "tomato"], result="tomato_soup", cook_time=20)
#   Recipe(["onion", "onion", "tomato"], result="soup_onion_onion_tomato", cook_time=20)
#   Recipe(["onion", "tomato", "tomato"], result="soup_onion_tomato_tomato", cook_time=20)
#   ... (20 total)

# Collect all soup result names for delivery zone and pickupable_types
V2_SOUP_NAMES = sorted({r.result for r in V2_POT_RECIPES})


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: OpenPot — registered as separate type with expanded recipes
# ═══════════════════════════════════════════════════════════════════════════
#
# Uses char 'u' (lowercase) to coexist with standard Pot 'U'.
# Container(capacity=3, pickup_requires="plate") — same as standard Pot.
# The recipes list covers all 20 combos so the pot accepts anything.
#
# @register_object_type("open_pot", scope="overcooked")
# class OpenPot(GridObj):
#     char = "u"
#     container = Container(capacity=3, pickup_requires="plate")
#     recipes = V2_POT_RECIPES


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Layouts
# ═══════════════════════════════════════════════════════════════════════════
#
# Translated from JaxMARL. Character mapping:
#   W→C  A→+  X→@  B(plate pile)→=  P→u(open_pot)  R→R  L→L
#   0→O(onion_stack)  1→T(tomato_stack)  2→B(broccoli_stack)  3→M(mushroom_stack)

layouts.register_layout(
    "overcooked_v2_grounded_coord_simple",
    [
        "CCBCCCcC",
        "C  C=  O",
        "R +Lu+ @",
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
        "R+@+u + R",
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
        "R +Cu+ @",
        "C  C=  T",
        "CCBCCCCC",
    ],
)

layouts.register_layout(
    "overcooked_v2_test_time_wide",
    [
        "CC@=CC",
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
        "C     +u+ @",
        "T      C  =",
        "CCCCCRBCTCC",
    ],
)

layouts.register_layout(
    "overcooked_v2_demo_cook_wide",
    [
        "CCCC=@=CCCC",
        "CCCO + TCCC",
        "CCCCCuCCCCC",
        "C    +    C",
        "O  CMRMC  O",
        "CTCCCCCCCTC",
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Shared V2 base config
# ═══════════════════════════════════════════════════════════════════════════

_v2_base_config = {
    "name": "overcooked",
    "num_agents": 2,
    "n_agents": 2,
    "action_set": "cardinal_actions",
    "observable_radius": 2,
    "max_steps": 400,
    "scope": "overcooked",
    # Observation: local view handles partial obs via observable_radius
    "features": ["local_view"],
    # All items agents can hold (ingredients + plate + all soup types)
    "pickupable_types": [
        "onion",
        "tomato",
        "broccoli",
        "mushroom",
        "plate",
        *V2_SOUP_NAMES,
    ],
    # Target recipe system: one recipe result is "correct" at a time,
    # sampled uniformly on reset and resampled on each correct delivery.
    "target_recipes": ["onion_soup", "tomato_soup"],
    "resample_on_delivery": True,
}


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Per-environment configs + registration
# ═══════════════════════════════════════════════════════════════════════════
#
# Grounded Coordination: R + L (button costs -5), incorrect delivery = -20
# Test-Time Protocol:    R only, incorrect delivery = -20
# Demo Cook:             R only, no penalty for incorrect delivery

# --- Grounded Coordination Simple ---
gc_simple_config = copy.deepcopy(_v2_base_config)
gc_simple_config["grid"] = {"layout": "overcooked_v2_grounded_coord_simple"}
gc_simple_config["interactions"] = ["branch_activate_button"]
gc_simple_config["rewards"] = [
    TargetRecipeDeliveryReward(
        coefficient=20.0,
        common_reward=True,
        penalize_incorrect=True,
    ),
    ButtonActivationCost(cost=5.0),
]
gc_simple_config["tick_fn"] = build_target_recipe_tick(gc_simple_config)

registry.register(
    "OvercookedV2-GroundedCoordSimple-V0",
    functools.partial(CoGridEnv, config=gc_simple_config, agent_class=OvercookedAgent),
)

# --- Grounded Coordination Ring ---
gc_ring_config = copy.deepcopy(gc_simple_config)
gc_ring_config["grid"] = {"layout": "overcooked_v2_grounded_coord_ring"}
gc_ring_config["tick_fn"] = build_target_recipe_tick(gc_ring_config)

registry.register(
    "OvercookedV2-GroundedCoordRing-V0",
    functools.partial(CoGridEnv, config=gc_ring_config, agent_class=OvercookedAgent),
)

# --- Test-Time Protocol Formation Simple ---
ttp_simple_config = copy.deepcopy(_v2_base_config)
ttp_simple_config["grid"] = {"layout": "overcooked_v2_test_time_simple"}
ttp_simple_config["rewards"] = [
    TargetRecipeDeliveryReward(
        coefficient=20.0,
        common_reward=True,
        penalize_incorrect=True,
    ),
]
ttp_simple_config["tick_fn"] = build_target_recipe_tick(ttp_simple_config)

registry.register(
    "OvercookedV2-TestTimeSimple-V0",
    functools.partial(CoGridEnv, config=ttp_simple_config, agent_class=OvercookedAgent),
)

# --- Test-Time Protocol Formation Wide ---
ttp_wide_config = copy.deepcopy(ttp_simple_config)
ttp_wide_config["grid"] = {"layout": "overcooked_v2_test_time_wide"}
ttp_wide_config["tick_fn"] = build_target_recipe_tick(ttp_wide_config)

registry.register(
    "OvercookedV2-TestTimeWide-V0",
    functools.partial(CoGridEnv, config=ttp_wide_config, agent_class=OvercookedAgent),
)

# --- Demo Cook Simple ---
dc_simple_config = copy.deepcopy(_v2_base_config)
dc_simple_config["grid"] = {"layout": "overcooked_v2_demo_cook_simple"}
dc_simple_config["rewards"] = [
    TargetRecipeDeliveryReward(
        coefficient=20.0,
        common_reward=True,
        penalize_incorrect=False,
    ),
]
dc_simple_config["tick_fn"] = build_target_recipe_tick(dc_simple_config)

registry.register(
    "OvercookedV2-DemoCookSimple-V0",
    functools.partial(CoGridEnv, config=dc_simple_config, agent_class=OvercookedAgent),
)

# --- Demo Cook Wide ---
dc_wide_config = copy.deepcopy(dc_simple_config)
dc_wide_config["grid"] = {"layout": "overcooked_v2_demo_cook_wide"}
dc_wide_config["tick_fn"] = build_target_recipe_tick(dc_wide_config)

registry.register(
    "OvercookedV2-DemoCookWide-V0",
    functools.partial(CoGridEnv, config=dc_wide_config, agent_class=OvercookedAgent),
)
