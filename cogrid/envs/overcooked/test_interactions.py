"""Overcooked interaction parity tests.

Validates that array-based Overcooked interactions match the object-based
implementation using the new state-in/state-out process_interactions API.

Run with::

    python -c "from cogrid.envs.overcooked.test_interactions import \
        test_interaction_parity; test_interaction_parity()"
"""

from cogrid.core.autowire import build_scope_config_from_components
from cogrid.core.interactions import process_interactions
from cogrid.envs.overcooked.config import overcooked_interaction_fn, overcooked_tick


def _make_state(agent_pos, agent_dir, agent_inv, otm, osm, pot_contents, pot_timer, pot_positions):
    """Build a minimal EnvState for testing."""
    import numpy as np

    from cogrid.backend.env_state import EnvState

    H, W = otm.shape
    n_agents = agent_pos.shape[0]
    return EnvState(
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        agent_inv=agent_inv,
        wall_map=np.zeros((H, W), dtype=np.int32),
        object_type_map=otm,
        object_state_map=osm,
        extra_state={
            "overcooked.pot_contents": pot_contents,
            "overcooked.pot_timer": pot_timer,
            "overcooked.pot_positions": pot_positions,
        },
        rng_key=None,
        time=np.int32(0),
        done=np.zeros(n_agents, dtype=np.bool_),
        n_agents=n_agents,
        height=H,
        width=W,
        action_set="cardinal",
    )


def test_interaction_parity():
    """Validate array-based interactions match existing object-based interactions.

    Tests deterministic scenarios covering all interaction branches and the
    pot cooking state machine.
    """
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401 -- trigger environment registration
    from cogrid.core.agent import get_dir_vec_table
    from cogrid.core.grid_object import build_lookup_tables

    scope = "overcooked"
    tables = build_lookup_tables(scope=scope)
    scope_cfg = build_scope_config_from_components(scope)
    scope_cfg["interaction_fn"] = overcooked_interaction_fn
    type_ids = scope_cfg["type_ids"]
    dir_vec = get_dir_vec_table()

    onion_id = type_ids["onion"]
    tomato_id = type_ids["tomato"]
    plate_id = type_ids["plate"]
    pot_id = type_ids["pot"]
    onion_soup_id = type_ids["onion_soup"]
    tomato_soup_id = type_ids["tomato_soup"]

    PICKUP_DROP = 4
    NOOP = 6

    # Reusable empty pot state for tests
    pc_empty = np.array([[-1, -1, -1]], dtype=np.int32)
    pt_empty = np.array([30], dtype=np.int32)
    pp_dummy = np.array([[0, 0]], dtype=np.int32)

    # ---- Test 1: overcooked_tick parity with Pot.tick() ----
    print("Parity test 1: overcooked_tick vs Pot.tick()")

    from cogrid.envs.overcooked.overcooked_grid_objects import Onion, Pot

    # Empty pot
    pot_obj = Pot(capacity=3)
    pc = np.array([[-1, -1, -1]], dtype=np.int32)
    pt = np.array([30], dtype=np.int32)

    pot_obj.tick()
    _, new_pt, new_ps = overcooked_tick(pc, pt)
    assert pot_obj.cooking_timer == int(new_pt[0]), (
        f"Empty pot timer mismatch: obj={pot_obj.cooking_timer} arr={int(new_pt[0])}"
    )
    assert pot_obj.state == int(new_ps[0]), (
        f"Empty pot state mismatch: obj={pot_obj.state} arr={int(new_ps[0])}"
    )

    # Partially filled pot
    pot_obj2 = Pot(capacity=3)
    pot_obj2.objects_in_pot = [Onion(), Onion()]
    pc2 = np.array([[onion_id, onion_id, -1]], dtype=np.int32)
    pt2 = np.array([30], dtype=np.int32)

    pot_obj2.tick()
    _, new_pt2, new_ps2 = overcooked_tick(pc2, pt2)
    assert pot_obj2.cooking_timer == int(new_pt2[0]), (
        f"Partial pot timer mismatch: obj={pot_obj2.cooking_timer} arr={int(new_pt2[0])}"
    )
    assert pot_obj2.state == int(new_ps2[0]), (
        f"Partial pot state mismatch: obj={pot_obj2.state} arr={int(new_ps2[0])}"
    )

    # Full pot -- cooking cycle
    pot_obj3 = Pot(capacity=3)
    pot_obj3.objects_in_pot = [Onion(), Onion(), Onion()]
    pc3 = np.array([[onion_id, onion_id, onion_id]], dtype=np.int32)
    pt3 = np.array([30], dtype=np.int32)

    for _ in range(35):  # more than needed to fully cook
        pot_obj3.tick()
        _, pt3, ps3 = overcooked_tick(pc3, pt3)
        assert pot_obj3.cooking_timer == int(pt3[0]), (
            f"Cooking timer mismatch at step: obj={pot_obj3.cooking_timer} arr={int(pt3[0])}"
        )
        assert pot_obj3.state == int(ps3[0]), (
            f"Cooking state mismatch at step: obj={pot_obj3.state} arr={int(ps3[0])}"
        )

    print("  PASSED")

    # ---- Test 2: Deterministic scenario -- pickup onion from stack ----
    print("Parity test 2: Deterministic pickup from stack")

    agent_pos = np.array([[2, 3]], dtype=np.int32)
    agent_dir = np.array([0], dtype=np.int32)  # Right
    agent_inv = np.array([[-1]], dtype=np.int32)
    actions_arr = np.array([PICKUP_DROP], dtype=np.int32)

    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = type_ids["onion_stack"]

    state = _make_state(agent_pos, agent_dir, agent_inv, otm, osm, pc_empty, pt_empty, pp_dummy)
    state = process_interactions(
        state,
        actions_arr,
        overcooked_interaction_fn,
        tables,
        scope_cfg,
        dir_vec,
        PICKUP_DROP,
        5,
    )

    assert state.agent_inv[0, 0] == onion_id, f"Expected onion: {state.agent_inv[0, 0]}"
    assert state.object_type_map[2, 4] == type_ids["onion_stack"], "Stack should remain"
    print("  PASSED")

    # ---- Test 3: Full pot workflow ----
    print("Parity test 3: Full pot workflow (place, cook, pickup)")

    agent_pos = np.array([[2, 3]], dtype=np.int32)
    agent_dir = np.array([0], dtype=np.int32)
    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = pot_id
    pp = np.array([[2, 4]], dtype=np.int32)
    pc = np.array([[-1, -1, -1]], dtype=np.int32)
    pt = np.array([30], dtype=np.int32)

    for step_num in range(3):
        agent_inv = np.array([[onion_id]], dtype=np.int32)
        actions_arr = np.array([PICKUP_DROP], dtype=np.int32)
        state = _make_state(agent_pos, agent_dir, agent_inv, otm, osm, pc, pt, pp)
        state = process_interactions(
            state,
            actions_arr,
            overcooked_interaction_fn,
            tables,
            scope_cfg,
            dir_vec,
            PICKUP_DROP,
            5,
        )
        otm = state.object_type_map
        osm = state.object_state_map
        pc = state.extra_state["overcooked.pot_contents"]
        pt = state.extra_state["overcooked.pot_timer"]
        assert state.agent_inv[0, 0] == -1, f"Step {step_num}: agent should have placed onion"

    assert int(np.sum(pc[0] != -1)) == 3, f"Pot should have 3 items: {pc[0]}"

    # Step B: Cook for 30 ticks
    for _ in range(30):
        _, pt, ps = overcooked_tick(pc, pt)
    assert int(pt[0]) == 0, f"Pot should be done: {pt[0]}"

    # Step C: Pickup with plate
    agent_inv = np.array([[plate_id]], dtype=np.int32)
    actions_arr = np.array([PICKUP_DROP], dtype=np.int32)
    state = _make_state(agent_pos, agent_dir, agent_inv, otm, osm, pc, pt, pp)
    state = process_interactions(
        state,
        actions_arr,
        overcooked_interaction_fn,
        tables,
        scope_cfg,
        dir_vec,
        PICKUP_DROP,
        5,
    )
    assert state.agent_inv[0, 0] == onion_soup_id, f"Expected onion soup: {state.agent_inv[0, 0]}"
    assert int(state.extra_state["overcooked.pot_contents"][0, 0]) == -1, "Pot should be cleared"
    assert int(state.extra_state["overcooked.pot_timer"][0]) == 30, "Pot timer should reset"
    print("  PASSED")

    # ---- Test 4: Delivery zone parity ----
    print("Parity test 4: Delivery zone accepts soup only")
    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = type_ids["delivery_zone"]

    # Soup: accepted
    state = _make_state(
        agent_pos,
        agent_dir,
        np.array([[onion_soup_id]], dtype=np.int32),
        otm,
        osm,
        pc_empty,
        pt_empty,
        pp_dummy,
    )
    state = process_interactions(
        state,
        np.array([PICKUP_DROP], dtype=np.int32),
        overcooked_interaction_fn,
        tables,
        scope_cfg,
        dir_vec,
        PICKUP_DROP,
        5,
    )
    assert state.agent_inv[0, 0] == -1, "Soup should be delivered"

    # Non-soup: rejected
    state = _make_state(
        agent_pos,
        agent_dir,
        np.array([[onion_id]], dtype=np.int32),
        otm,
        osm,
        pc_empty,
        pt_empty,
        pp_dummy,
    )
    state = process_interactions(
        state,
        np.array([PICKUP_DROP], dtype=np.int32),
        overcooked_interaction_fn,
        tables,
        scope_cfg,
        dir_vec,
        PICKUP_DROP,
        5,
    )
    assert state.agent_inv[0, 0] == onion_id, "Onion should be rejected"
    print("  PASSED")

    # ---- Test 5: Counter place-on and pickup parity ----
    print("Parity test 5: Counter place and pickup")
    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = type_ids["counter"]

    # Place onion on counter
    state = _make_state(
        agent_pos,
        agent_dir,
        np.array([[onion_id]], dtype=np.int32),
        otm,
        osm,
        pc_empty,
        pt_empty,
        pp_dummy,
    )
    state = process_interactions(
        state,
        np.array([PICKUP_DROP], dtype=np.int32),
        overcooked_interaction_fn,
        tables,
        scope_cfg,
        dir_vec,
        PICKUP_DROP,
        5,
    )
    assert state.agent_inv[0, 0] == -1, "Agent should have placed on counter"
    assert state.object_state_map[2, 4] == onion_id, "Counter should store onion in state"

    # Try to place another item on occupied counter
    state2 = _make_state(
        agent_pos,
        agent_dir,
        np.array([[tomato_id]], dtype=np.int32),
        state.object_type_map,
        state.object_state_map,
        pc_empty,
        pt_empty,
        pp_dummy,
    )
    state2 = process_interactions(
        state2,
        np.array([PICKUP_DROP], dtype=np.int32),
        overcooked_interaction_fn,
        tables,
        scope_cfg,
        dir_vec,
        PICKUP_DROP,
        5,
    )
    assert state2.agent_inv[0, 0] == tomato_id, "Should reject: counter occupied"
    print("  PASSED")

    # ---- Test 6: Tomato soup from all-tomato pot ----
    print("Parity test 6: Tomato soup type detection")
    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = pot_id
    pp = np.array([[2, 4]], dtype=np.int32)
    pc = np.array([[tomato_id, tomato_id, tomato_id]], dtype=np.int32)
    pt = np.array([0], dtype=np.int32)  # ready

    state = _make_state(
        agent_pos,
        agent_dir,
        np.array([[plate_id]], dtype=np.int32),
        otm,
        osm,
        pc,
        pt,
        pp,
    )
    state = process_interactions(
        state,
        np.array([PICKUP_DROP], dtype=np.int32),
        overcooked_interaction_fn,
        tables,
        scope_cfg,
        dir_vec,
        PICKUP_DROP,
        5,
    )
    assert state.agent_inv[0, 0] == tomato_soup_id, f"Expected tomato soup: {state.agent_inv[0, 0]}"
    print("  PASSED")

    # ---- Test 7: Priority order -- pickup > pickup_from > drop > place_on ----
    print("Parity test 7: Priority order")

    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = onion_id  # pickupable

    state = _make_state(
        agent_pos,
        agent_dir,
        np.array([[-1]], dtype=np.int32),
        otm,
        osm,
        pc_empty,
        pt_empty,
        pp_dummy,
    )
    state = process_interactions(
        state,
        np.array([PICKUP_DROP], dtype=np.int32),
        overcooked_interaction_fn,
        tables,
        scope_cfg,
        dir_vec,
        PICKUP_DROP,
        5,
    )
    assert state.agent_inv[0, 0] == onion_id, "Pickup should take priority"
    assert state.object_type_map[2, 4] == 0, "Onion should be removed from grid"
    print("  PASSED")

    # ---- Test 8: Agent-ahead blocking ----
    print("Parity test 8: Agent ahead blocks interaction")
    agent_pos2 = np.array([[2, 3], [2, 4]], dtype=np.int32)
    agent_dir2 = np.array([0, 0], dtype=np.int32)
    agent_inv2 = np.array([[-1], [-1]], dtype=np.int32)
    actions2 = np.array([PICKUP_DROP, NOOP], dtype=np.int32)

    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = onion_id

    state = _make_state(agent_pos2, agent_dir2, agent_inv2, otm, osm, pc_empty, pt_empty, pp_dummy)
    state = process_interactions(
        state,
        actions2,
        overcooked_interaction_fn,
        tables,
        scope_cfg,
        dir_vec,
        PICKUP_DROP,
        5,
    )
    assert state.agent_inv[0, 0] == -1, "Agent 0 should be blocked by agent 1"
    print("  PASSED")

    print()
    print("ALL PARITY TESTS PASSED")


def test_at_most_one_branch_fires():
    """Verify at most one branch fires per agent-interaction pair across random states.

    Generates 600+ random interaction states (200 per configuration variant)
    using seeded RNG for reproducibility. For each state, runs the full
    branch sequence with handled accumulation and asserts that at most one
    branch fires.

    Run with::

        python -c "from cogrid.envs.overcooked.test_interactions import \\
            test_at_most_one_branch_fires; test_at_most_one_branch_fires()"
    """
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401 -- trigger environment registration

    from cogrid.envs.overcooked.config import _BRANCHES

    scope = "overcooked"
    scope_cfg = build_scope_config_from_components(scope)
    type_ids = scope_cfg["type_ids"]
    static_tables = scope_cfg["static_tables"]

    # All type IDs for random forward cell generation
    all_type_ids = [type_ids[n] for n in type_ids]
    # Items an agent can hold (or -1 for empty hand)
    holdable = [-1] + [type_ids[n] for n in ["onion", "tomato", "plate", "onion_soup", "tomato_soup"]]
    # Ingredient types for pot contents
    ingredient_ids = [type_ids["onion"], type_ids["tomato"]]

    rng = np.random.default_rng(seed=42)

    print("Invariant test: at-most-one-branch-fires")

    # Three configuration variants to cover different pot/grid setups:
    #   Variant A: pot at fwd cell, varied pot states
    #   Variant B: non-pot objects at fwd cell (stacks, counters, delivery zones)
    #   Variant C: empty fwd cell or random objects, varied inventory
    n_states_per_variant = 200
    total_states = 0
    failures = 0

    for variant in ["pot_focus", "non_pot_focus", "mixed"]:
        for _ in range(n_states_per_variant):
            # Random agent state
            inv_item = int(rng.choice(holdable))
            base_ok_val = bool(rng.choice([True, True, True, False]))  # 75% True to test branches

            # Grid setup: 7x7 grid, agent at (2,3) facing right toward (2,4)
            otm = np.zeros((7, 7), dtype=np.int32)
            osm = np.zeros((7, 7), dtype=np.int32)

            # Pot setup: 1 pot at (2,4) for pot-focused variants, plus
            # a second pot at (4,4) to ensure pot_positions has >1 entry
            if variant == "pot_focus":
                # Forward cell is always a pot
                fwd_type_val = int(type_ids["pot"])
                otm[2, 4] = fwd_type_val
                otm[4, 4] = fwd_type_val
                pot_positions = np.array([[2, 4], [4, 4]], dtype=np.int32)

                # Random pot contents: empty, partial (1-2 same type), full, or cooked
                pot_state_choice = rng.choice(["empty", "partial", "full", "cooked"])
                if pot_state_choice == "empty":
                    pot_contents = np.array([[-1, -1, -1], [-1, -1, -1]], dtype=np.int32)
                    pot_timer = np.array([30, 30], dtype=np.int32)
                elif pot_state_choice == "partial":
                    n_items = int(rng.choice([1, 2]))
                    ing = int(rng.choice(ingredient_ids))
                    row = [-1, -1, -1]
                    for s in range(n_items):
                        row[s] = ing
                    pot_contents = np.array([row, [-1, -1, -1]], dtype=np.int32)
                    pot_timer = np.array([30, 30], dtype=np.int32)
                elif pot_state_choice == "full":
                    ing = int(rng.choice(ingredient_ids))
                    pot_contents = np.array([[ing, ing, ing], [-1, -1, -1]], dtype=np.int32)
                    timer_val = int(rng.integers(1, 31))
                    pot_timer = np.array([timer_val, 30], dtype=np.int32)
                else:  # cooked
                    ing = int(rng.choice(ingredient_ids))
                    pot_contents = np.array([[ing, ing, ing], [-1, -1, -1]], dtype=np.int32)
                    pot_timer = np.array([0, 30], dtype=np.int32)

            elif variant == "non_pot_focus":
                # Forward cell is a non-pot object
                non_pot_types = [
                    type_ids[n]
                    for n in [
                        "onion", "tomato", "plate", "onion_soup", "tomato_soup",
                        "onion_stack", "tomato_stack", "plate_stack",
                        "counter", "delivery_zone",
                    ]
                ]
                fwd_type_val = int(rng.choice(non_pot_types))
                otm[2, 4] = fwd_type_val

                # Random counter state (occupied or empty)
                if fwd_type_val == type_ids["counter"]:
                    osm[2, 4] = int(rng.choice([0, type_ids["onion"], type_ids["plate"]]))

                # Minimal pot setup (no pot at fwd cell)
                pot_positions = np.array([[4, 4]], dtype=np.int32)
                pot_contents = np.array([[-1, -1, -1]], dtype=np.int32)
                pot_timer = np.array([30], dtype=np.int32)

            else:  # mixed
                # Random forward cell: could be anything including empty
                fwd_type_val = int(rng.choice([0] + all_type_ids))
                if fwd_type_val > 0:
                    otm[2, 4] = fwd_type_val

                # If counter, randomize occupancy
                if fwd_type_val == type_ids.get("counter", -1):
                    osm[2, 4] = int(rng.choice([0, type_ids["onion"]]))

                # If it's a pot, set up pot_positions to include it
                if fwd_type_val == type_ids["pot"]:
                    pot_positions = np.array([[2, 4]], dtype=np.int32)
                    pot_state_choice = rng.choice(["empty", "partial", "full", "cooked"])
                    if pot_state_choice == "empty":
                        pot_contents = np.array([[-1, -1, -1]], dtype=np.int32)
                        pot_timer = np.array([30], dtype=np.int32)
                    elif pot_state_choice == "partial":
                        n_items = int(rng.choice([1, 2]))
                        ing = int(rng.choice(ingredient_ids))
                        row = [-1, -1, -1]
                        for s in range(n_items):
                            row[s] = ing
                        pot_contents = np.array([row], dtype=np.int32)
                        pot_timer = np.array([30], dtype=np.int32)
                    elif pot_state_choice == "full":
                        ing = int(rng.choice(ingredient_ids))
                        pot_contents = np.array([[ing, ing, ing]], dtype=np.int32)
                        pot_timer = np.array([int(rng.integers(1, 31))], dtype=np.int32)
                    else:
                        ing = int(rng.choice(ingredient_ids))
                        pot_contents = np.array([[ing, ing, ing]], dtype=np.int32)
                        pot_timer = np.array([0], dtype=np.int32)
                else:
                    pot_positions = np.array([[4, 4]], dtype=np.int32)
                    pot_contents = np.array([[-1, -1, -1]], dtype=np.int32)
                    pot_timer = np.array([30], dtype=np.int32)

            # Build scalar arrays
            agent_inv = np.array([[inv_item]], dtype=np.int32)
            fwd_r = np.int32(2)
            fwd_c = np.int32(4)
            fwd_type = np.int32(fwd_type_val)
            inv_item_arr = np.int32(inv_item)
            base_ok = np.bool_(base_ok_val)
            agent_idx = np.int32(0)

            # Pot matching (same logic as orchestrator)
            fwd_pos_2d = np.stack([fwd_r, fwd_c])
            pot_match = np.all(pot_positions == fwd_pos_2d[None, :], axis=1)
            pot_idx = np.argmax(pot_match)
            has_pot_match = np.any(pot_match)

            # Assemble ctx dict (same structure as overcooked_interaction_body)
            ctx = {
                "base_ok": base_ok,
                "fwd_type": fwd_type,
                "fwd_r": fwd_r,
                "fwd_c": fwd_c,
                "inv_item": inv_item_arr,
                "agent_idx": agent_idx,
                "agent_inv": agent_inv,
                "object_type_map": otm,
                "object_state_map": osm,
                "pot_contents": pot_contents,
                "pot_timer": pot_timer,
                "pot_idx": pot_idx,
                "has_pot_match": has_pot_match,
                "CAN_PICKUP": static_tables["CAN_PICKUP"],
                "CAN_PICKUP_FROM": static_tables["CAN_PICKUP_FROM"],
                "CAN_PLACE_ON": static_tables["CAN_PLACE_ON"],
                "pickup_from_produces": static_tables["pickup_from_produces"],
                "legal_pot_ingredients": static_tables["legal_pot_ingredients"],
                "pot_id": static_tables["pot_id"],
                "plate_id": static_tables["plate_id"],
                "tomato_id": static_tables["tomato_id"],
                "onion_soup_id": static_tables["onion_soup_id"],
                "tomato_soup_id": static_tables["tomato_soup_id"],
                "delivery_zone_id": static_tables["delivery_zone_id"],
                "cooking_time": static_tables["cooking_time"],
                "recipe_ingredients": static_tables["recipe_ingredients"],
                "recipe_result": static_tables["recipe_result"],
                "recipe_cooking_time": static_tables["recipe_cooking_time"],
                "max_ingredients": static_tables["max_ingredients"],
                "IS_DELIVERABLE": static_tables["IS_DELIVERABLE"],
            }

            # Run branches with handled accumulation (as the real orchestrator does)
            handled = np.bool_(False)
            fired_count = 0
            fired_names = []
            for branch_fn in _BRANCHES:
                cond, updates, handled = branch_fn(handled, ctx)
                if bool(cond):
                    fired_count += 1
                    fired_names.append(branch_fn.__name__)

            if fired_count > 1:
                failures += 1
                print(
                    f"  FAIL: {fired_count} branches fired: {fired_names} "
                    f"(variant={variant}, inv={inv_item}, fwd_type={fwd_type_val}, "
                    f"base_ok={base_ok_val})"
                )

            total_states += 1

    assert failures == 0, f"{failures} states had multiple branches fire out of {total_states}"
    print(f"  Tested {total_states} random states, all passed at-most-one invariant")
    print("  PASSED")


def test_default_recipes_backward_compat():
    """Verify DEFAULT_RECIPES compile to arrays matching current hardcoded behavior."""
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401

    from cogrid.core.grid_object import object_to_idx
    from cogrid.envs.overcooked.config import (
        DEFAULT_RECIPES,
        _build_interaction_tables,
        compile_recipes,
    )

    scope = "overcooked"
    tables = compile_recipes(DEFAULT_RECIPES, scope=scope)

    # Verify shapes
    assert tables["recipe_ingredients"].shape == (2, 3), (
        f"Expected (2, 3), got {tables['recipe_ingredients'].shape}"
    )
    assert tables["recipe_result"].shape == (2,)
    assert tables["recipe_cooking_time"].shape == (2,)
    assert tables["recipe_reward"].shape == (2,)

    onion_id = object_to_idx("onion", scope=scope)
    tomato_id = object_to_idx("tomato", scope=scope)
    onion_soup_id = object_to_idx("onion_soup", scope=scope)
    tomato_soup_id = object_to_idx("tomato_soup", scope=scope)

    # Recipe 0: 3x onion -> onion_soup
    assert list(tables["recipe_ingredients"][0]) == [onion_id, onion_id, onion_id]
    assert tables["recipe_result"][0] == onion_soup_id

    # Recipe 1: 3x tomato -> tomato_soup
    assert list(tables["recipe_ingredients"][1]) == [tomato_id, tomato_id, tomato_id]
    assert tables["recipe_result"][1] == tomato_soup_id

    # Cooking times
    assert list(tables["recipe_cooking_time"]) == [30, 30]

    # Rewards
    assert list(tables["recipe_reward"]) == [20.0, 20.0]

    # legal_pot_ingredients must match current hardcoded behavior
    old_itables = _build_interaction_tables(scope)
    assert np.array_equal(
        tables["legal_pot_ingredients"],
        old_itables["legal_pot_ingredients"],
    ), "legal_pot_ingredients must match current hardcoded values"

    print("  Default recipe backward compatibility: PASSED")


def test_recipe_validation_errors():
    """Verify malformed recipes raise clear ValueError at init time."""
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401

    from cogrid.envs.overcooked.config import compile_recipes

    scope = "overcooked"

    test_cases = [
        ("empty list", []),
        (
            "missing key (no reward)",
            [{"ingredients": ["onion"], "result": "onion_soup", "cook_time": 30}],
        ),
        (
            "extra key",
            [
                {
                    "ingredients": ["onion", "onion", "onion"],
                    "result": "onion_soup",
                    "cook_time": 30,
                    "reward": 20.0,
                    "extra_key": True,
                }
            ],
        ),
        (
            "non-string ingredient",
            [{"ingredients": [123], "result": "onion_soup", "cook_time": 30, "reward": 20.0}],
        ),
        (
            "invalid ingredient name",
            [
                {
                    "ingredients": ["nonexistent_object"],
                    "result": "onion_soup",
                    "cook_time": 30,
                    "reward": 20.0,
                }
            ],
        ),
        (
            "invalid result name",
            [
                {
                    "ingredients": ["onion", "onion", "onion"],
                    "result": "nonexistent_soup",
                    "cook_time": 30,
                    "reward": 20.0,
                }
            ],
        ),
        (
            "non-positive cook_time (0)",
            [
                {
                    "ingredients": ["onion", "onion", "onion"],
                    "result": "onion_soup",
                    "cook_time": 0,
                    "reward": 20.0,
                }
            ],
        ),
        (
            "non-positive cook_time (-1)",
            [
                {
                    "ingredients": ["onion", "onion", "onion"],
                    "result": "onion_soup",
                    "cook_time": -1,
                    "reward": 20.0,
                }
            ],
        ),
        (
            "non-numeric reward",
            [
                {
                    "ingredients": ["onion", "onion", "onion"],
                    "result": "onion_soup",
                    "cook_time": 30,
                    "reward": "bad",
                }
            ],
        ),
        (
            "duplicate sorted ingredients",
            [
                {
                    "ingredients": ["onion", "onion", "onion"],
                    "result": "onion_soup",
                    "cook_time": 30,
                    "reward": 20.0,
                },
                {
                    "ingredients": ["onion", "onion", "onion"],
                    "result": "tomato_soup",
                    "cook_time": 30,
                    "reward": 15.0,
                },
            ],
        ),
    ]

    for name, config in test_cases:
        try:
            compile_recipes(config, scope)
            assert False, f"Expected ValueError for: {name}"
        except ValueError as e:
            print(f"    {name}: {e}")

    print("  Recipe validation errors: PASSED")


def test_custom_recipe_compilation():
    """Verify a non-default recipe config compiles with sorted ingredients."""
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401

    from cogrid.core.grid_object import object_to_idx
    from cogrid.envs.overcooked.config import compile_recipes

    scope = "overcooked"

    # 2 onions + 1 tomato, deliberately unsorted
    custom_recipe = {
        "ingredients": ["onion", "tomato", "onion"],
        "result": "onion_soup",
        "cook_time": 20,
        "reward": 15.0,
    }
    tables = compile_recipes([custom_recipe], scope)

    # Shape check
    assert tables["recipe_ingredients"].shape == (1, 3), (
        f"Expected (1, 3), got {tables['recipe_ingredients'].shape}"
    )

    # Ingredient IDs must be sorted ascending
    row = list(tables["recipe_ingredients"][0])
    assert row == sorted(row), f"Ingredients not sorted: {row}"

    onion_id = object_to_idx("onion", scope=scope)
    tomato_id = object_to_idx("tomato", scope=scope)
    expected = sorted([onion_id, onion_id, tomato_id])
    assert row == expected, f"Expected {expected}, got {row}"

    # Cooking time and reward
    assert tables["recipe_cooking_time"][0] == 20
    assert tables["recipe_reward"][0] == 15.0

    # legal_pot_ingredients has 1 for both onion and tomato
    assert tables["legal_pot_ingredients"][onion_id] == 1
    assert tables["legal_pot_ingredients"][tomato_id] == 1

    # max_ingredients
    assert tables["max_ingredients"] == 3

    print("  Custom recipe compilation: PASSED")


def test_mixed_recipe_end_to_end():
    """Verify a mixed-ingredient recipe (2 onion + 1 tomato -> onion_soup) works end-to-end.

    Validates RCPE-06: places 2 onions and 1 tomato into a pot via prefix-match,
    cooks for 20 ticks (custom cook_time, not default 30), picks up onion_soup
    with a plate, and delivers via IS_DELIVERABLE lookup.
    """
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401

    from cogrid.core.agent import get_dir_vec_table
    from cogrid.core.grid_object import build_lookup_tables, object_to_idx
    from cogrid.envs.overcooked.config import (
        _build_interaction_tables,
        _build_static_tables,
        _build_type_ids,
        compile_recipes,
        overcooked_interaction_fn,
        overcooked_tick,
    )

    scope = "overcooked"
    dir_vec = get_dir_vec_table()
    PICKUP_DROP = 4

    # Custom recipe: 2 onion + 1 tomato -> onion_soup, cook_time=20
    custom_recipes = [
        {
            "ingredients": ["onion", "onion", "tomato"],
            "result": "onion_soup",
            "cook_time": 20,
            "reward": 15.0,
        },
    ]

    # Build scope_config with custom recipe tables
    scope_cfg = build_scope_config_from_components(scope)
    itables = _build_interaction_tables(scope)
    type_ids_dict = _build_type_ids(scope)
    recipe_tables = compile_recipes(custom_recipes, scope=scope)
    static_tables = _build_static_tables(scope, itables, type_ids_dict, recipe_tables=recipe_tables)
    scope_cfg["static_tables"] = static_tables
    scope_cfg["interaction_fn"] = overcooked_interaction_fn
    tables = build_lookup_tables(scope=scope)

    type_ids = scope_cfg["type_ids"]
    onion_id = type_ids["onion"]
    tomato_id = type_ids["tomato"]
    plate_id = type_ids["plate"]
    pot_id = type_ids["pot"]
    onion_soup_id = type_ids["onion_soup"]
    delivery_zone_id = type_ids["delivery_zone"]

    print("Mixed-recipe end-to-end test:")

    # --- Setup: 7x7 grid, pot at (2,4), delivery zone at (2,5) ---
    agent_pos = np.array([[2, 3]], dtype=np.int32)
    agent_dir = np.array([0], dtype=np.int32)  # Right
    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = pot_id
    otm[2, 5] = delivery_zone_id
    pp = np.array([[2, 4]], dtype=np.int32)
    pc = np.array([[-1, -1, -1]], dtype=np.int32)
    pt = np.array([30], dtype=np.int32)

    # --- Step A: Place 2 onions into pot ---
    for i in range(2):
        agent_inv = np.array([[onion_id]], dtype=np.int32)
        actions_arr = np.array([PICKUP_DROP], dtype=np.int32)
        state = _make_state(agent_pos, agent_dir, agent_inv, otm, osm, pc, pt, pp)
        state = process_interactions(
            state, actions_arr, overcooked_interaction_fn, tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
        )
        otm = state.object_type_map
        osm = state.object_state_map
        pc = state.extra_state["overcooked.pot_contents"]
        pt = state.extra_state["overcooked.pot_timer"]
        assert state.agent_inv[0, 0] == -1, f"Onion {i}: agent should have placed onion"

    assert int(np.sum(pc[0] != -1)) == 2, f"Pot should have 2 items: {pc[0]}"
    print("  Placed 2 onions: OK")

    # --- Step B: Place 1 tomato (fills the pot) ---
    agent_inv = np.array([[tomato_id]], dtype=np.int32)
    actions_arr = np.array([PICKUP_DROP], dtype=np.int32)
    state = _make_state(agent_pos, agent_dir, agent_inv, otm, osm, pc, pt, pp)
    state = process_interactions(
        state, actions_arr, overcooked_interaction_fn, tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
    )
    otm = state.object_type_map
    osm = state.object_state_map
    pc = state.extra_state["overcooked.pot_contents"]
    pt = state.extra_state["overcooked.pot_timer"]
    assert state.agent_inv[0, 0] == -1, "Agent should have placed tomato"
    assert int(np.sum(pc[0] != -1)) == 3, f"Pot should have 3 items: {pc[0]}"
    assert int(pt[0]) == 20, f"Pot timer should be 20 (custom cook_time), got {int(pt[0])}"
    print("  Placed 1 tomato, pot full, timer=20: OK")

    # --- Step C: Cook for exactly 20 ticks ---
    for tick_i in range(20):
        _, pt, _ = overcooked_tick(pc, pt)
    assert int(pt[0]) == 0, f"Pot should be done after 20 ticks: {pt[0]}"
    print("  Cooked for 20 ticks, done: OK")

    # --- Step D: Pickup with plate ---
    agent_inv = np.array([[plate_id]], dtype=np.int32)
    actions_arr = np.array([PICKUP_DROP], dtype=np.int32)
    state = _make_state(agent_pos, agent_dir, agent_inv, otm, osm, pc, pt, pp)
    state = process_interactions(
        state, actions_arr, overcooked_interaction_fn, tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
    )
    assert state.agent_inv[0, 0] == onion_soup_id, (
        f"Expected onion_soup ({onion_soup_id}), got {state.agent_inv[0, 0]}"
    )
    pc = state.extra_state["overcooked.pot_contents"]
    pt = state.extra_state["overcooked.pot_timer"]
    assert int(pc[0, 0]) == -1, "Pot should be cleared after pickup"
    print("  Picked up onion_soup with plate: OK")

    # --- Step E: Deliver ---
    # Agent now faces delivery zone at (2,5). Move agent to (2,4) facing right.
    agent_pos_dz = np.array([[2, 4]], dtype=np.int32)
    agent_inv = np.array([[onion_soup_id]], dtype=np.int32)
    actions_arr = np.array([PICKUP_DROP], dtype=np.int32)
    otm_dz = state.object_type_map
    osm_dz = state.object_state_map
    state = _make_state(agent_pos_dz, agent_dir, agent_inv, otm_dz, osm_dz, pc, pt, pp)
    state = process_interactions(
        state, actions_arr, overcooked_interaction_fn, tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
    )
    assert state.agent_inv[0, 0] == -1, "Soup should be delivered via IS_DELIVERABLE"
    print("  Delivered onion_soup: OK")

    print("  PASSED")


def test_per_recipe_cook_time():
    """Verify per-recipe cook times: onion=10 ticks, tomato=50 ticks.

    Validates RCPE-05: each recipe has its own cook_time set in pot_timer
    when the pot fills. Uses custom recipes with different cook times to
    confirm the default 30 is NOT used.
    """
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401

    from cogrid.core.agent import get_dir_vec_table
    from cogrid.core.grid_object import build_lookup_tables, object_to_idx
    from cogrid.envs.overcooked.config import (
        _build_interaction_tables,
        _build_static_tables,
        _build_type_ids,
        compile_recipes,
        overcooked_interaction_fn,
        overcooked_tick,
    )

    scope = "overcooked"
    dir_vec = get_dir_vec_table()
    PICKUP_DROP = 4

    # Custom recipes: onion soup cooks in 10 ticks, tomato soup in 50 ticks
    custom_recipes = [
        {
            "ingredients": ["onion", "onion", "onion"],
            "result": "onion_soup",
            "cook_time": 10,
            "reward": 20.0,
        },
        {
            "ingredients": ["tomato", "tomato", "tomato"],
            "result": "tomato_soup",
            "cook_time": 50,
            "reward": 20.0,
        },
    ]

    # Build scope_config with custom recipe tables
    scope_cfg = build_scope_config_from_components(scope)
    itables = _build_interaction_tables(scope)
    type_ids_dict = _build_type_ids(scope)
    recipe_tables = compile_recipes(custom_recipes, scope=scope)
    static_tables = _build_static_tables(scope, itables, type_ids_dict, recipe_tables=recipe_tables)
    scope_cfg["static_tables"] = static_tables
    scope_cfg["interaction_fn"] = overcooked_interaction_fn
    tables = build_lookup_tables(scope=scope)

    type_ids = scope_cfg["type_ids"]
    onion_id = type_ids["onion"]
    tomato_id = type_ids["tomato"]
    plate_id = type_ids["plate"]
    pot_id = type_ids["pot"]
    onion_soup_id = type_ids["onion_soup"]
    tomato_soup_id = type_ids["tomato_soup"]

    print("Per-recipe cook time test:")

    agent_pos = np.array([[2, 3]], dtype=np.int32)
    agent_dir = np.array([0], dtype=np.int32)  # Right
    pp = np.array([[2, 4]], dtype=np.int32)

    # ---- Part A: Onion recipe (cook_time=10) ----
    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = pot_id
    pc = np.array([[-1, -1, -1]], dtype=np.int32)
    pt = np.array([30], dtype=np.int32)

    # Fill pot with 3 onions
    for i in range(3):
        agent_inv = np.array([[onion_id]], dtype=np.int32)
        actions_arr = np.array([PICKUP_DROP], dtype=np.int32)
        state = _make_state(agent_pos, agent_dir, agent_inv, otm, osm, pc, pt, pp)
        state = process_interactions(
            state, actions_arr, overcooked_interaction_fn, tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
        )
        otm = state.object_type_map
        osm = state.object_state_map
        pc = state.extra_state["overcooked.pot_contents"]
        pt = state.extra_state["overcooked.pot_timer"]
        assert state.agent_inv[0, 0] == -1, f"Onion {i}: agent should have placed onion"

    assert int(pt[0]) == 10, f"Onion recipe timer should be 10, got {int(pt[0])}"
    print("  Onion pot timer=10: OK")

    # Cook for 10 ticks
    for _ in range(10):
        _, pt, _ = overcooked_tick(pc, pt)
    assert int(pt[0]) == 0, f"Onion pot should be done after 10 ticks: {pt[0]}"
    print("  Cooked for 10 ticks, done: OK")

    # Pickup with plate
    agent_inv = np.array([[plate_id]], dtype=np.int32)
    state = _make_state(agent_pos, agent_dir, agent_inv, otm, osm, pc, pt, pp)
    state = process_interactions(
        state, np.array([PICKUP_DROP], dtype=np.int32),
        overcooked_interaction_fn, tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
    )
    assert state.agent_inv[0, 0] == onion_soup_id, (
        f"Expected onion_soup, got {state.agent_inv[0, 0]}"
    )
    print("  Picked up onion_soup: OK")

    # ---- Part B: Tomato recipe (cook_time=50) ----
    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = pot_id
    pc = np.array([[-1, -1, -1]], dtype=np.int32)
    pt = np.array([30], dtype=np.int32)

    # Fill pot with 3 tomatoes
    for i in range(3):
        agent_inv = np.array([[tomato_id]], dtype=np.int32)
        actions_arr = np.array([PICKUP_DROP], dtype=np.int32)
        state = _make_state(agent_pos, agent_dir, agent_inv, otm, osm, pc, pt, pp)
        state = process_interactions(
            state, actions_arr, overcooked_interaction_fn, tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
        )
        otm = state.object_type_map
        osm = state.object_state_map
        pc = state.extra_state["overcooked.pot_contents"]
        pt = state.extra_state["overcooked.pot_timer"]
        assert state.agent_inv[0, 0] == -1, f"Tomato {i}: agent should have placed tomato"

    assert int(pt[0]) == 50, f"Tomato recipe timer should be 50, got {int(pt[0])}"
    print("  Tomato pot timer=50: OK")

    # Cook for 50 ticks
    for _ in range(50):
        _, pt, _ = overcooked_tick(pc, pt)
    assert int(pt[0]) == 0, f"Tomato pot should be done after 50 ticks: {pt[0]}"
    print("  Cooked for 50 ticks, done: OK")

    # Pickup with plate
    agent_inv = np.array([[plate_id]], dtype=np.int32)
    state = _make_state(agent_pos, agent_dir, agent_inv, otm, osm, pc, pt, pp)
    state = process_interactions(
        state, np.array([PICKUP_DROP], dtype=np.int32),
        overcooked_interaction_fn, tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
    )
    assert state.agent_inv[0, 0] == tomato_soup_id, (
        f"Expected tomato_soup, got {state.agent_inv[0, 0]}"
    )
    print("  Picked up tomato_soup: OK")

    print("  PASSED")
