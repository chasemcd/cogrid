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
    r"""Verify at most one branch fires per agent-interaction pair across random states.

    Generates 600+ random interaction states (200 per configuration variant)
    using seeded RNG for reproducibility. For each state, runs the full
    branch sequence with handled accumulation and asserts that at most one
    branch fires.

    Run with::

        python -c "from cogrid.envs.overcooked.test_interactions import \
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
    holdable = [-1] + [
        type_ids[n] for n in ["onion", "tomato", "plate", "onion_soup", "tomato_soup"]
    ]
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
                        "onion",
                        "tomato",
                        "plate",
                        "onion_soup",
                        "tomato_soup",
                        "onion_stack",
                        "tomato_stack",
                        "plate_stack",
                        "counter",
                        "delivery_zone",
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
    from cogrid.core.grid_object import build_lookup_tables
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
        assert state.agent_inv[0, 0] == -1, f"Onion {i}: agent should have placed onion"

    assert int(np.sum(pc[0] != -1)) == 2, f"Pot should have 2 items: {pc[0]}"
    print("  Placed 2 onions: OK")

    # --- Step B: Place 1 tomato (fills the pot) ---
    agent_inv = np.array([[tomato_id]], dtype=np.int32)
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
        state,
        actions_arr,
        overcooked_interaction_fn,
        tables,
        scope_cfg,
        dir_vec,
        PICKUP_DROP,
        5,
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
        state,
        actions_arr,
        overcooked_interaction_fn,
        tables,
        scope_cfg,
        dir_vec,
        PICKUP_DROP,
        5,
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
    from cogrid.core.grid_object import build_lookup_tables
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
        state,
        np.array([PICKUP_DROP], dtype=np.int32),
        overcooked_interaction_fn,
        tables,
        scope_cfg,
        dir_vec,
        PICKUP_DROP,
        5,
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
        state,
        np.array([PICKUP_DROP], dtype=np.int32),
        overcooked_interaction_fn,
        tables,
        scope_cfg,
        dir_vec,
        PICKUP_DROP,
        5,
    )
    assert state.agent_inv[0, 0] == tomato_soup_id, (
        f"Expected tomato_soup, got {state.agent_inv[0, 0]}"
    )
    print("  Picked up tomato_soup: OK")

    print("  PASSED")


def test_pickup_from_produces_config_driven():
    """Verify config-driven pickup_from_produces table matches expected mappings."""
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.core.grid_object import object_to_idx
    from cogrid.core.grid_object_registry import build_lookup_tables

    scope = "overcooked"
    tables = build_lookup_tables(scope)
    pfp = tables["pickup_from_produces"]

    onion_stack_id = object_to_idx("onion_stack", scope=scope)
    tomato_stack_id = object_to_idx("tomato_stack", scope=scope)
    plate_stack_id = object_to_idx("plate_stack", scope=scope)
    onion_id = object_to_idx("onion", scope=scope)
    tomato_id = object_to_idx("tomato", scope=scope)
    plate_id = object_to_idx("plate", scope=scope)

    # Verify expected mappings
    assert int(pfp[onion_stack_id]) == onion_id, (
        f"onion_stack should produce onion: {int(pfp[onion_stack_id])} != {onion_id}"
    )
    assert int(pfp[tomato_stack_id]) == tomato_id, (
        f"tomato_stack should produce tomato: {int(pfp[tomato_stack_id])} != {tomato_id}"
    )
    assert int(pfp[plate_stack_id]) == plate_id, (
        f"plate_stack should produce plate: {int(pfp[plate_stack_id])} != {plate_id}"
    )

    # Verify no spurious mappings among non-stack types.
    # Other tests may register additional stacks in the same process,
    # so we only check types whose name does NOT end with "_stack".
    from cogrid.core.grid_object import get_object_names

    names = get_object_names(scope=scope)
    known_stacks = {onion_stack_id, tomato_stack_id, plate_stack_id}
    for i, name in enumerate(names):
        if i in known_stacks:
            continue
        if name is not None and name.endswith("_stack"):
            continue  # skip stacks registered by other tests
        assert int(pfp[i]) == 0, f"Unexpected non-zero entry at index {i} ({name}): {int(pfp[i])}"

    print("  Config-driven pickup_from_produces: PASSED")


def test_stack_subclasses_are_thin():
    """Verify stack classes have no method overrides -- only class attributes."""
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.envs.overcooked.overcooked_grid_objects import (
        OnionStack,
        PlateStack,
        TomatoStack,
        _BaseStack,
    )

    for cls in (OnionStack, TomatoStack, PlateStack):
        assert issubclass(cls, _BaseStack), f"{cls.__name__} should extend _BaseStack"

        # Verify NO method overrides in the subclass __dict__
        for method_name in ("pick_up_from", "render"):
            assert method_name not in cls.__dict__, (
                f"{cls.__name__} should NOT define its own {method_name}"
            )

        # Verify produces is set to a non-None string
        assert isinstance(cls.produces, str) and cls.produces is not None, (
            f"{cls.__name__}.produces should be a non-None string"
        )

    print("  Stack subclasses are thin: PASSED")


def test_factory_registers_new_types():
    """Verify make_ingredient_and_stack creates and registers types correctly.

    Demonstrates the required calling pattern: factory BEFORE table build.
    """
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.core.grid_object import get_object_names, make_object, object_to_idx
    from cogrid.core.grid_object_registry import build_lookup_tables
    from cogrid.envs.overcooked.overcooked_grid_objects import make_ingredient_and_stack

    scope = "overcooked"

    # Check if already registered (from a previous test run in the same process)
    names = get_object_names(scope=scope)
    if "test_mushroom" not in names:
        make_ingredient_and_stack("test_mushroom", "9", [139, 90, 43], "test_mushroom_stack", "0")

    # Verify both types are registered
    names = get_object_names(scope=scope)
    assert "test_mushroom" in names, "test_mushroom should be registered"
    assert "test_mushroom_stack" in names, "test_mushroom_stack should be registered"

    # Verify stack produces attribute
    stack_obj = make_object("test_mushroom_stack", scope=scope)
    assert stack_obj.produces == "test_mushroom", (
        f"Stack.produces should be 'test_mushroom', got {stack_obj.produces}"
    )

    # Build tables AFTER factory call and verify scan picks it up
    tables = build_lookup_tables(scope)
    pfp = tables["pickup_from_produces"]
    mushroom_stack_id = object_to_idx("test_mushroom_stack", scope=scope)
    mushroom_id = object_to_idx("test_mushroom", scope=scope)
    assert int(pfp[mushroom_stack_id]) == mushroom_id, (
        f"pickup_from_produces should map test_mushroom_stack -> test_mushroom: "
        f"{int(pfp[mushroom_stack_id])} != {mushroom_id}"
    )

    print("  Factory registers new types: PASSED")


def test_factory_stack_dispenses_item():
    """Verify a factory-created stack works at object level and through Branch 2B."""
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.core.grid_object import get_object_names, make_object, object_to_idx
    from cogrid.core.grid_object_registry import build_lookup_tables
    from cogrid.envs.overcooked.overcooked_grid_objects import make_ingredient_and_stack

    scope = "overcooked"

    # Register if not already done
    names = get_object_names(scope=scope)
    if "test_mushroom" not in names:
        make_ingredient_and_stack("test_mushroom", "9", [139, 90, 43], "test_mushroom_stack", "0")

    # Object-level check
    stack = make_object("test_mushroom_stack", scope=scope)
    assert stack.can_pickup_from, "Stack should allow pickup"
    item = stack.pick_up_from(None)
    assert item.object_id == "test_mushroom", (
        f"Dispensed item should be test_mushroom, got {item.object_id}"
    )

    # Branch 2B integration check: rebuild tables and verify entry
    tables = build_lookup_tables(scope)
    pfp = tables["pickup_from_produces"]
    mushroom_stack_id = object_to_idx("test_mushroom_stack", scope=scope)
    mushroom_id = object_to_idx("test_mushroom", scope=scope)
    assert int(pfp[mushroom_stack_id]) > 0, (
        "pickup_from_produces should have non-zero entry for test_mushroom_stack"
    )
    assert int(pfp[mushroom_stack_id]) == mushroom_id, (
        "pickup_from_produces[test_mushroom_stack] should equal test_mushroom id"
    )

    print("  Factory stack dispenses item: PASSED")


# ======================================================================
# Order queue tests
# ======================================================================


def _make_state_with_orders(
    agent_pos,
    agent_dir,
    agent_inv,
    otm,
    osm,
    pot_contents,
    pot_timer,
    pot_positions,
    order_recipe=None,
    order_timer=None,
    order_spawn_counter=None,
    order_recipe_counter=None,
    order_n_expired=None,
):
    """Build a minimal EnvState with optional order arrays for testing."""
    import numpy as np

    from cogrid.backend.env_state import EnvState

    H, W = otm.shape
    n_agents = agent_pos.shape[0]
    extra_state = {
        "overcooked.pot_contents": pot_contents,
        "overcooked.pot_timer": pot_timer,
        "overcooked.pot_positions": pot_positions,
    }
    if order_recipe is not None:
        extra_state["overcooked.order_recipe"] = order_recipe
        extra_state["overcooked.order_timer"] = order_timer
        extra_state["overcooked.order_spawn_counter"] = order_spawn_counter
        extra_state["overcooked.order_recipe_counter"] = order_recipe_counter
        extra_state["overcooked.order_n_expired"] = order_n_expired
    return EnvState(
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        agent_inv=agent_inv,
        wall_map=np.zeros((H, W), dtype=np.int32),
        object_type_map=otm,
        object_state_map=osm,
        extra_state=extra_state,
        rng_key=None,
        time=np.int32(0),
        done=np.zeros(n_agents, dtype=np.bool_),
        n_agents=n_agents,
        height=H,
        width=W,
        action_set="cardinal",
    )


def test_order_tick_lifecycle():
    """Test order tick mechanics: spawn, countdown, expiry, re-spawn.

    Uses spawn_interval=5, max_active=2, time_limit=10, uniform weights
    over 2 recipes.
    """
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.envs.overcooked.config import (
        DEFAULT_RECIPES,
        _build_interaction_tables,
        _build_order_tables,
        _build_static_tables,
        _build_type_ids,
        compile_recipes,
        overcooked_tick_state,
    )

    scope = "overcooked"
    order_config = {
        "spawn_interval": 5,
        "max_active": 2,
        "time_limit": 10,
        "recipe_weights": [1.0, 1.0],
    }

    # Build scope config with order tables
    itables = _build_interaction_tables(scope)
    type_ids_dict = _build_type_ids(scope)
    recipe_tables = compile_recipes(DEFAULT_RECIPES, scope=scope)
    order_tables = _build_order_tables(order_config, n_recipes=len(DEFAULT_RECIPES))
    static_tables = _build_static_tables(
        scope,
        itables,
        type_ids_dict,
        recipe_tables=recipe_tables,
        order_tables=order_tables,
    )
    scope_cfg = build_scope_config_from_components(scope)
    scope_cfg["static_tables"] = static_tables

    # Setup: minimal grid with 1 pot
    agent_pos = np.array([[2, 3]], dtype=np.int32)
    agent_dir = np.array([0], dtype=np.int32)
    agent_inv = np.array([[-1]], dtype=np.int32)
    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    pot_id = type_ids_dict["pot"]
    otm[2, 4] = pot_id
    pp = np.array([[2, 4]], dtype=np.int32)
    pc = np.array([[-1, -1, -1]], dtype=np.int32)
    pt = np.array([30], dtype=np.int32)

    # Order arrays: empty queue, counter starts at 5
    order_recipe = np.full((2,), -1, dtype=np.int32)
    order_timer = np.zeros((2,), dtype=np.int32)
    order_spawn_counter = np.int32(5)
    order_recipe_counter = np.int32(0)
    order_n_expired = np.int32(0)

    state = _make_state_with_orders(
        agent_pos,
        agent_dir,
        agent_inv,
        otm,
        osm,
        pc,
        pt,
        pp,
        order_recipe=order_recipe,
        order_timer=order_timer,
        order_spawn_counter=order_spawn_counter,
        order_recipe_counter=order_recipe_counter,
        order_n_expired=order_n_expired,
    )

    print("Order tick lifecycle test:")

    # Phase A: 5 ticks -> first order spawns
    for i in range(5):
        state = overcooked_tick_state(state, scope_cfg)

    or_ = state.extra_state["overcooked.order_recipe"]
    ot_ = state.extra_state["overcooked.order_timer"]
    osc = state.extra_state["overcooked.order_spawn_counter"]
    assert int(or_[0]) == 0, f"Phase A: first order should be recipe 0, got {int(or_[0])}"
    assert int(ot_[0]) == 10, f"Phase A: first order timer should be 10, got {int(ot_[0])}"
    assert int(osc) == 5, f"Phase A: spawn counter should reset to 5, got {int(osc)}"
    print("  Phase A (first spawn after 5 ticks): OK")

    # Phase B: 5 more ticks -> second order spawns, first timer decrements
    for i in range(5):
        state = overcooked_tick_state(state, scope_cfg)

    or_ = state.extra_state["overcooked.order_recipe"]
    ot_ = state.extra_state["overcooked.order_timer"]
    assert int(or_[1]) == 1, f"Phase B: second order should be recipe 1, got {int(or_[1])}"
    assert int(ot_[1]) == 10, f"Phase B: second order timer should be 10, got {int(ot_[1])}"
    assert int(ot_[0]) == 5, f"Phase B: first order timer should be 5, got {int(ot_[0])}"
    print("  Phase B (second spawn, first countdown): OK")

    # Phase C: 5 more ticks -> first order expires (timer was 5)
    for i in range(5):
        state = overcooked_tick_state(state, scope_cfg)

    or_ = state.extra_state["overcooked.order_recipe"]
    ot_ = state.extra_state["overcooked.order_timer"]
    one = state.extra_state["overcooked.order_n_expired"]
    assert int(or_[0]) != -1 or int(one) >= 1, (
        f"Phase C: first order should have expired. or_[0]={int(or_[0])}, n_expired={int(one)}"
    )
    # The first order expired at tick 10 (timer went from 5 to 0).
    # But a new spawn also happened at tick 15 (counter reset at tick 10, then -5 again).
    # The new spawn fills the first empty slot (slot 0 since it expired).
    print(f"  Phase C (expiry + re-spawn): n_expired={int(one)}, or_={[int(x) for x in or_]}")

    # Phase D: Verify we've had at least one expiry
    assert int(one) >= 1, f"Phase D: expected at least 1 expired order, got {int(one)}"
    print("  Phase D (expiry tracked): OK")

    print("  PASSED")


def test_order_delivery_consumes_order():
    """Test that delivering a matching soup consumes the first matching active order."""
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.core.agent import get_dir_vec_table
    from cogrid.core.grid_object import build_lookup_tables
    from cogrid.envs.overcooked.config import (
        DEFAULT_RECIPES,
        _build_interaction_tables,
        _build_order_tables,
        _build_static_tables,
        _build_type_ids,
        compile_recipes,
        overcooked_interaction_fn,
    )

    scope = "overcooked"
    dir_vec = get_dir_vec_table()
    PICKUP_DROP = 4

    order_config = {
        "spawn_interval": 1,
        "max_active": 3,
        "time_limit": 1000,
    }

    # Build scope_config with order tables
    scope_cfg = build_scope_config_from_components(scope)
    itables = _build_interaction_tables(scope)
    type_ids_dict = _build_type_ids(scope)
    recipe_tables = compile_recipes(DEFAULT_RECIPES, scope=scope)
    order_tables = _build_order_tables(order_config, n_recipes=len(DEFAULT_RECIPES))
    static_tables = _build_static_tables(
        scope,
        itables,
        type_ids_dict,
        recipe_tables=recipe_tables,
        order_tables=order_tables,
    )
    scope_cfg["static_tables"] = static_tables
    scope_cfg["interaction_fn"] = overcooked_interaction_fn
    tables = build_lookup_tables(scope=scope)

    type_ids = scope_cfg["type_ids"]
    onion_soup_id = type_ids["onion_soup"]
    delivery_zone_id = type_ids["delivery_zone"]

    print("Order delivery consumes order test:")

    # Setup: agent holding onion_soup, facing delivery_zone
    agent_pos = np.array([[2, 3]], dtype=np.int32)
    agent_dir = np.array([0], dtype=np.int32)  # Right
    agent_inv = np.array([[onion_soup_id]], dtype=np.int32)
    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = delivery_zone_id
    pp = np.array([[0, 0]], dtype=np.int32)
    pc = np.array([[-1, -1, -1]], dtype=np.int32)
    pt = np.array([30], dtype=np.int32)

    # Inject active order: recipe 0 = onion_soup
    order_recipe = np.array([0, -1, -1], dtype=np.int32)
    order_timer = np.array([100, 0, 0], dtype=np.int32)
    order_spawn_counter = np.int32(10)
    order_recipe_counter = np.int32(1)
    order_n_expired = np.int32(0)

    state = _make_state_with_orders(
        agent_pos,
        agent_dir,
        agent_inv,
        otm,
        osm,
        pc,
        pt,
        pp,
        order_recipe=order_recipe,
        order_timer=order_timer,
        order_spawn_counter=order_spawn_counter,
        order_recipe_counter=order_recipe_counter,
        order_n_expired=order_n_expired,
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
    or_ = state.extra_state["overcooked.order_recipe"]
    ot_ = state.extra_state["overcooked.order_timer"]
    assert int(or_[0]) == -1, f"Order should be consumed, got recipe={int(or_[0])}"
    assert int(ot_[0]) == 0, f"Order timer should be cleared, got {int(ot_[0])}"
    print("  Delivery consumed matching order: OK")
    print("  PASSED")


def test_order_delivery_without_matching_order():
    """Test that delivery succeeds even when no active order matches."""
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.core.agent import get_dir_vec_table
    from cogrid.core.grid_object import build_lookup_tables
    from cogrid.envs.overcooked.config import (
        DEFAULT_RECIPES,
        _build_interaction_tables,
        _build_order_tables,
        _build_static_tables,
        _build_type_ids,
        compile_recipes,
        overcooked_interaction_fn,
    )

    scope = "overcooked"
    dir_vec = get_dir_vec_table()
    PICKUP_DROP = 4

    order_config = {
        "spawn_interval": 1,
        "max_active": 3,
        "time_limit": 1000,
    }

    scope_cfg = build_scope_config_from_components(scope)
    itables = _build_interaction_tables(scope)
    type_ids_dict = _build_type_ids(scope)
    recipe_tables = compile_recipes(DEFAULT_RECIPES, scope=scope)
    order_tables = _build_order_tables(order_config, n_recipes=len(DEFAULT_RECIPES))
    static_tables = _build_static_tables(
        scope,
        itables,
        type_ids_dict,
        recipe_tables=recipe_tables,
        order_tables=order_tables,
    )
    scope_cfg["static_tables"] = static_tables
    scope_cfg["interaction_fn"] = overcooked_interaction_fn
    tables = build_lookup_tables(scope=scope)

    type_ids = scope_cfg["type_ids"]
    onion_soup_id = type_ids["onion_soup"]
    delivery_zone_id = type_ids["delivery_zone"]

    print("Order delivery without matching order test:")

    agent_pos = np.array([[2, 3]], dtype=np.int32)
    agent_dir = np.array([0], dtype=np.int32)
    agent_inv = np.array([[onion_soup_id]], dtype=np.int32)
    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = delivery_zone_id
    pp = np.array([[0, 0]], dtype=np.int32)
    pc = np.array([[-1, -1, -1]], dtype=np.int32)
    pt = np.array([30], dtype=np.int32)

    # No active orders
    order_recipe = np.array([-1, -1, -1], dtype=np.int32)
    order_timer = np.array([0, 0, 0], dtype=np.int32)
    order_spawn_counter = np.int32(10)
    order_recipe_counter = np.int32(0)
    order_n_expired = np.int32(0)

    state = _make_state_with_orders(
        agent_pos,
        agent_dir,
        agent_inv,
        otm,
        osm,
        pc,
        pt,
        pp,
        order_recipe=order_recipe,
        order_timer=order_timer,
        order_spawn_counter=order_spawn_counter,
        order_recipe_counter=order_recipe_counter,
        order_n_expired=order_n_expired,
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

    assert state.agent_inv[0, 0] == -1, "Soup should still be delivered"
    or_ = state.extra_state["overcooked.order_recipe"]
    # All orders should still be -1 (no change)
    assert all(int(or_[i]) == -1 for i in range(3)), (
        f"Order arrays should be unchanged, got {[int(or_[i]) for i in range(3)]}"
    )
    print("  Delivery succeeded without matching order: OK")
    print("  PASSED")


def test_order_backward_compat_no_config():
    """Test that no orders config produces identical behavior to pre-order code."""
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.core.agent import get_dir_vec_table
    from cogrid.core.grid_object import build_lookup_tables
    from cogrid.envs.overcooked.config import (
        build_overcooked_extra_state,
        overcooked_interaction_fn,
        overcooked_tick,
    )

    scope = "overcooked"
    dir_vec = get_dir_vec_table()
    PICKUP_DROP = 4

    print("Order backward compat (no config) test:")

    # Build extra_state with no order_config
    otm = np.zeros((7, 7), dtype=np.int32)
    from cogrid.core.grid_object import object_to_idx

    pot_id = object_to_idx("pot", scope=scope)
    onion_id = object_to_idx("onion", scope=scope)
    plate_id = object_to_idx("plate", scope=scope)
    onion_soup_id = object_to_idx("onion_soup", scope=scope)
    delivery_zone_id = object_to_idx("delivery_zone", scope=scope)
    otm[2, 4] = pot_id

    extra = build_overcooked_extra_state({"object_type_map": otm}, scope=scope)

    # Assert no order keys
    assert "overcooked.order_recipe" not in extra, (
        "No order keys should be present without order_config"
    )
    assert "overcooked.order_timer" not in extra
    assert "overcooked.order_spawn_counter" not in extra
    print("  No order keys in extra_state: OK")

    # Full workflow: place 3 onions, cook, pickup, deliver
    scope_cfg = build_scope_config_from_components(scope)
    scope_cfg["interaction_fn"] = overcooked_interaction_fn
    tables = build_lookup_tables(scope=scope)

    agent_pos = np.array([[2, 3]], dtype=np.int32)
    agent_dir = np.array([0], dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    pp = np.array([[2, 4]], dtype=np.int32)
    pc = np.array([[-1, -1, -1]], dtype=np.int32)
    pt = np.array([30], dtype=np.int32)

    # Place 3 onions
    for i in range(3):
        agent_inv = np.array([[onion_id]], dtype=np.int32)
        state = _make_state(agent_pos, agent_dir, agent_inv, otm, osm, pc, pt, pp)
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
        otm = state.object_type_map
        osm = state.object_state_map
        pc = state.extra_state["overcooked.pot_contents"]
        pt = state.extra_state["overcooked.pot_timer"]

    # Cook for 30 ticks
    for _ in range(30):
        _, pt, _ = overcooked_tick(pc, pt)

    # Pickup with plate
    agent_inv = np.array([[plate_id]], dtype=np.int32)
    state = _make_state(agent_pos, agent_dir, agent_inv, otm, osm, pc, pt, pp)
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
    assert state.agent_inv[0, 0] == onion_soup_id, "Should get onion_soup"
    print("  Full workflow without orders: OK")

    # Deliver
    otm_dz = state.object_type_map.copy()
    osm_dz = state.object_state_map.copy()
    otm_dz[2, 5] = delivery_zone_id
    agent_pos_dz = np.array([[2, 4]], dtype=np.int32)
    agent_inv = np.array([[onion_soup_id]], dtype=np.int32)
    pc_out = state.extra_state["overcooked.pot_contents"]
    pt_out = state.extra_state["overcooked.pot_timer"]
    state = _make_state(agent_pos_dz, agent_dir, agent_inv, otm_dz, osm_dz, pc_out, pt_out, pp)
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
    assert state.agent_inv[0, 0] == -1, "Delivery should succeed"
    print("  Delivery without orders: OK")

    print("  PASSED")


# ======================================================================
# Order-aware reward tests
# ======================================================================


def _sv_from_dict(fields, n_agents=2, H=5, W=5):
    """Convert a plain dict of arrays to a StateView, filling missing core fields."""
    import numpy as _np

    from cogrid.backend.state_view import StateView

    defaults = dict(
        agent_pos=_np.zeros((n_agents, 2), dtype=_np.int32),
        agent_dir=_np.zeros(n_agents, dtype=_np.int32),
        agent_inv=_np.full((n_agents, 1), -1, dtype=_np.int32),
        wall_map=_np.zeros((H, W), dtype=_np.int32),
        object_type_map=_np.zeros((H, W), dtype=_np.int32),
        object_state_map=_np.zeros((H, W), dtype=_np.int32),
    )
    _CORE = {
        "agent_pos",
        "agent_dir",
        "agent_inv",
        "wall_map",
        "object_type_map",
        "object_state_map",
    }
    extra = {}
    for k, v in fields.items():
        if k in _CORE:
            defaults[k] = v
        else:
            extra[k] = v
    return StateView(**defaults, extra=extra)


def test_delivery_reward_uses_is_deliverable():
    """Verify IS_DELIVERABLE lookup replaces hardcoded onion_soup check."""
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.core.autowire import build_scope_config_from_components
    from cogrid.envs.overcooked.rewards import delivery_reward

    scope = "overcooked"
    scope_cfg = build_scope_config_from_components(scope)
    type_ids = scope_cfg["type_ids"]
    static_tables = scope_cfg["static_tables"]

    onion_soup_id = type_ids["onion_soup"]
    onion_id = type_ids["onion"]
    dz_id = type_ids["delivery_zone"]

    n_agents = 1
    reward_config = {
        "type_ids": type_ids,
        "n_agents": n_agents,
        "action_pickup_drop_idx": 4,
        "static_tables": static_tables,
    }

    print("test_delivery_reward_uses_is_deliverable:")

    # Agent holds onion_soup (deliverable), faces delivery zone
    otm = np.zeros((5, 5), dtype=np.int32)
    otm[1, 3] = dz_id
    prev_state = _sv_from_dict(
        {
            "agent_pos": np.array([[1, 2]], dtype=np.int32),
            "agent_dir": np.array([0], dtype=np.int32),  # Right -> (1,3)
            "agent_inv": np.array([[onion_soup_id]], dtype=np.int32),
            "object_type_map": otm,
        },
        n_agents=1,
    )
    actions = np.array([4], dtype=np.int32)

    r = delivery_reward(
        prev_state, prev_state, actions, type_ids, n_agents, reward_config=reward_config
    )
    assert float(r[0]) > 0, f"Deliverable item should earn reward, got {float(r[0])}"
    print("  Deliverable item (onion_soup) earns reward: OK")

    # Agent holds onion (NOT deliverable), faces delivery zone
    prev_state2 = _sv_from_dict(
        {
            "agent_pos": np.array([[1, 2]], dtype=np.int32),
            "agent_dir": np.array([0], dtype=np.int32),
            "agent_inv": np.array([[onion_id]], dtype=np.int32),
            "object_type_map": otm,
        },
        n_agents=1,
    )
    r2 = delivery_reward(
        prev_state2, prev_state2, actions, type_ids, n_agents, reward_config=reward_config
    )
    assert float(r2[0]) == 0.0, f"Non-deliverable should get 0, got {float(r2[0])}"
    print("  Non-deliverable item (onion) gets zero: OK")

    print("  PASSED")


def test_delivery_reward_per_recipe_values():
    """Verify per-recipe reward lookup returns correct values per recipe."""
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.core.autowire import build_scope_config_from_components
    from cogrid.envs.overcooked.config import (
        _build_interaction_tables,
        _build_static_tables,
        _build_type_ids,
        compile_recipes,
    )
    from cogrid.envs.overcooked.rewards import delivery_reward

    scope = "overcooked"
    scope_cfg = build_scope_config_from_components(scope)
    type_ids = scope_cfg["type_ids"]

    # Custom recipes with different reward values
    custom_recipes = [
        {
            "ingredients": ["onion", "onion", "onion"],
            "result": "onion_soup",
            "cook_time": 30,
            "reward": 20.0,
        },
        {
            "ingredients": ["tomato", "tomato", "tomato"],
            "result": "tomato_soup",
            "cook_time": 30,
            "reward": 30.0,
        },
    ]
    itables = _build_interaction_tables(scope)
    type_ids_dict = _build_type_ids(scope)
    recipe_tables = compile_recipes(custom_recipes, scope=scope)
    static_tables = _build_static_tables(scope, itables, type_ids_dict, recipe_tables=recipe_tables)

    onion_soup_id = type_ids["onion_soup"]
    tomato_soup_id = type_ids["tomato_soup"]
    dz_id = type_ids["delivery_zone"]
    n_agents = 1

    reward_config = {
        "type_ids": type_ids,
        "n_agents": n_agents,
        "action_pickup_drop_idx": 4,
        "static_tables": static_tables,
    }

    print("test_delivery_reward_per_recipe_values:")

    otm = np.zeros((5, 5), dtype=np.int32)
    otm[1, 3] = dz_id
    actions = np.array([4], dtype=np.int32)

    # Deliver tomato_soup -> should get 30.0 (not 20.0)
    prev_state = _sv_from_dict(
        {
            "agent_pos": np.array([[1, 2]], dtype=np.int32),
            "agent_dir": np.array([0], dtype=np.int32),
            "agent_inv": np.array([[tomato_soup_id]], dtype=np.int32),
            "object_type_map": otm,
        },
        n_agents=1,
    )
    r = delivery_reward(
        prev_state, prev_state, actions, type_ids, n_agents, reward_config=reward_config
    )
    assert float(r[0]) == 30.0, f"Expected 30.0 for tomato_soup, got {float(r[0])}"
    print("  Tomato soup reward = 30.0: OK")

    # Deliver onion_soup -> should get 20.0
    prev_state2 = _sv_from_dict(
        {
            "agent_pos": np.array([[1, 2]], dtype=np.int32),
            "agent_dir": np.array([0], dtype=np.int32),
            "agent_inv": np.array([[onion_soup_id]], dtype=np.int32),
            "object_type_map": otm,
        },
        n_agents=1,
    )
    r2 = delivery_reward(
        prev_state2, prev_state2, actions, type_ids, n_agents, reward_config=reward_config
    )
    assert float(r2[0]) == 20.0, f"Expected 20.0 for onion_soup, got {float(r2[0])}"
    print("  Onion soup reward = 20.0: OK")

    print("  PASSED")


def test_delivery_reward_order_match_required():
    """Verify delivery reward fires only when matching active order exists."""
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.core.autowire import build_scope_config_from_components
    from cogrid.envs.overcooked.config import (
        DEFAULT_RECIPES,
        _build_interaction_tables,
        _build_order_tables,
        _build_static_tables,
        _build_type_ids,
        compile_recipes,
    )
    from cogrid.envs.overcooked.rewards import delivery_reward

    scope = "overcooked"
    scope_cfg = build_scope_config_from_components(scope)
    type_ids = scope_cfg["type_ids"]

    order_config = {"spawn_interval": 1, "max_active": 3, "time_limit": 1000}
    itables = _build_interaction_tables(scope)
    type_ids_dict = _build_type_ids(scope)
    recipe_tables = compile_recipes(DEFAULT_RECIPES, scope=scope)
    order_tables = _build_order_tables(order_config, n_recipes=len(DEFAULT_RECIPES))
    static_tables = _build_static_tables(
        scope, itables, type_ids_dict, recipe_tables=recipe_tables, order_tables=order_tables
    )

    onion_soup_id = type_ids["onion_soup"]
    dz_id = type_ids["delivery_zone"]
    n_agents = 1

    reward_config = {
        "type_ids": type_ids,
        "n_agents": n_agents,
        "action_pickup_drop_idx": 4,
        "static_tables": static_tables,
    }

    print("test_delivery_reward_order_match_required:")

    otm = np.zeros((5, 5), dtype=np.int32)
    otm[1, 3] = dz_id
    actions = np.array([4], dtype=np.int32)

    # Case 1: Active order for recipe 0 (onion_soup), deliver onion_soup -> reward fires
    # prev_state has order active, state has order consumed (recipe=-1)
    prev_sv = _sv_from_dict(
        {
            "agent_pos": np.array([[1, 2]], dtype=np.int32),
            "agent_dir": np.array([0], dtype=np.int32),
            "agent_inv": np.array([[onion_soup_id]], dtype=np.int32),
            "object_type_map": otm,
            "order_recipe": np.array([0, -1, -1], dtype=np.int32),
            "order_timer": np.array([100, 0, 0], dtype=np.int32),
        },
        n_agents=1,
    )
    curr_sv = _sv_from_dict(
        {
            "agent_pos": np.array([[1, 2]], dtype=np.int32),
            "agent_dir": np.array([0], dtype=np.int32),
            "agent_inv": np.array([[-1]], dtype=np.int32),
            "object_type_map": otm,
            "order_recipe": np.array([-1, -1, -1], dtype=np.int32),  # order consumed
            "order_timer": np.array([0, 0, 0], dtype=np.int32),
        },
        n_agents=1,
    )
    r = delivery_reward(prev_sv, curr_sv, actions, type_ids, n_agents, reward_config=reward_config)
    assert float(r[0]) > 0, f"Matching order should earn reward, got {float(r[0])}"
    print("  Matching order -> reward fires: OK")

    # Case 2: No active orders, deliver onion_soup -> reward is zero
    prev_sv2 = _sv_from_dict(
        {
            "agent_pos": np.array([[1, 2]], dtype=np.int32),
            "agent_dir": np.array([0], dtype=np.int32),
            "agent_inv": np.array([[onion_soup_id]], dtype=np.int32),
            "object_type_map": otm,
            "order_recipe": np.array([-1, -1, -1], dtype=np.int32),
            "order_timer": np.array([0, 0, 0], dtype=np.int32),
        },
        n_agents=1,
    )
    curr_sv2 = _sv_from_dict(
        {
            "agent_pos": np.array([[1, 2]], dtype=np.int32),
            "agent_dir": np.array([0], dtype=np.int32),
            "agent_inv": np.array([[-1]], dtype=np.int32),
            "object_type_map": otm,
            "order_recipe": np.array([-1, -1, -1], dtype=np.int32),
            "order_timer": np.array([0, 0, 0], dtype=np.int32),
        },
        n_agents=1,
    )
    r2 = delivery_reward(
        prev_sv2, curr_sv2, actions, type_ids, n_agents, reward_config=reward_config
    )
    assert float(r2[0]) == 0.0, f"No matching order should yield zero, got {float(r2[0])}"
    print("  No matching order -> zero reward: OK")

    print("  PASSED")


def test_expired_order_penalty():
    """Verify ExpiredOrderPenalty returns penalty for newly expired orders."""
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.envs.overcooked.rewards import ExpiredOrderPenalty

    print("test_expired_order_penalty:")

    n_agents = 2
    reward_config = {"n_agents": n_agents, "expired_order_penalty": -5.0}

    # prev_state: order_n_expired=0, state: order_n_expired=2
    prev_sv = _sv_from_dict({"order_n_expired": np.int32(0)}, n_agents=n_agents)
    curr_sv = _sv_from_dict({"order_n_expired": np.int32(2)}, n_agents=n_agents)
    actions = np.array([6, 6], dtype=np.int32)  # Noop

    penalty = ExpiredOrderPenalty()
    r = penalty.compute(prev_sv, curr_sv, actions, reward_config)
    expected = 2 * (-5.0)
    assert float(r[0]) == expected, f"Expected {expected}, got {float(r[0])}"
    assert float(r[1]) == expected, f"Expected {expected} for all agents, got {float(r[1])}"
    print(f"  2 newly expired * -5.0 = {expected} for all agents: OK")

    # No expiry: prev=3, curr=3
    prev_sv2 = _sv_from_dict({"order_n_expired": np.int32(3)}, n_agents=n_agents)
    curr_sv2 = _sv_from_dict({"order_n_expired": np.int32(3)}, n_agents=n_agents)
    r2 = penalty.compute(prev_sv2, curr_sv2, actions, reward_config)
    assert float(r2[0]) == 0.0, f"No new expiry should give 0, got {float(r2[0])}"
    print("  No new expiry -> zero penalty: OK")

    print("  PASSED")


def test_delivery_reward_backward_compat_no_orders():
    """Verify delivery reward fires unconditionally when no order arrays exist."""
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.core.autowire import build_scope_config_from_components
    from cogrid.envs.overcooked.rewards import delivery_reward

    scope = "overcooked"
    scope_cfg = build_scope_config_from_components(scope)
    type_ids = scope_cfg["type_ids"]
    static_tables = scope_cfg["static_tables"]

    onion_soup_id = type_ids["onion_soup"]
    dz_id = type_ids["delivery_zone"]
    n_agents = 1

    reward_config = {
        "type_ids": type_ids,
        "n_agents": n_agents,
        "action_pickup_drop_idx": 4,
        "static_tables": static_tables,
    }

    print("test_delivery_reward_backward_compat_no_orders:")

    otm = np.zeros((5, 5), dtype=np.int32)
    otm[1, 3] = dz_id
    actions = np.array([4], dtype=np.int32)

    # No order arrays -> reward fires unconditionally
    prev_sv = _sv_from_dict(
        {
            "agent_pos": np.array([[1, 2]], dtype=np.int32),
            "agent_dir": np.array([0], dtype=np.int32),
            "agent_inv": np.array([[onion_soup_id]], dtype=np.int32),
            "object_type_map": otm,
        },
        n_agents=1,
    )
    r = delivery_reward(prev_sv, prev_sv, actions, type_ids, n_agents, reward_config=reward_config)
    assert float(r[0]) > 0, f"No orders -> unconditional reward, got {float(r[0])}"
    print("  No order arrays -> reward fires unconditionally: OK")

    print("  PASSED")


def test_static_tables_in_reward_config_via_env():
    """Verify static_tables flows through cogrid_env.py to reward_config."""
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.cogrid_env import CoGridEnv

    print("test_static_tables_in_reward_config_via_env:")

    config = cogrid.envs.cramped_room_config
    env = CoGridEnv(config)
    env.reset(seed=42)

    assert "static_tables" in env._reward_config, "static_tables missing from _reward_config"
    st = env._reward_config["static_tables"]
    assert "IS_DELIVERABLE" in st, "IS_DELIVERABLE missing from static_tables"
    assert "recipe_reward" in st, "recipe_reward missing from static_tables"
    assert "recipe_result" in st, "recipe_result missing from static_tables"
    print("  env._reward_config has static_tables with IS_DELIVERABLE, recipe_reward: OK")

    print("  PASSED")


def test_order_config_validation():
    """Test _build_order_tables edge cases and config validation."""
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.envs.overcooked.config import _build_order_tables

    print("Order config validation test:")

    # Case 1: None -> disabled
    result = _build_order_tables(None, 2)
    assert result == {"order_enabled": False}, f"Expected disabled, got {result}"
    print("  None config -> disabled: OK")

    # Case 2: Valid config with defaults
    result = _build_order_tables({"spawn_interval": 10, "max_active": 5, "time_limit": 100}, 2)
    assert result["order_enabled"] is True
    assert int(result["order_spawn_interval"]) == 10
    assert int(result["order_max_active"]) == 5
    assert int(result["order_time_limit"]) == 100
    assert len(result["order_spawn_cycle"]) == 2, (
        "Uniform weights over 2 recipes should give cycle of length 2, "
        f"got {len(result['order_spawn_cycle'])}"
    )
    assert list(result["order_spawn_cycle"]) == [0, 1]
    print("  Valid config with uniform weights: OK")

    # Case 3: Weighted config
    result = _build_order_tables(
        {"spawn_interval": 10, "max_active": 5, "time_limit": 100, "recipe_weights": [2.0, 1.0]},
        2,
    )
    assert result["order_enabled"] is True
    cycle = list(result["order_spawn_cycle"])
    assert cycle == [0, 0, 1], f"Weights [2.0, 1.0] should give cycle [0, 0, 1], got {cycle}"
    print("  Weighted config [2.0, 1.0] -> cycle [0, 0, 1]: OK")

    print("  PASSED")


def test_delivery_reward_tip_bonus():
    """Verify tip_coefficient produces tip bonus proportional to remaining time."""
    import numpy as np

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401
    from cogrid.core.autowire import build_scope_config_from_components
    from cogrid.envs.overcooked.config import (
        DEFAULT_RECIPES,
        _build_interaction_tables,
        _build_order_tables,
        _build_static_tables,
        _build_type_ids,
        compile_recipes,
    )
    from cogrid.envs.overcooked.rewards import delivery_reward

    scope = "overcooked"
    scope_cfg = build_scope_config_from_components(scope)
    type_ids = scope_cfg["type_ids"]

    order_config = {"spawn_interval": 1, "max_active": 3, "time_limit": 200}
    itables = _build_interaction_tables(scope)
    type_ids_dict = _build_type_ids(scope)
    recipe_tables = compile_recipes(DEFAULT_RECIPES, scope=scope)
    order_tables = _build_order_tables(order_config, n_recipes=len(DEFAULT_RECIPES))
    static_tables = _build_static_tables(
        scope, itables, type_ids_dict, recipe_tables=recipe_tables, order_tables=order_tables
    )

    onion_soup_id = type_ids["onion_soup"]
    dz_id = type_ids["delivery_zone"]
    n_agents = 1

    print("test_delivery_reward_tip_bonus:")

    otm = np.zeros((5, 5), dtype=np.int32)
    otm[1, 3] = dz_id
    actions = np.array([4], dtype=np.int32)

    # Build states: prev has active order with 100 steps remaining, curr has order consumed
    prev_sv = _sv_from_dict(
        {
            "agent_pos": np.array([[1, 2]], dtype=np.int32),
            "agent_dir": np.array([0], dtype=np.int32),
            "agent_inv": np.array([[onion_soup_id]], dtype=np.int32),
            "object_type_map": otm,
            "order_recipe": np.array([0, -1, -1], dtype=np.int32),
            "order_timer": np.array([100, 0, 0], dtype=np.int32),
        },
        n_agents=1,
    )
    curr_sv = _sv_from_dict(
        {
            "agent_pos": np.array([[1, 2]], dtype=np.int32),
            "agent_dir": np.array([0], dtype=np.int32),
            "agent_inv": np.array([[-1]], dtype=np.int32),
            "object_type_map": otm,
            "order_recipe": np.array([-1, -1, -1], dtype=np.int32),
            "order_timer": np.array([0, 0, 0], dtype=np.int32),
        },
        n_agents=1,
    )

    # Case 1: tip_coefficient = 10.0 -> tip = 100/200 * 10.0 = 5.0
    reward_config_with_tip = {
        "type_ids": type_ids,
        "n_agents": n_agents,
        "action_pickup_drop_idx": 4,
        "static_tables": static_tables,
        "tip_coefficient": 10.0,
    }
    r = delivery_reward(
        prev_sv, curr_sv, actions, type_ids, n_agents, reward_config=reward_config_with_tip
    )
    # Base recipe reward (20.0 for onion_soup) + tip (5.0) = 25.0
    expected_tip = 100.0 / 200.0 * 10.0  # 5.0
    base_reward = float(r[0]) - expected_tip
    assert base_reward > 0, f"Base reward should be positive, got {base_reward}"
    assert abs(float(r[0]) - (base_reward + expected_tip)) < 1e-5, (
        f"Reward with tip should be {base_reward + expected_tip}, got {float(r[0])}"
    )
    print(
        f"  tip_coefficient=10.0 -> reward={float(r[0])}"
        f" (base={base_reward} + tip={expected_tip}): OK"
    )

    # Case 2: tip_coefficient absent -> tip = 0 (backward compat)
    reward_config_no_tip = {
        "type_ids": type_ids,
        "n_agents": n_agents,
        "action_pickup_drop_idx": 4,
        "static_tables": static_tables,
    }
    r2 = delivery_reward(
        prev_sv, curr_sv, actions, type_ids, n_agents, reward_config=reward_config_no_tip
    )
    assert float(r2[0]) == base_reward, (
        f"Without tip_coefficient, reward should be {base_reward}, got {float(r2[0])}"
    )
    print(f"  tip_coefficient absent -> reward={float(r2[0])} (no tip): OK")

    print("  PASSED")
