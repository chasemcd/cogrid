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
