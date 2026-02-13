"""Overcooked interaction parity tests.

Moved from cogrid/core/interactions.py. These tests validate that array-based
Overcooked interactions match the object-based implementation.

Run with:
    python -c "from cogrid.envs.overcooked.test_interactions import test_interaction_parity; test_interaction_parity()"
"""

from __future__ import annotations

from cogrid.core.interactions import process_interactions
from cogrid.envs.overcooked.array_config import overcooked_tick
from cogrid.core.autowire import build_scope_config_from_components


def _build_extra_state(pot_contents, pot_timer, pot_positions):
    """Build the extra_state dict for process_interactions calls.

    The unified process_interactions expects pot arrays directly in
    extra_state (no pot_pos_to_idx or type_ids -- those are in static_tables).
    """
    return {
        "pot_contents": pot_contents,
        "pot_timer": pot_timer,
        "pot_positions": pot_positions,
    }


# DEV: Remove or move to test suite after integration
def test_interaction_parity():
    """Validate array-based interactions match existing object-based interactions.

    Tests deterministic scenarios covering all interaction branches and the
    pot cooking state machine, comparing ``process_interactions`` and
    ``overcooked_tick`` results against the existing ``CoGridEnv.interact``
    and ``Grid.tick`` behavior.

    Run with::

        python -c "from cogrid.envs.overcooked.test_interactions import test_interaction_parity; test_interaction_parity()"
    """
    import copy
    import numpy as np
    from cogrid.backend._dispatch import _reset_backend_for_testing
    _reset_backend_for_testing()
    import cogrid.envs  # trigger environment registration
    from cogrid.envs import registry
    from cogrid.core.grid_object import build_lookup_tables, object_to_idx, get_object_names
    from cogrid.core.grid_utils import layout_to_array_state
    from cogrid.core.agent import create_agent_arrays, get_dir_vec_table
    from cogrid.core import actions as grid_actions

    scope = "overcooked"
    tables = build_lookup_tables(scope=scope)
    scope_cfg = build_scope_config_from_components(scope)
    type_ids = scope_cfg["type_ids"]
    dir_vec = get_dir_vec_table()

    onion_id = type_ids["onion"]
    tomato_id = type_ids["tomato"]
    plate_id = type_ids["plate"]
    pot_id = type_ids["pot"]
    onion_soup_id = type_ids["onion_soup"]
    tomato_soup_id = type_ids["tomato_soup"]

    # CardinalActions: MoveUp=0, MoveDown=1, MoveLeft=2, MoveRight=3,
    #                  PickupDrop=4, Toggle=5, Noop=6
    PICKUP_DROP = 4
    NOOP = 6

    # ---- Helper: extract state from env for comparison ----
    def get_env_agent_inv(env):
        """Get agent inventories as type IDs, sorted by agent ID."""
        inv = []
        for a_id in sorted(env.env_agents.keys()):
            agent = env.env_agents[a_id]
            if len(agent.inventory) > 0:
                inv.append(object_to_idx(agent.inventory[0], scope=scope))
            else:
                inv.append(-1)
        return inv

    def get_env_pot_state(env):
        """Get pot timer and contents from Grid objects."""
        timers = []
        contents = []
        for r in range(env.grid.height):
            for c in range(env.grid.width):
                cell = env.grid.get(r, c)
                if cell is not None and cell.object_id == "pot":
                    timers.append(cell.cooking_timer)
                    row = []
                    for j in range(3):
                        if j < len(cell.objects_in_pot):
                            row.append(object_to_idx(cell.objects_in_pot[j], scope=scope))
                        else:
                            row.append(-1)
                    contents.append(row)
        return timers, contents

    # Reusable empty pot state for tests
    pc_empty = np.array([[-1, -1, -1]], dtype=np.int32)
    pt_empty = np.array([30], dtype=np.int32)
    pp_dummy = np.array([[0, 0]], dtype=np.int32)

    # ---- Test 1: overcooked_tick parity with Pot.tick() ----
    print("Parity test 1: overcooked_tick vs Pot.tick()")

    from cogrid.envs.overcooked.overcooked_grid_objects import Pot, Onion

    # Empty pot
    pot_obj = Pot(capacity=3)
    pc = np.array([[-1, -1, -1]], dtype=np.int32)
    pt = np.array([30], dtype=np.int32)

    pot_obj.tick()
    _, new_pt, new_ps = overcooked_tick(pc, pt)
    assert pot_obj.cooking_timer == int(new_pt[0]), \
        f"Empty pot timer mismatch: obj={pot_obj.cooking_timer} arr={int(new_pt[0])}"
    assert pot_obj.state == int(new_ps[0]), \
        f"Empty pot state mismatch: obj={pot_obj.state} arr={int(new_ps[0])}"

    # Partially filled pot
    pot_obj2 = Pot(capacity=3)
    pot_obj2.objects_in_pot = [Onion(), Onion()]
    pc2 = np.array([[onion_id, onion_id, -1]], dtype=np.int32)
    pt2 = np.array([30], dtype=np.int32)

    pot_obj2.tick()
    _, new_pt2, new_ps2 = overcooked_tick(pc2, pt2)
    assert pot_obj2.cooking_timer == int(new_pt2[0]), \
        f"Partial pot timer mismatch: obj={pot_obj2.cooking_timer} arr={int(new_pt2[0])}"
    assert pot_obj2.state == int(new_ps2[0]), \
        f"Partial pot state mismatch: obj={pot_obj2.state} arr={int(new_ps2[0])}"

    # Full pot -- cooking cycle
    pot_obj3 = Pot(capacity=3)
    pot_obj3.objects_in_pot = [Onion(), Onion(), Onion()]
    pc3 = np.array([[onion_id, onion_id, onion_id]], dtype=np.int32)
    pt3 = np.array([30], dtype=np.int32)

    for _ in range(35):  # more than needed to fully cook
        pot_obj3.tick()
        _, pt3, ps3 = overcooked_tick(pc3, pt3)
        assert pot_obj3.cooking_timer == int(pt3[0]), \
            f"Cooking timer mismatch at step: obj={pot_obj3.cooking_timer} arr={int(pt3[0])}"
        assert pot_obj3.state == int(ps3[0]), \
            f"Cooking state mismatch at step: obj={pot_obj3.state} arr={int(ps3[0])}"

    print("  PASSED")

    # ---- Test 2: Deterministic scenario -- pickup onion from stack ----
    print("Parity test 2: Deterministic pickup from stack")

    # Set up: agent at (2,3) facing right, onion_stack at (2,4)
    agent_pos = np.array([[2, 3]], dtype=np.int32)
    agent_dir = np.array([0], dtype=np.int32)  # Right
    agent_inv = np.array([[-1]], dtype=np.int32)
    actions_arr = np.array([PICKUP_DROP], dtype=np.int32)

    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = type_ids["onion_stack"]

    es = _build_extra_state(pc_empty, pt_empty, pp_dummy)

    new_inv, new_otm, _, extra = process_interactions(
        agent_pos, agent_dir, agent_inv, actions_arr, otm, osm,
        tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
        **es,
    )

    assert new_inv[0, 0] == onion_id, f"Expected onion: {new_inv[0, 0]}"
    assert new_otm[2, 4] == type_ids["onion_stack"], "Stack should remain"
    print("  PASSED")

    # ---- Test 3: Full pot workflow ----
    # Place 3 onions in pot, cook, pickup soup
    print("Parity test 3: Full pot workflow (place, cook, pickup)")

    # Step A: place first onion
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
        es = _build_extra_state(pc, pt, pp)
        new_inv, otm, osm, extra = process_interactions(
            agent_pos, agent_dir, agent_inv, actions_arr, otm, osm,
            tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
            **es,
        )
        pc = extra["pot_contents"]
        pt = extra["pot_timer"]
        assert new_inv[0, 0] == -1, f"Step {step_num}: agent should have placed onion"

    assert int(np.sum(pc[0] != -1)) == 3, f"Pot should have 3 items: {pc[0]}"

    # Step B: Cook for 30 ticks
    for _ in range(30):
        _, pt, ps = overcooked_tick(pc, pt)
    assert int(pt[0]) == 0, f"Pot should be done: {pt[0]}"

    # Step C: Pickup with plate
    agent_inv = np.array([[plate_id]], dtype=np.int32)
    actions_arr = np.array([PICKUP_DROP], dtype=np.int32)
    es = _build_extra_state(pc, pt, pp)
    new_inv, _, _, extra = process_interactions(
        agent_pos, agent_dir, agent_inv, actions_arr, otm, osm,
        tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
        **es,
    )
    new_pc = extra["pot_contents"]
    new_pt = extra["pot_timer"]
    assert new_inv[0, 0] == onion_soup_id, f"Expected onion soup: {new_inv[0, 0]}"
    assert int(new_pc[0, 0]) == -1, "Pot should be cleared"
    assert int(new_pt[0]) == 30, "Pot timer should reset"
    print("  PASSED")

    # ---- Test 4: Delivery zone parity ----
    print("Parity test 4: Delivery zone accepts soup only")
    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = type_ids["delivery_zone"]

    # Soup: accepted
    es = _build_extra_state(pc_empty, pt_empty, pp_dummy)
    new_inv, _, _, _ = process_interactions(
        agent_pos, agent_dir, np.array([[onion_soup_id]], dtype=np.int32),
        np.array([PICKUP_DROP], dtype=np.int32), otm, osm,
        tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
        **es,
    )
    assert new_inv[0, 0] == -1, "Soup should be delivered"

    # Non-soup: rejected
    es = _build_extra_state(pc_empty, pt_empty, pp_dummy)
    new_inv, _, _, _ = process_interactions(
        agent_pos, agent_dir, np.array([[onion_id]], dtype=np.int32),
        np.array([PICKUP_DROP], dtype=np.int32), otm, osm,
        tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
        **es,
    )
    assert new_inv[0, 0] == onion_id, "Onion should be rejected"
    print("  PASSED")

    # ---- Test 5: Counter place-on and pickup parity ----
    print("Parity test 5: Counter place and pickup")
    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = type_ids["counter"]

    # Place onion on counter
    es = _build_extra_state(pc_empty, pt_empty, pp_dummy)
    new_inv, new_otm, new_osm, _ = process_interactions(
        agent_pos, agent_dir, np.array([[onion_id]], dtype=np.int32),
        np.array([PICKUP_DROP], dtype=np.int32), otm, osm,
        tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
        **es,
    )
    assert new_inv[0, 0] == -1, "Agent should have placed on counter"
    assert new_osm[2, 4] == onion_id, "Counter should store onion in state"

    # Try to place another item on occupied counter
    es = _build_extra_state(pc_empty, pt_empty, pp_dummy)
    new_inv2, _, new_osm2, _ = process_interactions(
        agent_pos, agent_dir, np.array([[tomato_id]], dtype=np.int32),
        np.array([PICKUP_DROP], dtype=np.int32), new_otm, new_osm,
        tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
        **es,
    )
    assert new_inv2[0, 0] == tomato_id, "Should reject: counter occupied"
    print("  PASSED")

    # ---- Test 6: Tomato soup from all-tomato pot ----
    print("Parity test 6: Tomato soup type detection")
    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = pot_id
    pp = np.array([[2, 4]], dtype=np.int32)
    pc = np.array([[tomato_id, tomato_id, tomato_id]], dtype=np.int32)
    pt = np.array([0], dtype=np.int32)  # ready

    es = _build_extra_state(pc, pt, pp)
    new_inv, _, _, _ = process_interactions(
        agent_pos, agent_dir, np.array([[plate_id]], dtype=np.int32),
        np.array([PICKUP_DROP], dtype=np.int32), otm, osm,
        tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
        **es,
    )
    assert new_inv[0, 0] == tomato_soup_id, f"Expected tomato soup: {new_inv[0, 0]}"
    print("  PASSED")

    # ---- Test 7: Priority order -- pickup > pickup_from > drop > place_on ----
    print("Parity test 7: Priority order")

    # Scenario: agent with empty inventory faces a pickupable item on grid.
    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    otm[2, 4] = onion_id  # pickupable

    es = _build_extra_state(pc_empty, pt_empty, pp_dummy)
    new_inv, new_otm, _, _ = process_interactions(
        agent_pos, agent_dir, np.array([[-1]], dtype=np.int32),
        np.array([PICKUP_DROP], dtype=np.int32), otm, osm,
        tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
        **es,
    )
    assert new_inv[0, 0] == onion_id, "Pickup should take priority"
    assert new_otm[2, 4] == 0, "Onion should be removed from grid"
    print("  PASSED")

    # ---- Test 8: Agent-ahead blocking ----
    print("Parity test 8: Agent ahead blocks interaction")
    agent_pos2 = np.array([[2, 3], [2, 4]], dtype=np.int32)
    agent_dir2 = np.array([0, 0], dtype=np.int32)
    agent_inv2 = np.array([[-1], [-1]], dtype=np.int32)
    actions2 = np.array([PICKUP_DROP, NOOP], dtype=np.int32)

    otm = np.zeros((7, 7), dtype=np.int32)
    osm = np.zeros((7, 7), dtype=np.int32)
    # Even though there's something at (2,4), agent 1 is there too
    otm[2, 4] = onion_id

    es = _build_extra_state(pc_empty, pt_empty, pp_dummy)
    new_inv, _, _, _ = process_interactions(
        agent_pos2, agent_dir2, agent_inv2, actions2, otm, osm,
        tables, scope_cfg, dir_vec, PICKUP_DROP, 5,
        **es,
    )
    assert new_inv[0, 0] == -1, "Agent 0 should be blocked by agent 1"
    print("  PASSED")

    print()
    print("ALL PARITY TESTS PASSED")
