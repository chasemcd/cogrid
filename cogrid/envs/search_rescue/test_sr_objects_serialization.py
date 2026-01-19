"""Roundtrip serialization tests for Search & Rescue stateless objects.

Tests verify that all stateless S&R objects survive get_state/set_state roundtrip.

Object Classification:
- STATELESS (6): MedKit, Pickaxe, Rubble, GreenVictim, PurpleVictim, YellowVictim
  These objects have no internal mutable state beyond presence/absence on the grid.
  They return None from get_extra_state() and roundtrip via state integer alone.

- STATEFUL (1): RedVictim (implemented in Phase 1, tested in test_redvictim_serialization.py)
  Has toggle_countdown and first_toggle_agent for two-step rescue mechanic.

Requirements verified:
- SRCH-01: Victim objects serialize rescue state
  - Green/Purple/Yellow are stateless (no rescue state - single-step rescue)
  - RedVictim has extra state (two-step rescue with countdown)

- SRCH-02: Rubble serializes cleared state
  - Cleared Rubble is removed from grid; uncleared Rubble is stateless

- SRCH-03: Tools serialize ownership/usage state
  - MedKit/Pickaxe are stateless pickupable items
  - Ownership tracked by Agent inventory (Phase 4), not the tool itself

- SRCH-04: All S&R objects audited and serialized
  - All 7 objects analyzed: 6 stateless, 1 implemented
"""

import pytest

from cogrid.envs.search_rescue.search_rescue_grid_objects import (
    MedKit,
    Pickaxe,
    Rubble,
    GreenVictim,
    PurpleVictim,
    YellowVictim,
    RedVictim,
)


class TestStatelessSRObjects:
    """Verify stateless S&R objects need no extra serialization.

    These objects can be fully reconstructed from their object_id and state integer.
    Their get_extra_state() method should return None.
    """

    @pytest.mark.parametrize(
        "obj_class,expected_object_id",
        [
            (MedKit, "medkit"),
            (Pickaxe, "pickaxe"),
            (Rubble, "rubble"),
            (GreenVictim, "green_victim"),
            (PurpleVictim, "purple_victim"),
            (YellowVictim, "yellow_victim"),
        ],
    )
    def test_stateless_object_no_extra_state(self, obj_class, expected_object_id):
        """Stateless objects return None from get_extra_state.

        This confirms these objects have no internal mutable state beyond
        their presence on the grid - they can be serialized via state integer alone.
        """
        obj = obj_class(state=0)

        # Should return None - no extra state to serialize
        assert obj.get_extra_state() is None
        assert obj.get_extra_state(scope="search_rescue") is None

        # Verify object_id matches expected
        assert obj.object_id == expected_object_id

    @pytest.mark.parametrize(
        "obj_class",
        [
            MedKit,
            Pickaxe,
            Rubble,
            GreenVictim,
            PurpleVictim,
            YellowVictim,
        ],
    )
    def test_stateless_object_roundtrip_via_state_integer(self, obj_class):
        """Stateless objects roundtrip via encode/constructor with state integer.

        The full serialization cycle for stateless objects:
        1. Create object with state=0
        2. encode() captures (char, 0, state) tuple
        3. New object created with same state integer
        4. New object is equivalent to original
        """
        obj = obj_class(state=0)

        # encode() returns (char_or_idx, 0, state)
        encoded = obj.encode()
        state_value = encoded[2]  # State is third element of tuple

        # Create new object with same state
        new_obj = obj_class(state=state_value)

        # Verify roundtrip produces equivalent object
        assert new_obj.object_id == obj.object_id
        assert new_obj.state == obj.state
        assert new_obj.get_extra_state() is None


class TestRedVictimIsStateful:
    """Confirm RedVictim is NOT stateless - it has extra state requiring serialization.

    This verifies the Phase 1 implementation is necessary and correctly categorized.
    RedVictim has a two-step rescue mechanic:
    1. Medic/agent with MedKit toggles, starting 30-step countdown
    2. Different agent must toggle within countdown window to complete rescue

    The toggle_countdown and first_toggle_agent attributes are mutable state
    that must be serialized for correct environment restoration.
    """

    def test_redvictim_default_state_returns_none(self):
        """RedVictim at default state (countdown=0, no first_toggle_agent) returns None.

        This is an optimization - no need to serialize when at default state.
        """
        victim = RedVictim(state=0)

        # Default state: toggle_countdown=0, first_toggle_agent not set
        assert victim.toggle_countdown == 0
        assert not hasattr(victim, "first_toggle_agent") or victim.first_toggle_agent is None

        # Should return None in default state
        assert victim.get_extra_state() is None

    def test_redvictim_active_rescue_has_extra_state(self):
        """RedVictim with active rescue (countdown > 0) has extra state to serialize.

        This confirms Phase 1 implementation is correctly handling the stateful case.
        """
        victim = RedVictim(state=0)

        # Simulate first toggle by Medic - starts countdown
        victim.toggle_countdown = 25
        victim.first_toggle_agent = "agent_0"

        # Now has extra state
        extra_state = victim.get_extra_state()
        assert extra_state is not None
        assert extra_state["toggle_countdown"] == 25
        assert extra_state["first_toggle_agent"] == "agent_0"

    def test_redvictim_roundtrip_active_rescue(self):
        """RedVictim with active rescue roundtrips correctly via extra state.

        This is a sanity check that the Phase 1 implementation works.
        """
        victim = RedVictim(state=0)
        victim.toggle_countdown = 15
        victim.first_toggle_agent = "agent_1"

        # Serialize
        extra_state = victim.get_extra_state()

        # Create new victim and restore
        new_victim = RedVictim(state=0)
        new_victim.set_extra_state(extra_state)

        # Verify roundtrip
        assert new_victim.toggle_countdown == 15
        assert new_victim.first_toggle_agent == "agent_1"


class TestToolsAreStateless:
    """Verify MedKit and Pickaxe tools are stateless (SRCH-03).

    Tools like MedKit and Pickaxe have no ownership or usage state stored
    on the object itself. They are simple pickupable items.

    - Ownership is tracked by Agent.inventory (serialized in Phase 4)
    - Usage count is not tracked (tools can be used unlimited times)
    - The only state is presence/absence on the grid
    """

    def test_medkit_is_pickupable_and_stateless(self):
        """MedKit can be picked up by any agent and has no internal state."""
        medkit = MedKit(state=0)

        # can_pickup returns True (any agent can pick up)
        assert medkit.can_pickup(agent=None) is True

        # No extra state
        assert medkit.get_extra_state() is None

        # Roundtrip via state integer
        encoded = medkit.encode()
        new_medkit = MedKit(state=encoded[2])
        assert new_medkit.object_id == medkit.object_id

    def test_pickaxe_is_pickupable_and_stateless(self):
        """Pickaxe can be picked up by any agent and has no internal state."""
        pickaxe = Pickaxe(state=0)

        # can_pickup returns True (any agent can pick up)
        assert pickaxe.can_pickup(agent=None) is True

        # No extra state
        assert pickaxe.get_extra_state() is None

        # Roundtrip via state integer
        encoded = pickaxe.encode()
        new_pickaxe = Pickaxe(state=encoded[2])
        assert new_pickaxe.object_id == pickaxe.object_id


class TestVictimsAreStatelessExceptRed:
    """Verify Green, Purple, Yellow victims are stateless (SRCH-01).

    These victims have single-step rescue:
    - Any adjacent agent toggles -> victim is removed from grid
    - No rescue progress or countdown to track
    - The only state is presence/absence on the grid

    RedVictim is the exception with its two-step rescue mechanic.
    """

    def test_green_victim_single_step_rescue(self):
        """GreenVictim has single-step rescue - no state to track."""
        victim = GreenVictim(state=0)

        # Verify toggle_value is set (reward for rescuing)
        assert victim.toggle_value == 0.1

        # No extra state
        assert victim.get_extra_state() is None

    def test_purple_victim_single_step_rescue(self):
        """PurpleVictim has single-step rescue - no state to track."""
        victim = PurpleVictim(state=0)

        # Verify toggle_value is set
        assert victim.toggle_value == 0.2

        # No extra state
        assert victim.get_extra_state() is None

    def test_yellow_victim_single_step_rescue(self):
        """YellowVictim has single-step rescue (requires Medic) - no state to track."""
        victim = YellowVictim(state=0)

        # Verify toggle_value is set
        assert victim.toggle_value == 0.2

        # No extra state - the Medic requirement is checked at toggle time
        # not stored as victim state
        assert victim.get_extra_state() is None


class TestRubbleIsStateless:
    """Verify Rubble is stateless (SRCH-02).

    Rubble has no internal state:
    - Present on grid = blocking passage
    - Toggled/cleared = removed from grid entirely
    - No "partially cleared" state or damage tracking
    """

    def test_rubble_no_partial_state(self):
        """Rubble has no partial cleared state - it's binary (present or removed)."""
        rubble = Rubble(state=0)

        # Verify toggle_value is set (reward for clearing)
        assert rubble.toggle_value == 0.05

        # Rubble blocks vision
        assert rubble.see_behind() is False

        # No extra state to serialize
        assert rubble.get_extra_state() is None

    def test_rubble_roundtrip(self):
        """Rubble roundtrips via state integer."""
        rubble = Rubble(state=0)

        encoded = rubble.encode()
        new_rubble = Rubble(state=encoded[2])

        assert new_rubble.object_id == rubble.object_id
        assert new_rubble.state == rubble.state
        assert new_rubble.toggle_value == rubble.toggle_value
