"""Roundtrip serialization tests for RedVictim and Door objects.

Tests verify that:
1. RedVictim toggle_countdown and first_toggle_agent survive get_state/set_state roundtrip
2. Door is_open/is_locked state survives roundtrip via state integer
"""

import pytest

from cogrid.core.grid_object import Door
from cogrid.envs.search_rescue.search_rescue_grid_objects import RedVictim


class TestRedVictimSerialization:
    """Tests for RedVictim serialization roundtrip."""

    def test_redvictim_default_state_roundtrip(self):
        """RedVictim in default state should return None from get_extra_state."""
        victim = RedVictim(state=0)

        # Default state: toggle_countdown=0, no first_toggle_agent
        assert victim.toggle_countdown == 0
        assert victim.get_extra_state() is None

    def test_redvictim_active_countdown_roundtrip(self):
        """RedVictim with active countdown should roundtrip correctly."""
        # Create RedVictim with active countdown state
        victim = RedVictim(state=0)
        victim.toggle_countdown = 15
        victim.first_toggle_agent = "agent_0"

        # Serialize
        extra_state = victim.get_extra_state()
        assert extra_state is not None
        assert extra_state["toggle_countdown"] == 15
        assert extra_state["first_toggle_agent"] == "agent_0"

        # Create new RedVictim and restore state
        new_victim = RedVictim(state=0)
        new_victim.set_extra_state(extra_state)

        # Verify roundtrip
        assert new_victim.toggle_countdown == 15
        assert new_victim.first_toggle_agent == "agent_0"

    def test_redvictim_partial_state_roundtrip(self):
        """RedVictim with only countdown (no agent) should roundtrip correctly."""
        victim = RedVictim(state=0)
        victim.toggle_countdown = 10
        # first_toggle_agent not set

        extra_state = victim.get_extra_state()
        assert extra_state is not None
        assert extra_state["toggle_countdown"] == 10

        new_victim = RedVictim(state=0)
        new_victim.set_extra_state(extra_state)

        assert new_victim.toggle_countdown == 10


class TestDoorSerialization:
    """Tests for Door serialization via state integer.

    Door does not need get_extra_state/set_extra_state because is_open
    and is_locked are derived from the state integer in __init__.
    """

    def test_door_locked_roundtrip(self):
        """Door state=0 (locked) should roundtrip correctly."""
        # Create locked door
        door = Door(state=0)
        assert door.is_locked is True
        assert door.is_open is False

        # Get encoded state - encode() returns (char_or_idx, 0, state)
        encoded = door.encode()
        state_value = encoded[2]  # State is third element of tuple
        assert state_value == 0

        # Create new door with same state
        new_door = Door(state=state_value)
        assert new_door.is_locked is True
        assert new_door.is_open is False

    def test_door_closed_unlocked_roundtrip(self):
        """Door state=1 (closed, unlocked) should roundtrip correctly."""
        # Create closed unlocked door
        door = Door(state=1)
        assert door.is_locked is False
        assert door.is_open is False

        # Get encoded state - encode() returns (char_or_idx, 0, state)
        encoded = door.encode()
        state_value = encoded[2]  # State is third element of tuple
        assert state_value == 1

        # Create new door with same state
        new_door = Door(state=state_value)
        assert new_door.is_locked is False
        assert new_door.is_open is False

    def test_door_open_roundtrip(self):
        """Door state=2 (open) should roundtrip correctly."""
        # Create open door
        door = Door(state=2)
        assert door.is_open is True
        assert door.is_locked is False

        # Get encoded state - encode() returns (char_or_idx, 0, state)
        encoded = door.encode()
        state_value = encoded[2]  # State is third element of tuple
        assert state_value == 2

        # Create new door with same state
        new_door = Door(state=state_value)
        assert new_door.is_open is True
        assert new_door.is_locked is False

    def test_door_toggle_updates_state_encoding(self):
        """Toggling door should update the state integer correctly."""
        # Start with closed unlocked door
        door = Door(state=1)
        assert door.is_open is False

        # Toggle to open (simulating toggle behavior)
        door.is_open = True

        # Encode should reflect new state - encode() returns (char_or_idx, 0, state)
        encoded = door.encode()
        state_value = encoded[2]  # State is third element of tuple
        assert state_value == 2  # Open state
