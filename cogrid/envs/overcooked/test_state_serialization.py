"""Tests for environment state serialization and deserialization.

This module tests the get_state() and set_state() methods to ensure
complete environment state can be saved and restored accurately.
"""

import unittest
import functools
import numpy as np
import copy

from cogrid.core.actions import Actions
from cogrid.core.directions import Directions
from cogrid.envs.overcooked import overcooked_grid_objects
from cogrid.envs.overcooked import overcooked_features
from cogrid.core import grid_object
from cogrid.envs.overcooked import overcooked
from cogrid import cogrid_env
from cogrid.core import layouts
from cogrid.envs import registry


# Register a simple test layout ('+' is the spawn point character)
layouts.register_layout(
    "state_test_simple",
    [
        "#####",
        "#   #",
        "#+ +#",
        "#   #",
        "#####",
    ],
)

# Register an Overcooked test layout
layouts.register_layout(
    "state_test_overcooked",
    [
        "#######",
        "#CCUCC#",
        "#T   O#",
        "#C   C#",
        "#C=C@C#",
        "#######",
    ],
)


class TestSimpleEnvStateSerialization(unittest.TestCase):
    """Test state serialization for simple CoGridEnv environments."""

    def setUp(self):
        """Create a simple test environment."""
        self.config = {
            "name": "SimpleStateTest",
            "num_agents": 2,
            "action_set": "cardinal_actions",
            "features": ["full_map_encoding"],
            "rewards": [],
            "scope": "global",
            "grid": {"layout": "state_test_simple"},
            "max_steps": 100,
        }
        self.env = cogrid_env.CoGridEnv(config=self.config)

    def test_basic_get_state(self):
        """Test that get_state returns a dictionary with expected keys."""
        self.env.reset(seed=42)
        state = self.env.get_state()

        # Check that all expected keys are present
        expected_keys = {
            "version",
            "config",
            "scope",
            "timestep",
            "cumulative_score",
            "current_layout_id",
            "rng_state",
            "grid",
            "agents",
            "spawn_points",
            "prev_actions",
            "per_agent_reward",
            "per_component_reward",
        }
        self.assertEqual(set(state.keys()), expected_keys)
        self.assertEqual(state["version"], "1.0")
        self.assertEqual(state["timestep"], 0)

    def test_roundtrip_after_reset(self):
        """Test that state can be saved and restored immediately after reset."""
        # Reset env1 with a specific seed
        self.env.reset(seed=42)
        state = self.env.get_state()

        # Create a new environment and restore state
        env2 = cogrid_env.CoGridEnv(config=self.config)
        env2.set_state(state)

        # Verify key attributes match
        self.assertEqual(self.env.t, env2.t)
        self.assertEqual(self.env.cumulative_score, env2.cumulative_score)
        self.assertEqual(len(self.env.env_agents), len(env2.env_agents))

        # Verify agent positions match
        for agent_id in self.env.agent_ids:
            orig_agent = self.env.env_agents[agent_id]
            restored_agent = env2.env_agents[agent_id]
            np.testing.assert_array_equal(orig_agent.pos, restored_agent.pos)
            self.assertEqual(orig_agent.dir, restored_agent.dir)

    def test_roundtrip_after_steps(self):
        """Test that state can be saved and restored after taking steps."""
        # Run environment for several steps
        self.env.reset(seed=42)
        actions = {0: Actions.MoveRight, 1: Actions.MoveLeft}

        for _ in range(10):
            self.env.step(actions)

        # Save state
        state = self.env.get_state()

        # Create new environment and restore
        env2 = cogrid_env.CoGridEnv(config=self.config)
        env2.set_state(state)

        # Verify timestep
        self.assertEqual(self.env.t, env2.t)
        self.assertEqual(self.env.t, 10)

        # Verify agent positions match
        for agent_id in self.env.agent_ids:
            orig_agent = self.env.env_agents[agent_id]
            restored_agent = env2.env_agents[agent_id]
            np.testing.assert_array_equal(orig_agent.pos, restored_agent.pos)
            self.assertEqual(orig_agent.dir, restored_agent.dir)

        # Verify grid encoding matches
        orig_encoding = self.env.grid.encode(scope=self.env.scope)
        restored_encoding = env2.grid.encode(scope=env2.scope)
        np.testing.assert_array_equal(orig_encoding, restored_encoding)

    def test_deterministic_after_restore(self):
        """Test that environment behavior is deterministic after state restoration."""
        # Run env1 for a few steps, save state, continue
        self.env.reset(seed=42)
        for _ in range(5):
            self.env.step({0: Actions.MoveRight, 1: Actions.MoveLeft})

        state = self.env.get_state()

        # Continue env1 for more steps
        actions_sequence = [
            {0: Actions.MoveUp, 1: Actions.MoveDown},
            {0: Actions.MoveLeft, 1: Actions.MoveRight},
            {0: Actions.MoveDown, 1: Actions.MoveUp},
        ]

        env1_results = []
        for actions in actions_sequence:
            obs, rewards, _, _, _ = self.env.step(actions)
            env1_results.append((obs, rewards, copy.deepcopy(self.env.env_agents)))

        # Restore state to env2 and take same actions
        env2 = cogrid_env.CoGridEnv(config=self.config)
        env2.set_state(state)

        env2_results = []
        for actions in actions_sequence:
            obs, rewards, _, _, _ = env2.step(actions)
            env2_results.append((obs, rewards, copy.deepcopy(env2.env_agents)))

        # Verify results match
        for i, ((obs1, rew1, agents1), (obs2, rew2, agents2)) in enumerate(
            zip(env1_results, env2_results)
        ):
            # Check rewards
            for agent_id in rew1.keys():
                self.assertAlmostEqual(
                    rew1[agent_id],
                    rew2[agent_id],
                    msg=f"Rewards differ at step {i} for agent {agent_id}",
                )

            # Check agent positions
            for agent_id in agents1.keys():
                np.testing.assert_array_equal(
                    agents1[agent_id].pos,
                    agents2[agent_id].pos,
                    err_msg=f"Agent positions differ at step {i} for agent {agent_id}",
                )

    def test_rng_state_preservation(self):
        """Test that RNG state is properly preserved and restored."""
        # Initialize with specific seed
        self.env.reset(seed=42)

        # Generate some random numbers
        orig_randoms = [self.env.np_random.random() for _ in range(5)]

        # Save state (RNG has advanced)
        state = self.env.get_state()

        # Generate more random numbers in env1
        env1_next_randoms = [self.env.np_random.random() for _ in range(5)]

        # Restore to env2
        env2 = cogrid_env.CoGridEnv(config=self.config)
        env2.set_state(state)

        # Generate random numbers in env2
        env2_next_randoms = [env2.np_random.random() for _ in range(5)]

        # Verify that env2 generates the same sequence as env1 after restoration
        for i, (r1, r2) in enumerate(zip(env1_next_randoms, env2_next_randoms)):
            self.assertAlmostEqual(
                r1, r2, places=10, msg=f"Random number {i} differs after restoration"
            )


class TestOvercookedStateSerialization(unittest.TestCase):
    """Test state serialization for Overcooked environments with complex objects."""

    def setUp(self):
        """Create an Overcooked test environment."""
        self.config = {
            "name": "OvercookedStateTest",
            "num_agents": 2,
            "action_set": "cardinal_actions",
            "features": ["full_map_encoding"],
            "rewards": ["onion_in_pot_reward", "soup_in_dish_reward"],
            "scope": "overcooked",
            "grid": {"layout": "state_test_overcooked"},
            "max_steps": 1000,
        }

        registry.register(
            "OvercookedStateTest",
            functools.partial(overcooked.Overcooked, config=self.config),
        )
        self.env = registry.make("OvercookedStateTest")

    def test_pot_state_serialization(self):
        """Test that Pot objects with ingredients are properly serialized."""
        self.env.reset(seed=42)

        # Find a pot in the grid
        pot_pos = None
        for row in range(self.env.grid.height):
            for col in range(self.env.grid.width):
                obj = self.env.grid.get(row, col)
                if isinstance(obj, overcooked_grid_objects.Pot):
                    pot_pos = (row, col)
                    break
            if pot_pos:
                break

        self.assertIsNotNone(pot_pos, "No pot found in the grid")

        pot = self.env.grid.get(*pot_pos)

        # Manually add ingredients to the pot
        pot.objects_in_pot = [
            overcooked_grid_objects.Tomato(),
            overcooked_grid_objects.Tomato(),
        ]
        pot.cooking_timer = 25

        # Save state
        state = self.env.get_state()

        # Verify pot state is in object_metadata
        pot_key = f"{pot_pos[0]},{pot_pos[1]}"
        self.assertIn(pot_key, state["grid"]["object_metadata"])
        pot_metadata = state["grid"]["object_metadata"][pot_key]
        self.assertEqual(len(pot_metadata["extra_state"]["objects_in_pot"]), 2)
        self.assertEqual(pot_metadata["extra_state"]["cooking_timer"], 25)

        # Restore state
        env2 = registry.make("OvercookedStateTest")
        env2.set_state(state)

        # Verify pot state was restored
        restored_pot = env2.grid.get(*pot_pos)
        self.assertEqual(len(restored_pot.objects_in_pot), 2)
        self.assertEqual(restored_pot.cooking_timer, 25)
        for obj in restored_pot.objects_in_pot:
            self.assertIsInstance(obj, overcooked_grid_objects.Tomato)

    def test_agent_inventory_serialization(self):
        """Test that agent inventory items are properly serialized."""
        self.env.reset(seed=42)

        # Give agent an item
        agent = self.env.env_agents[0]
        agent.inventory = [overcooked_grid_objects.Plate()]

        # Save and restore
        state = self.env.get_state()
        env2 = registry.make("OvercookedStateTest")
        env2.set_state(state)

        # Verify inventory restored
        restored_agent = env2.env_agents[0]
        self.assertEqual(len(restored_agent.inventory), 1)
        self.assertIsInstance(
            restored_agent.inventory[0], overcooked_grid_objects.Plate
        )

    def test_cooking_pot_roundtrip(self):
        """Test full cooking scenario with state save/restore."""
        self.env.reset(seed=42)

        # Find pot and add ingredients
        pot_pos = None
        for row in range(self.env.grid.height):
            for col in range(self.env.grid.width):
                obj = self.env.grid.get(row, col)
                if isinstance(obj, overcooked_grid_objects.Pot):
                    pot_pos = (row, col)
                    break
            if pot_pos:
                break

        pot = self.env.grid.get(*pot_pos)
        pot.objects_in_pot = [
            overcooked_grid_objects.Tomato(),
            overcooked_grid_objects.Tomato(),
            overcooked_grid_objects.Tomato(),
        ]
        pot.cooking_timer = 15

        # Take a step (should decrement timer)
        self.env.step({0: Actions.Noop, 1: Actions.Noop})
        self.assertEqual(pot.cooking_timer, 14)

        # Save state mid-cooking
        state = self.env.get_state()

        # Continue in env1
        for _ in range(14):
            self.env.step({0: Actions.Noop, 1: Actions.Noop})

        self.assertEqual(pot.cooking_timer, 0)
        self.assertTrue(pot.dish_ready)

        # Restore and verify we can reach same state
        env2 = registry.make("OvercookedStateTest")
        env2.set_state(state)

        pot2 = env2.grid.get(*pot_pos)
        self.assertEqual(pot2.cooking_timer, 14)

        # Continue cooking in env2
        for _ in range(14):
            env2.step({0: Actions.Noop, 1: Actions.Noop})

        self.assertEqual(pot2.cooking_timer, 0)
        self.assertTrue(pot2.dish_ready)

    def test_counter_with_object_serialization(self):
        """Test that Counter objects with items placed on them are serialized."""
        self.env.reset(seed=42)

        # Find a counter
        counter_pos = None
        for row in range(self.env.grid.height):
            for col in range(self.env.grid.width):
                obj = self.env.grid.get(row, col)
                if isinstance(obj, grid_object.Counter):
                    counter_pos = (row, col)
                    break
            if counter_pos:
                break

        self.assertIsNotNone(counter_pos, "No counter found in the grid")

        counter = self.env.grid.get(*counter_pos)
        counter.obj_placed_on = overcooked_grid_objects.Onion()

        # Save and restore
        state = self.env.get_state()
        env2 = registry.make("OvercookedStateTest")
        env2.set_state(state)

        # Verify object on counter restored
        restored_counter = env2.grid.get(*counter_pos)
        self.assertIsNotNone(restored_counter.obj_placed_on)
        self.assertIsInstance(
            restored_counter.obj_placed_on, overcooked_grid_objects.Onion
        )

    def test_pot_cooking_state_roundtrip(self):
        """Test Pot cooking state roundtrip preserves all properties (OVER-01).

        Verifies:
        - cooking_timer value preserved after roundtrip
        - is_cooking property returns same value after roundtrip
        - dish_ready (is_ready) property returns same value after roundtrip
        - objects_in_pot list length and ingredient types match after roundtrip
        """
        # Test mid-cooking state (timer > 0, pot full)
        pot = overcooked_grid_objects.Pot()
        pot.objects_in_pot = [
            overcooked_grid_objects.Onion(),
            overcooked_grid_objects.Onion(),
            overcooked_grid_objects.Onion(),
        ]
        pot.cooking_timer = 15  # Mid-cooking

        # Verify is_cooking is True before roundtrip
        self.assertTrue(pot.is_cooking)
        self.assertFalse(pot.dish_ready)

        # Serialize
        extra_state = pot.get_extra_state(scope="overcooked")
        self.assertIsNotNone(extra_state)
        self.assertEqual(extra_state["cooking_timer"], 15)
        self.assertEqual(len(extra_state["objects_in_pot"]), 3)

        # Create new pot and restore state
        restored_pot = overcooked_grid_objects.Pot()
        restored_pot.set_extra_state(extra_state, scope="overcooked")

        # Verify all properties match
        self.assertEqual(restored_pot.cooking_timer, 15)
        self.assertEqual(len(restored_pot.objects_in_pot), 3)
        self.assertTrue(restored_pot.is_cooking)
        self.assertFalse(restored_pot.dish_ready)
        for obj in restored_pot.objects_in_pot:
            self.assertIsInstance(obj, overcooked_grid_objects.Onion)

        # Test ready state (timer == 0, pot full)
        pot_ready = overcooked_grid_objects.Pot()
        pot_ready.objects_in_pot = [
            overcooked_grid_objects.Tomato(),
            overcooked_grid_objects.Tomato(),
            overcooked_grid_objects.Tomato(),
        ]
        pot_ready.cooking_timer = 0  # Ready

        # Verify dish_ready is True before roundtrip
        self.assertFalse(pot_ready.is_cooking)
        self.assertTrue(pot_ready.dish_ready)

        # Serialize and restore
        extra_state_ready = pot_ready.get_extra_state(scope="overcooked")
        restored_pot_ready = overcooked_grid_objects.Pot()
        restored_pot_ready.set_extra_state(extra_state_ready, scope="overcooked")

        # Verify ready state preserved
        self.assertEqual(restored_pot_ready.cooking_timer, 0)
        self.assertEqual(len(restored_pot_ready.objects_in_pot), 3)
        self.assertFalse(restored_pot_ready.is_cooking)
        self.assertTrue(restored_pot_ready.dish_ready)
        for obj in restored_pot_ready.objects_in_pot:
            self.assertIsInstance(obj, overcooked_grid_objects.Tomato)


class TestStatelessObjectsRoundtrip(unittest.TestCase):
    """Test roundtrip serialization for stateless Overcooked objects.

    These objects have no internal state beyond their type - they can be
    reconstructed purely from their object_id. Their get_extra_state() should
    return None.

    Verifies requirements OVER-03, OVER-04, OVER-05, OVER-06.
    """

    def test_onion_roundtrip(self):
        """Test Onion is stateless and roundtrips via object_id."""
        onion = overcooked_grid_objects.Onion()

        # Stateless objects should return None from get_extra_state
        extra_state = onion.get_extra_state(scope="overcooked")
        self.assertIsNone(extra_state)

        # Recreate via object_id
        restored = grid_object.make_object(onion.object_id, scope="overcooked")
        self.assertIsInstance(restored, overcooked_grid_objects.Onion)
        self.assertEqual(restored.object_id, onion.object_id)

    def test_tomato_roundtrip(self):
        """Test Tomato is stateless and roundtrips via object_id."""
        tomato = overcooked_grid_objects.Tomato()

        extra_state = tomato.get_extra_state(scope="overcooked")
        self.assertIsNone(extra_state)

        restored = grid_object.make_object(tomato.object_id, scope="overcooked")
        self.assertIsInstance(restored, overcooked_grid_objects.Tomato)
        self.assertEqual(restored.object_id, tomato.object_id)

    def test_plate_roundtrip(self):
        """Test Plate is stateless and roundtrips via object_id (OVER-03).

        Plate is stateless - soup is a separate object (OnionSoup/TomatoSoup),
        not a state of the plate.
        """
        plate = overcooked_grid_objects.Plate()

        extra_state = plate.get_extra_state(scope="overcooked")
        self.assertIsNone(extra_state)

        restored = grid_object.make_object(plate.object_id, scope="overcooked")
        self.assertIsInstance(restored, overcooked_grid_objects.Plate)
        self.assertEqual(restored.object_id, plate.object_id)

    def test_onion_soup_roundtrip(self):
        """Test OnionSoup is stateless and roundtrips via object_id."""
        soup = overcooked_grid_objects.OnionSoup()

        extra_state = soup.get_extra_state(scope="overcooked")
        self.assertIsNone(extra_state)

        restored = grid_object.make_object(soup.object_id, scope="overcooked")
        self.assertIsInstance(restored, overcooked_grid_objects.OnionSoup)
        self.assertEqual(restored.object_id, soup.object_id)

    def test_tomato_soup_roundtrip(self):
        """Test TomatoSoup is stateless and roundtrips via object_id."""
        soup = overcooked_grid_objects.TomatoSoup()

        extra_state = soup.get_extra_state(scope="overcooked")
        self.assertIsNone(extra_state)

        restored = grid_object.make_object(soup.object_id, scope="overcooked")
        self.assertIsInstance(restored, overcooked_grid_objects.TomatoSoup)
        self.assertEqual(restored.object_id, soup.object_id)

    def test_delivery_zone_roundtrip(self):
        """Test DeliveryZone is stateless and roundtrips via object_id (OVER-06)."""
        zone = overcooked_grid_objects.DeliveryZone()

        extra_state = zone.get_extra_state(scope="overcooked")
        self.assertIsNone(extra_state)

        restored = grid_object.make_object(zone.object_id, scope="overcooked")
        self.assertIsInstance(restored, overcooked_grid_objects.DeliveryZone)
        self.assertEqual(restored.object_id, zone.object_id)

    def test_onion_stack_stateless(self):
        """Test OnionStack has no count state - infinite source by design (OVER-04).

        OnionStack represents an infinite pile of onions. There is no
        'remaining count' to track - agents can always pick up onions from it.
        """
        stack = overcooked_grid_objects.OnionStack()

        # Should return None - no state to serialize
        extra_state = stack.get_extra_state(scope="overcooked")
        self.assertIsNone(extra_state)

        # Verify it's an infinite source (can_pickup_from returns True)
        # This is a design property, not a serialization property
        self.assertTrue(stack.can_pickup_from(agent=None))

        restored = grid_object.make_object(stack.object_id, scope="overcooked")
        self.assertIsInstance(restored, overcooked_grid_objects.OnionStack)

    def test_tomato_stack_stateless(self):
        """Test TomatoStack has no count state - infinite source by design (OVER-05).

        TomatoStack represents an infinite pile of tomatoes. There is no
        'remaining count' to track - agents can always pick up tomatoes from it.
        """
        stack = overcooked_grid_objects.TomatoStack()

        extra_state = stack.get_extra_state(scope="overcooked")
        self.assertIsNone(extra_state)

        self.assertTrue(stack.can_pickup_from(agent=None))

        restored = grid_object.make_object(stack.object_id, scope="overcooked")
        self.assertIsInstance(restored, overcooked_grid_objects.TomatoStack)

    def test_plate_stack_stateless(self):
        """Test PlateStack has no count state - infinite source by design.

        PlateStack represents an infinite pile of plates. There is no
        'remaining count' to track - agents can always pick up plates from it.
        """
        stack = overcooked_grid_objects.PlateStack()

        extra_state = stack.get_extra_state(scope="overcooked")
        self.assertIsNone(extra_state)

        self.assertTrue(stack.can_pickup_from(agent=None))

        restored = grid_object.make_object(stack.object_id, scope="overcooked")
        self.assertIsInstance(restored, overcooked_grid_objects.PlateStack)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for state serialization."""

    def setUp(self):
        """Create a test environment."""
        self.config = {
            "name": "EdgeCaseTest",
            "num_agents": 2,
            "action_set": "cardinal_actions",
            "features": ["full_map_encoding"],
            "rewards": [],
            "scope": "global",
            "grid": {"layout": "state_test_simple"},
            "max_steps": 100,
        }
        self.env = cogrid_env.CoGridEnv(config=self.config)

    def test_empty_inventory(self):
        """Test serialization with empty agent inventories."""
        self.env.reset(seed=42)

        # Ensure inventories are empty
        for agent in self.env.env_agents.values():
            agent.inventory = []

        state = self.env.get_state()
        env2 = cogrid_env.CoGridEnv(config=self.config)
        env2.set_state(state)

        # Verify empty inventories restored
        for agent_id in self.env.agent_ids:
            self.assertEqual(len(env2.env_agents[agent_id].inventory), 0)

    def test_invalid_version(self):
        """Test that invalid state version raises error."""
        self.env.reset(seed=42)
        state = self.env.get_state()

        # Modify version
        state["version"] = "2.0"

        env2 = cogrid_env.CoGridEnv(config=self.config)
        with self.assertRaises(ValueError):
            env2.set_state(state)


if __name__ == "__main__":
    unittest.main()
