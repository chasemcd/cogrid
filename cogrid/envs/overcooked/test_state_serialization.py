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
from cogrid.core.agent import Agent
from cogrid.envs.overcooked.agent import OvercookedAgent
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

    def test_counter_with_pot_soup_roundtrip(self):
        """Test Counter with nested soup object roundtrips with full state (OVER-02).

        Verifies that Counter.obj_placed_on preserves the full object state,
        including the nested object's object_id.
        """
        # Create counter with OnionSoup on it
        counter = grid_object.Counter()
        soup = overcooked_grid_objects.OnionSoup()
        counter.obj_placed_on = soup

        # Serialize
        extra_state = counter.get_extra_state(scope="overcooked")
        self.assertIsNotNone(extra_state)
        self.assertIn("obj_placed_on", extra_state)
        self.assertEqual(extra_state["obj_placed_on"]["object_id"], "onion_soup")

        # Create new counter and restore state
        restored_counter = grid_object.Counter()
        restored_counter.set_extra_state(extra_state, scope="overcooked")

        # Verify obj_placed_on is OnionSoup instance with matching object_id
        self.assertIsNotNone(restored_counter.obj_placed_on)
        self.assertIsInstance(
            restored_counter.obj_placed_on, overcooked_grid_objects.OnionSoup
        )
        self.assertEqual(restored_counter.obj_placed_on.object_id, soup.object_id)

    def test_counter_empty_roundtrip(self):
        """Test Counter with nothing on it returns None from get_extra_state.

        An empty counter has no extra state to serialize - it can be
        reconstructed purely from its object_id and state integer.
        """
        counter = grid_object.Counter()

        # Empty counter should return None from get_extra_state
        extra_state = counter.get_extra_state(scope="overcooked")
        self.assertIsNone(extra_state)

        # Verify we can still create a counter and it has no obj_placed_on
        restored_counter = grid_object.make_object(
            counter.object_id, scope="overcooked"
        )
        self.assertIsInstance(restored_counter, grid_object.Counter)
        self.assertIsNone(restored_counter.obj_placed_on)


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


class TestAgentSerializationRoundtrip(unittest.TestCase):
    """Test agent serialization roundtrips for AGNT-01 and AGNT-02.

    This class verifies that agent state (including inventory with various object
    types) roundtrips correctly through get_state() and from_state(), and that
    domain-specific agent types (OvercookedAgent) are preserved after roundtrip.
    """

    def setUp(self):
        """Create an Overcooked test environment for agent tests."""
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
        # Environment may already be registered from other tests
        try:
            registry.register(
                "OvercookedStateTest",
                functools.partial(overcooked.Overcooked, config=self.config),
            )
        except Exception:
            pass  # Already registered
        self.env = registry.make("OvercookedStateTest")

    def test_agent_holding_onion_soup_roundtrip(self):
        """Test agent holding OnionSoup roundtrips correctly (AGNT-01).

        Verifies that:
        - get_state captures inventory with correct object_id
        - from_state restores OnionSoup instance
        - object_id matches after roundtrip
        """
        self.env.reset(seed=42)

        # Give agent an OnionSoup
        agent = self.env.env_agents[0]
        soup = overcooked_grid_objects.OnionSoup()
        agent.inventory = [soup]

        # Serialize
        state = agent.get_state(scope="overcooked")
        self.assertEqual(len(state["inventory"]), 1)
        self.assertEqual(state["inventory"][0]["object_id"], "onion_soup")

        # Restore via from_state
        restored_agent = Agent.from_state(state, scope="overcooked")

        # Verify restored inventory
        self.assertEqual(len(restored_agent.inventory), 1)
        self.assertIsInstance(
            restored_agent.inventory[0], overcooked_grid_objects.OnionSoup
        )
        self.assertEqual(restored_agent.inventory[0].object_id, soup.object_id)

    def test_agent_holding_tomato_soup_roundtrip(self):
        """Test agent holding TomatoSoup roundtrips correctly (AGNT-01).

        Verifies TomatoSoup with different object_id roundtrips correctly.
        """
        self.env.reset(seed=42)

        # Give agent a TomatoSoup
        agent = self.env.env_agents[0]
        soup = overcooked_grid_objects.TomatoSoup()
        agent.inventory = [soup]

        # Serialize
        state = agent.get_state(scope="overcooked")
        self.assertEqual(len(state["inventory"]), 1)
        self.assertEqual(state["inventory"][0]["object_id"], "tomato_soup")

        # Restore via from_state
        restored_agent = Agent.from_state(state, scope="overcooked")

        # Verify restored inventory
        self.assertEqual(len(restored_agent.inventory), 1)
        self.assertIsInstance(
            restored_agent.inventory[0], overcooked_grid_objects.TomatoSoup
        )
        self.assertEqual(restored_agent.inventory[0].object_id, soup.object_id)

    def test_agent_holding_simple_objects_roundtrip(self):
        """Test agent holding stateless objects (Onion, Tomato, Plate) roundtrips (AGNT-01).

        Verifies that simple stateless objects roundtrip correctly via object_id.
        """
        self.env.reset(seed=42)

        # Test each simple object type
        simple_objects = [
            (overcooked_grid_objects.Onion, "onion"),
            (overcooked_grid_objects.Tomato, "tomato"),
            (overcooked_grid_objects.Plate, "plate"),
        ]

        for obj_class, expected_id in simple_objects:
            with self.subTest(object_type=expected_id):
                agent = self.env.env_agents[0]
                obj = obj_class()
                agent.inventory = [obj]

                # Serialize
                state = agent.get_state(scope="overcooked")
                self.assertEqual(len(state["inventory"]), 1)
                self.assertEqual(state["inventory"][0]["object_id"], expected_id)

                # Restore
                restored_agent = Agent.from_state(state, scope="overcooked")

                # Verify
                self.assertEqual(len(restored_agent.inventory), 1)
                self.assertIsInstance(restored_agent.inventory[0], obj_class)
                self.assertEqual(restored_agent.inventory[0].object_id, expected_id)

    def test_agent_empty_inventory_roundtrip(self):
        """Test agent with empty inventory roundtrips correctly (AGNT-01).

        Verifies that empty inventory [] is preserved after roundtrip.
        """
        self.env.reset(seed=42)

        agent = self.env.env_agents[0]
        agent.inventory = []

        # Serialize
        state = agent.get_state(scope="overcooked")
        self.assertEqual(len(state["inventory"]), 0)
        self.assertEqual(state["inventory"], [])

        # Restore
        restored_agent = Agent.from_state(state, scope="overcooked")

        # Verify empty inventory preserved
        self.assertEqual(len(restored_agent.inventory), 0)
        self.assertEqual(restored_agent.inventory, [])

    def test_agent_full_inventory_roundtrip(self):
        """Test agent with multiple items in inventory roundtrips correctly (AGNT-01).

        Verifies that agents with inventory_capacity > 1 holding multiple items
        have all items restored in correct order.
        """
        self.env.reset(seed=42)

        agent = self.env.env_agents[0]
        # Set higher capacity for this test
        agent.inventory_capacity = 3

        # Add multiple items
        items = [
            overcooked_grid_objects.Onion(),
            overcooked_grid_objects.Tomato(),
            overcooked_grid_objects.Plate(),
        ]
        agent.inventory = items

        # Serialize
        state = agent.get_state(scope="overcooked")
        self.assertEqual(len(state["inventory"]), 3)
        self.assertEqual(state["inventory"][0]["object_id"], "onion")
        self.assertEqual(state["inventory"][1]["object_id"], "tomato")
        self.assertEqual(state["inventory"][2]["object_id"], "plate")

        # Restore
        restored_agent = Agent.from_state(state, scope="overcooked")

        # Verify all items restored in correct order
        self.assertEqual(len(restored_agent.inventory), 3)
        self.assertIsInstance(restored_agent.inventory[0], overcooked_grid_objects.Onion)
        self.assertIsInstance(restored_agent.inventory[1], overcooked_grid_objects.Tomato)
        self.assertIsInstance(restored_agent.inventory[2], overcooked_grid_objects.Plate)
        self.assertEqual(restored_agent.inventory_capacity, 3)

    def test_overcooked_agent_type_preserved(self):
        """Test OvercookedAgent type is preserved after roundtrip (AGNT-02).

        Verifies that serializing and restoring via OvercookedAgent.from_state()
        returns an OvercookedAgent instance, not a base Agent.
        """
        self.env.reset(seed=42)

        # Create OvercookedAgent directly
        agent = OvercookedAgent(
            agent_id="test-agent-0",
            start_position=(2, 2),
            start_direction=Directions.Down,
        )
        agent.inventory = [overcooked_grid_objects.Plate()]

        # Serialize
        state = agent.get_state(scope="overcooked")

        # Restore via OvercookedAgent.from_state (not base Agent)
        restored_agent = OvercookedAgent.from_state(state, scope="overcooked")

        # Verify type is preserved
        self.assertIsInstance(restored_agent, OvercookedAgent)
        # Also verify it's not just a base Agent
        self.assertTrue(type(restored_agent).__name__ == "OvercookedAgent")

    def test_overcooked_agent_can_pickup_behavior(self):
        """Test OvercookedAgent can_pickup behavior works after roundtrip (AGNT-02).

        Verifies that the domain-specific can_pickup() behavior of OvercookedAgent
        is preserved after serialization/deserialization. Specifically tests that
        an agent with a Plate can still pick up from a Pot (special Overcooked rule).
        """
        self.env.reset(seed=42)

        # Create OvercookedAgent with a Plate
        agent = OvercookedAgent(
            agent_id="test-agent-0",
            start_position=(2, 2),
            start_direction=Directions.Down,
            inventory_capacity=1,
        )
        agent.inventory = [overcooked_grid_objects.Plate()]

        # Verify can_pickup behavior before roundtrip
        pot = overcooked_grid_objects.Pot()
        # OvercookedAgent with Plate can pickup from Pot (special rule)
        self.assertTrue(agent.can_pickup(pot))
        # Base Agent cannot (inventory full)
        base_agent = Agent(
            agent_id="base-agent",
            start_position=(2, 2),
            start_direction=Directions.Down,
            inventory_capacity=1,
        )
        base_agent.inventory = [overcooked_grid_objects.Plate()]
        self.assertFalse(base_agent.can_pickup(pot))

        # Serialize and restore OvercookedAgent
        state = agent.get_state(scope="overcooked")
        restored_agent = OvercookedAgent.from_state(state, scope="overcooked")

        # Verify can_pickup behavior preserved after roundtrip
        self.assertTrue(restored_agent.can_pickup(pot))
        self.assertEqual(len(restored_agent.inventory), 1)
        self.assertIsInstance(restored_agent.inventory[0], overcooked_grid_objects.Plate)

    def test_full_env_agent_type_preserved(self):
        """Test environment agents are correct type after set_state (AGNT-02).

        Verifies that after using set_state() on an Overcooked environment,
        the env_agents contain OvercookedAgent instances (not base Agent).
        """
        self.env.reset(seed=42)

        # Give an agent an item to verify full state restoration
        self.env.env_agents[0].inventory = [overcooked_grid_objects.OnionSoup()]

        # Save state
        state = self.env.get_state()

        # Restore to new environment
        env2 = registry.make("OvercookedStateTest")
        env2.set_state(state)

        # Verify agents are OvercookedAgent type
        for agent_id, agent in env2.env_agents.items():
            self.assertIsInstance(
                agent,
                OvercookedAgent,
                f"Agent {agent_id} should be OvercookedAgent, got {type(agent).__name__}",
            )

        # Verify inventory also restored
        self.assertEqual(len(env2.env_agents[0].inventory), 1)
        self.assertIsInstance(
            env2.env_agents[0].inventory[0], overcooked_grid_objects.OnionSoup
        )


if __name__ == "__main__":
    unittest.main()
