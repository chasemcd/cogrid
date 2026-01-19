# Testing Patterns

**Analysis Date:** 2026-01-19

## Test Framework

**Runner:**
- pytest (version 8.4.2 observed)
- Config: `pyproject.toml`

**Assertion Library:**
- `unittest.TestCase` for test classes
- `numpy.testing` for array assertions
- Standard `assert` statements for simple checks

**Run Commands:**
```bash
# Activate environment first
source /opt/anaconda3/bin/activate cogrid

# Run all tests
python -m pytest

# Run specific test file
python -m pytest cogrid/test_overcooked_env.py

# Run specific test class
python -m pytest cogrid/test_overcooked_env.py::TestOvercookedEnv

# Run specific test
python -m pytest cogrid/test_overcooked_env.py::TestOvercookedEnv::test_tomato_in_pot

# Run with verbose output
python -m pytest -v

# Run tests directly with unittest
python -m unittest cogrid.test_gridworld_env
```

## Test File Organization

**Location:**
- Tests co-located with source code in same package
- No separate `tests/` directory

**Naming:**
- Test files: `test_*.py` (e.g., `test_gridworld_env.py`, `test_overcooked_env.py`)
- Test classes: `Test*` (e.g., `TestMapEnv`, `TestOvercookedEnv`)
- Test methods: `test_*` (e.g., `test_step`, `test_walls`, `test_tomato_in_pot`)

**Structure:**
```
cogrid/
├── test_gridworld.py           # Legacy/commented out tests
├── test_gridworld_env.py       # Core environment tests
├── test_overcooked_env.py      # Overcooked domain tests
├── testing/
│   └── pettingzoo_test.py      # PettingZoo API compliance tests
└── envs/
    └── overcooked/
        └── test_state_serialization.py  # State serialization tests
```

## Test Structure

**Suite Organization:**
```python
import unittest
import numpy as np
from cogrid.core.actions import Actions
from cogrid.core.directions import Directions
from cogrid.envs.overcooked import overcooked_grid_objects

class TestOvercookedEnv(unittest.TestCase):
    def setUp(self):
        """Create test environment before each test."""
        self.env = make_env(num_agents=2, layout="overcooked_cramped_room_v1", render_mode="human")
        self.env.reset()

    def test_tomato_in_pot(self):
        """Test that we can get tomato from the stack and put it in the pot."""
        self.pick_tomato_and_move_to_pot()

        agent_0 = self.env.grid.grid_agents[0]
        pot_tile = self.env.grid.get(*agent_0.front_pos)

        self.assertIsInstance(pot_tile, overcooked_grid_objects.Pot)
        # ... more assertions
```

**Common Patterns:**
- `setUp()` method creates fresh environment for each test
- `tearDown()` method cleans up (sets `self.env = None`)
- Helper methods for common operations (e.g., `pick_tomato_and_move_to_pot()`)
- Descriptive test method docstrings

## Mocking

**Framework:** No explicit mocking framework. Tests use real objects.

**Patterns:**
- Create custom test environments with specific layouts
- Register temporary layouts for testing:
```python
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
```

- Create dummy/test environment classes:
```python
class DummyMapEnv(CoGridEnv):
    def __init__(self, config: dict, test_grid_data: tuple[np.ndarray, np.ndarray]):
        self.test_grid_data = test_grid_data
        super().__init__(config=config)

    def _generate_encoded_grid_states(self, **kwargs):
        return self.test_grid_data
```

**What to Mock:**
- Grid layouts (via custom test layouts)
- Environment configurations
- Initial agent positions and directions

**What NOT to Mock:**
- Core grid mechanics
- Object interactions
- Reward calculations

## Fixtures and Factories

**Test Data:**
```python
# Grid layouts defined as string lists
BASE_MAP = (
    [
        "#######",
        "#     #",
        "#     #",
        "#     #",
        "#######",
    ],
    np.zeros((7, 7)),  # State encoding
)

# Environment factory function
def make_env(num_agents=4, layout="overcooked_cramped_room_v1", render_mode="human"):
    config = N_agent_overcooked_config.copy()
    config["num_agents"] = num_agents
    config["grid"]["layout"] = layout

    registry.register("NAgentOvercooked-V0", functools.partial(overcooked.Overcooked, config=config))
    return registry.make("NAgentOvercooked-V0", render_mode=render_mode)
```

**Location:**
- Test data defined at module level in test files
- No separate fixtures directory

## Coverage

**Requirements:** Not enforced (no coverage config found)

**View Coverage:**
```bash
# Install pytest-cov
pip install pytest-cov

# Run with coverage
python -m pytest --cov=cogrid
```

## Test Types

**Unit Tests:**
- Test individual grid objects: `test_pot_can_place_on`, `test_delivery_zone_can_place_on`
- Test agent actions: `test_agent_actions`, `test_step`
- Test state serialization: `test_basic_get_state`, `test_roundtrip_after_reset`

**Integration Tests:**
- Full environment workflows: `test_cooking_tomato_soup`
- Multi-step interactions: `test_interact_sequence`
- State restore determinism: `test_deterministic_after_restore`

**E2E Tests:**
- PettingZoo API compliance: `test_overcooked_pettingzoo`
```python
def test_overcooked_pettingzoo(self):
    env = registry.make("Overcooked-CrampedRoom-V0")
    pettingzoo_test.parallel_api_test(env, num_cycles=1000)
```

**Fuzz/Random Tests:**
```python
def test_random_actions(self):
    """Test that random actions are valid and do not crash the environment."""
    for _ in range(100):
        action = {0: self.env.action_spaces[0].sample(), 1: self.env.action_spaces[1].sample()}
        obs, reward, _, _, _ = self.env.step(action)
```

## Common Patterns

**Environment Step Sequence:**
```python
def test_step_sequence(self):
    self.env.reset(seed=42)

    # Take actions
    obs, reward, _, _, _ = self.env.step({0: Actions.MoveLeft, 1: Actions.Noop})
    obs, reward, _, _, _ = self.env.step({0: Actions.PickupDrop, 1: Actions.Noop})

    # Verify state
    agent_0 = self.env.grid.grid_agents[0]
    self.assertIsInstance(agent_0.inventory[0], SomeObject)
```

**Array Assertions:**
```python
# Position equality
np.testing.assert_array_equal(self.env.env_agents[a_id].pos, [2, 2])

# Grid state equality
orig_encoding = self.env.grid.encode(scope=self.env.scope)
restored_encoding = env2.grid.encode(scope=env2.scope)
np.testing.assert_array_equal(orig_encoding, restored_encoding)
```

**Object Type Assertions:**
```python
# Check object type
self.assertIsInstance(pot_tile, overcooked_grid_objects.Pot)

# Check object in collection
self.assertTrue(any(
    isinstance(obj, overcooked_grid_objects.Tomato)
    for obj in pot_tile.objects_in_pot
))
```

**State Serialization Tests:**
```python
def test_roundtrip_after_steps(self):
    """Test that state can be saved and restored after taking steps."""
    self.env.reset(seed=42)

    # Take some steps
    for _ in range(10):
        self.env.step({0: Actions.MoveRight, 1: Actions.MoveLeft})

    # Save state
    state = self.env.get_state()

    # Create new environment and restore
    env2 = CoGridEnv(config=self.config)
    env2.set_state(state)

    # Verify timestep
    self.assertEqual(self.env.t, env2.t)

    # Verify agent positions
    for agent_id in self.env.agent_ids:
        np.testing.assert_array_equal(
            self.env.env_agents[agent_id].pos,
            env2.env_agents[agent_id].pos
        )
```

**Error Testing:**
```python
def test_invalid_version(self):
    """Test that invalid state version raises error."""
    self.env.reset(seed=42)
    state = self.env.get_state()

    # Modify version
    state["version"] = "2.0"

    env2 = CoGridEnv(config=self.config)
    with self.assertRaises(ValueError):
        env2.set_state(state)
```

**Determinism Testing:**
```python
def test_rng_state_preservation(self):
    """Test that RNG state is properly preserved and restored."""
    self.env.reset(seed=42)

    # Save state
    state = self.env.get_state()

    # Generate random numbers in env1
    env1_randoms = [self.env.np_random.random() for _ in range(5)]

    # Restore to env2
    env2 = CoGridEnv(config=self.config)
    env2.set_state(state)

    # Generate random numbers in env2
    env2_randoms = [env2.np_random.random() for _ in range(5)]

    # Verify same sequence
    for r1, r2 in zip(env1_randoms, env2_randoms):
        self.assertAlmostEqual(r1, r2, places=10)
```

## Known Test Issues

**Broken Tests:**
- `cogrid/test_gridworld_env.py` has import error: `ImportError: cannot import name 'FIXED_GRIDS' from 'cogrid.constants'`
- `cogrid/test_gridworld.py` is entirely commented out (legacy code)

**Currently Passing Tests (17 total):**
- `cogrid/envs/overcooked/test_state_serialization.py` (11 tests)
- `cogrid/test_overcooked_env.py` (5 tests)
- `cogrid/testing/pettingzoo_test.py` (1 test)

---

*Testing analysis: 2026-01-19*
