# Testing Patterns

**Analysis Date:** 2026-02-10

## Test Framework

**Runner:**
- unittest (Python standard library)
- Version: No explicit version constraint (built-in)
- Config: No pytest.ini or conftest.py

**Assertion Library:**
- unittest.TestCase assertions: `self.assertTrue()`, `self.assertEqual()`, `self.assertIsInstance()`
- NumPy assertions: `np.testing.assert_array_equal()`, `np.testing.assert_equal()`

**Run Commands:**
```bash
python -m unittest discover
# or specific test file
python -m unittest cogrid.test_gridworld_env
# or specific test class
python -m unittest cogrid.test_overcooked_env.TestOvercookedEnv
# or specific test method
python -m unittest cogrid.test_overcooked_env.TestOvercookedEnv.test_tomato_in_pot
```

## Test File Organization

**Location:**
- Co-located with source code, not in separate test directory
- Tests exist at project root: `cogrid/test_gridworld_env.py`, `cogrid/test_overcooked_env.py`, `cogrid/test_gridworld.py`
- Specialized test module: `cogrid/testing/pettingzoo_test.py`
- Environment-specific tests: `cogrid/envs/search_rescue/search_rescue_tests.py`

**Naming:**
- `test_*.py` prefix for standalone test files
- Test class suffix: `Test*` (e.g., `TestMapEnv`, `TestPettingZooAPI`, `TestOvercookedEnv`)
- Test method prefix: `test_*` (e.g., `test_step`, `test_walls`, `test_agent_actions`, `test_tomato_in_pot`)

**Structure:**
```
cogrid/
├── test_gridworld_env.py           # Environment integration tests
├── test_overcooked_env.py          # Environment-specific behavior tests
├── test_gridworld.py               # Mostly commented out legacy tests
├── testing/
│   └── pettingzoo_test.py          # PettingZoo API compatibility tests
└── envs/
    └── search_rescue/
        └── search_rescue_tests.py  # Environment-specific tests
```

## Test Structure

**Suite Organization:**
```python
class TestMapEnv(unittest.TestCase):
    def tearDown(self) -> None:
        self.env = None

    def _construct_map(self, map_encoding, num_agents=1, start_positions=None, ...):
        """Helper method to construct test environment"""
        dummy_config = { ... }
        self.env = env_cls(dummy_config, map_encoding)
        self.env.reset()
        self.env.update_grid_agents()

    def test_step(self):
        """Check that the step method works at all for all possible actions"""
        self._construct_map(BASE_MAP_2, num_agents=1)
        aid = self.env.agent_ids[0]
        action_dim = self.env.action_space.n
        for i in range(action_dim):
            self.env.step({aid: i})
```

**Patterns:**
- Setup via helper methods: `_construct_map()`, `_setup_agents()`
- Teardown via `tearDown()` method
- Assertion pattern: Direct assertions on state followed by NumPy assertions on positions
- Test isolation: Each test creates a fresh environment

## Mocking

**Framework:** Custom test doubles rather than unittest.mock

**Patterns:**
```python
class DummyAgent(Agent):
    def interact(self, char):
        return GridConstants.FreeSpace

class DummyMapEnv(CoGridEnv):
    def __init__(self, config: dict, test_grid_data: tuple[np.ndarray, np.ndarray]):
        self.test_grid_data: tuple[np.ndarray, np.ndarray] = test_grid_data
        self.start_positions: list[tuple] | None = config.get("start_positions")
        self.start_directions: list[int] | None = config.get("start_directions")
        super().__init__(config=config)

    def _setup_agents(self) -> None:
        for agent_id in range(self.config["num_agents"]):
            agent = Agent(...)
            self.env_agents[agent_id] = agent

    def _generate_encoded_grid_states(self, **kwargs):
        return self.test_grid_data
```

**What to Mock:**
- Grid generation: Override `_generate_encoded_grid_states()` to provide test data
- Agent behavior: Create dummy agents with minimal implementations
- Environment-specific methods: Override to inject test data

**What NOT to Mock:**
- Core grid logic: `Grid.get()`, position updates
- Agent movement and direction: Test actual implementation
- State assertions: Use real grid state to verify behavior

## Fixtures and Factories

**Test Data:**
```python
BASE_MAP = (
    [
        "#######",
        "#     #",
        "#     #",
        "#     #",
        "#     #",
        "#     #",
        "#######",
    ],
    np.zeros((7, 7)),
)

BASE_MAP_2 = (
    ["######", "# S  #", "#    #", "#    #", "#   S#", "######"],
    np.zeros((6, 6)),
)
```

**Location:**
- Module-level constants in test files: `BASE_MAP`, `BASE_MAP_2`, `SR_TEST_MAP`
- Config dictionaries built inline in test methods
- Environment factories for complex setups: `make_env(num_agents=4, layout="...", render_mode="human")`

**Pattern:**
```python
def make_env(num_agents=4, layout="overcooked_cramped_room_v1", render_mode="human"):
    config = N_agent_overcooked_config.copy()
    config["num_agents"] = num_agents
    config["grid"]["layout"] = layout

    registry.register(
        "NAgentOvercooked-V0",
        functools.partial(overcooked.Overcooked, config=config),
    )
    return registry.make("NAgentOvercooked-V0", render_mode=render_mode)
```

## Coverage

**Requirements:** No coverage requirements enforced (no .coverage files, no coverage reporting)

**View Coverage:** Not configured

## Test Types

**Unit Tests:**
- Scope: Individual method behavior (`test_agent_actions`)
- Approach: Create minimal environment, test specific action sequences
- Example: `test_agent_actions()` in `TestMapEnv` - tests all rotation and movement actions

**Integration Tests:**
- Scope: Multi-component behavior (`test_interact_sequence`)
- Approach: Full environment setup with agents, objects, interactions
- Example: `test_tomato_in_pot()` - tests agent pickup, movement, and pot interaction together
- Example: `test_cooking_tomato_soup()` - tests cooking timer, pot state, item pickup

**E2E Tests:**
- Framework: PettingZoo API compliance tests
- Location: `cogrid/testing/pettingzoo_test.py`
- Pattern: Uses upstream test suite: `pettingzoo_test.parallel_api_test(env, num_cycles=1000)`
- Scope: Validates environment conforms to PettingZoo protocol

## Common Patterns

**Async Testing:**
Not used (synchronous environment step calls)

**Error Testing:**
```python
def test_walls(self):
    """Check that the spawned map and base map have walls in the right place"""
    self._construct_map(map_encoding=BASE_MAP, num_agents=0)

    # Make sure wall objects surround the grid
    h, w = self.env.shape
    for row in range(h):
        for col in range(w):
            if row in [0, h - 1] or col in [0, w - 1]:
                cell = self.env.grid.get(row=row, col=col)
                assert isinstance(cell, grid_object.Wall)
```

**State Assertions:**
```python
def test_agent_actions(self):
    a_id = 0
    self._construct_map(BASE_MAP, 1, [(2, 2)], [Directions.Up])

    self.env.step({a_id: Actions.Noop})
    np.testing.assert_array_equal(self.env.env_agents[a_id].pos, [2, 2])
    np.testing.assert_array_equal(
        self.env.grid.grid_agents[a_id].pos, [2, 2]
    )
    np.testing.assert_array_equal(self.env.map_with_agents[2, 2], "1")
```

**Helper Methods for Complex Setup:**
```python
def change_agent_position(self, agent_id, new_pos, new_dir):
    agent = self.env.env_agents[agent_id]
    agent.pos = new_pos
    agent.dir = new_dir
    self.env.update_grid_agents()

def add_agent_to_env(self, agent_id, start_position, start_direction):
    self.env.env_agents[agent_id] = Agent(
        agent_id,
        start_position=start_position,
        start_direction=start_direction,
    )
    # ... setup feature generators ...
    self.env.update_grid_agents()
```

## Test Coverage Status

**Well-tested areas:**
- `cogrid/core/agent.py`: Agent positioning, rotation, movement - tested in `test_agent_actions`
- `cogrid/core/grid.py`: Wall placement, grid structure - tested in `test_walls`
- `cogrid/envs/overcooked/`: Pot interactions, soup cooking - tested extensively in `test_overcooked_env.py`
- PettingZoo API: Compliance verified via `pettingzoo_test.py`

**Partially tested:**
- `cogrid/feature_space/`: Feature generation tested through environment tests, not isolated unit tests
- `cogrid/visualization/`: No dedicated tests observed
- Search & Rescue environment: `search_rescue_tests.py` exists but specific tests not reviewed

**Untested areas:**
- Error conditions and edge cases (many tests commented out due to API changes)
- Multi-agent conflict resolution (commented out `test_agent_conflict`)
- View field-of-view calculations (commented out `test_view`)
- Serialization/deserialization (referenced in build but not in main source)

## Running Tests

**Command-line:**
```bash
# Run all tests
python -m unittest discover -s cogrid -p "test_*.py"

# Run specific test file
python -m unittest cogrid.test_gridworld_env

# Run specific test class
python -m unittest cogrid.test_overcooked_env.TestOvercookedEnv

# Run specific test
python -m unittest cogrid.test_overcooked_env.TestOvercookedEnv.test_tomato_in_pot

# With verbose output
python -m unittest discover -s cogrid -p "test_*.py" -v
```

---

*Testing analysis: 2026-02-10*
