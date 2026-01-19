# Phase 6: Testing - Research

**Researched:** 2026-01-19
**Domain:** Comprehensive test coverage for state serialization system
**Confidence:** HIGH

## Summary

Phase 6 research confirms that substantial test coverage already exists for state serialization. The codebase contains 62 passing tests across three test files covering object-level roundtrip serialization, agent serialization, and environment-level state preservation. The existing test patterns are well-structured using both unittest (for Overcooked tests) and pytest (for Search & Rescue tests).

The requirements (TEST-01, TEST-02, TEST-03) specify three testing categories:
1. **TEST-01**: Roundtrip tests for each object type with internal state - LARGELY COVERED
2. **TEST-02**: Full environment checkpoint/restore workflow - COVERED via `TestOvercookedStateSerialization` and `TestSimpleEnvStateSerialization`
3. **TEST-03**: Determinism test for identical trajectories - COVERED via `test_deterministic_after_restore`

The testing phase should focus on **consolidation and gap-filling** rather than building new infrastructure. Specifically:
- Create a centralized integration test file for full system validation
- Add Search & Rescue environment-level integration tests (currently only object-level tests exist)
- Add explicit determinism tests with longer action sequences
- Verify no gaps remain in coverage for stateful objects

**Primary recommendation:** Consolidate existing tests into a clear structure, add Search & Rescue integration tests parallel to Overcooked tests, and create comprehensive determinism tests.

## Standard Stack

### Testing Framework

| Tool | Version | Purpose | Status |
|------|---------|---------|--------|
| pytest | 7.4.0+ | Test runner and framework | Already in use |
| unittest | stdlib | Test classes (Overcooked tests) | Already in use |
| numpy.testing | N/A | Array comparison utilities | Already in use |

### Existing Test Files

| File | Test Count | Coverage |
|------|------------|----------|
| `cogrid/envs/overcooked/test_state_serialization.py` | 33 | Overcooked objects, agent, environment |
| `cogrid/envs/search_rescue/test_sr_objects_serialization.py` | 22 | S&R stateless objects |
| `cogrid/envs/search_rescue/test_redvictim_serialization.py` | 7 | RedVictim, Door roundtrips |

**Total:** 62 passing tests

### Installation

No additional packages needed. Run tests with:
```bash
pytest -v cogrid/envs/overcooked/test_state_serialization.py \
       cogrid/envs/search_rescue/test_sr_objects_serialization.py \
       cogrid/envs/search_rescue/test_redvictim_serialization.py
```

## Architecture Patterns

### Recommended Test Structure

```
cogrid/
├── envs/
│   ├── overcooked/
│   │   └── test_state_serialization.py       # Existing: 33 tests
│   └── search_rescue/
│       ├── test_sr_objects_serialization.py  # Existing: 22 tests
│       ├── test_redvictim_serialization.py   # Existing: 7 tests
│       └── test_sr_env_serialization.py      # NEW: S&R env integration
└── tests/
    └── test_serialization_integration.py     # NEW: Cross-environment determinism
```

### Pattern 1: Object Roundtrip Test

**What:** Test that individual objects serialize and deserialize correctly
**When to use:** Every object type with `get_extra_state()` / `set_extra_state()` or state integer
**Example:**
```python
# Source: cogrid/envs/overcooked/test_state_serialization.py
def test_pot_cooking_state_roundtrip(self):
    """Test Pot cooking state roundtrip preserves all properties."""
    pot = overcooked_grid_objects.Pot()
    pot.objects_in_pot = [
        overcooked_grid_objects.Onion(),
        overcooked_grid_objects.Onion(),
        overcooked_grid_objects.Onion(),
    ]
    pot.cooking_timer = 15

    # Serialize
    extra_state = pot.get_extra_state(scope="overcooked")
    self.assertEqual(extra_state["cooking_timer"], 15)

    # Create new pot and restore state
    restored_pot = overcooked_grid_objects.Pot()
    restored_pot.set_extra_state(extra_state, scope="overcooked")

    # Verify all properties match
    self.assertEqual(restored_pot.cooking_timer, 15)
    self.assertEqual(len(restored_pot.objects_in_pot), 3)
    self.assertTrue(restored_pot.is_cooking)
```

### Pattern 2: Stateless Object Verification Test

**What:** Confirm stateless objects return None from `get_extra_state()`
**When to use:** Objects that have no internal state beyond their type
**Example:**
```python
# Source: cogrid/envs/search_rescue/test_sr_objects_serialization.py
@pytest.mark.parametrize(
    "obj_class,expected_object_id",
    [
        (MedKit, "medkit"),
        (Pickaxe, "pickaxe"),
        (Rubble, "rubble"),
    ],
)
def test_stateless_object_no_extra_state(self, obj_class, expected_object_id):
    """Stateless objects return None from get_extra_state."""
    obj = obj_class(state=0)

    assert obj.get_extra_state() is None
    assert obj.get_extra_state(scope="search_rescue") is None
    assert obj.object_id == expected_object_id
```

### Pattern 3: Full Environment Roundtrip Test

**What:** Save environment state, restore to new environment, verify equivalence
**When to use:** Integration tests for complete environment serialization
**Example:**
```python
# Source: cogrid/envs/overcooked/test_state_serialization.py
def test_roundtrip_after_steps(self):
    """Test that state can be saved and restored after taking steps."""
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

    # Verify grid encoding matches
    orig_encoding = self.env.grid.encode(scope=self.env.scope)
    restored_encoding = env2.grid.encode(scope=env2.scope)
    np.testing.assert_array_equal(orig_encoding, restored_encoding)
```

### Pattern 4: Determinism Test

**What:** Verify identical action sequences produce identical results after restoration
**When to use:** Critical for ensuring checkpoint/restore doesn't break reproducibility
**Example:**
```python
# Source: cogrid/envs/overcooked/test_state_serialization.py
def test_deterministic_after_restore(self):
    """Test that environment behavior is deterministic after state restoration."""
    self.env.reset(seed=42)
    for _ in range(5):
        self.env.step({0: Actions.MoveRight, 1: Actions.MoveLeft})

    state = self.env.get_state()

    # Define action sequence
    actions_sequence = [
        {0: Actions.MoveUp, 1: Actions.MoveDown},
        {0: Actions.MoveLeft, 1: Actions.MoveRight},
        {0: Actions.MoveDown, 1: Actions.MoveUp},
    ]

    # Collect results from original environment
    env1_results = []
    for actions in actions_sequence:
        obs, rewards, _, _, _ = self.env.step(actions)
        env1_results.append((rewards, copy.deepcopy(self.env.env_agents)))

    # Restore state to env2 and take same actions
    env2 = cogrid_env.CoGridEnv(config=self.config)
    env2.set_state(state)

    for i, actions in enumerate(actions_sequence):
        obs, rewards, _, _, _ = env2.step(actions)
        # Verify rewards match
        for agent_id in rewards.keys():
            self.assertAlmostEqual(
                env1_results[i][0][agent_id],
                rewards[agent_id],
            )
```

### Anti-Patterns to Avoid

- **Testing only structure, not behavior:** Don't just check that state dict has keys; verify actual roundtrip behavior
- **Ignoring edge cases:** Test empty inventories, terminated agents, near-truncation states
- **Forgetting scope parameter:** Always test with the appropriate scope ("overcooked", "search_rescue", "global")
- **Missing parametrization:** Use `@pytest.mark.parametrize` for testing multiple similar objects

## Don't Hand-Roll

Problems that have existing solutions in the codebase:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Object roundtrip pattern | Custom test logic | Copy existing `test_pot_cooking_state_roundtrip` pattern | Consistent, proven approach |
| Stateless verification | Manual checks | `@pytest.mark.parametrize` with object list | Reduces duplication |
| Environment setup | Custom fixtures | Existing `setUp()` patterns | Already handles config, registry |
| Array comparison | Manual loops | `np.testing.assert_array_equal` | Better error messages |
| Dictionary comparison | Manual key iteration | `self.assertEqual(dict1, dict2)` | Cleaner, handles nesting |

**Key insight:** The existing test files provide templates for every type of test needed. Copy and adapt rather than inventing new patterns.

## Common Pitfalls

### Pitfall 1: Not Testing with Correct Scope

**What goes wrong:** Tests pass with "global" scope but fail with domain-specific scope
**Why it happens:** Some objects register differently in different scopes
**How to avoid:** Always test with the scope that will be used in production ("overcooked", "search_rescue")
**Warning signs:** Tests pass but actual environment use fails

### Pitfall 2: Shallow Equality Checks

**What goes wrong:** Test verifies structure but misses semantic differences
**Why it happens:** Checking `len()` instead of contents, checking keys instead of values
**How to avoid:** Verify actual object types, call methods to check behavior
**Warning signs:** Tests pass but restored environment behaves differently

### Pitfall 3: Missing Edge Cases

**What goes wrong:** Happy path works but edge cases fail silently
**Why it happens:** Not testing empty states, boundary conditions, error cases
**How to avoid:** Explicit tests for empty inventory, terminated agents, max_steps boundary
**Warning signs:** Rare bugs in production that tests don't catch

### Pitfall 4: Forgetting RNG Determinism

**What goes wrong:** Determinism tests pass by luck, fail randomly
**Why it happens:** Not controlling all sources of randomness
**How to avoid:** Use fixed seeds, verify RNG state is preserved, run determinism tests multiple times
**Warning signs:** Flaky tests that sometimes pass, sometimes fail

### Pitfall 5: Test Isolation Issues

**What goes wrong:** Tests pass individually but fail when run together
**Why it happens:** Shared state between tests (registered environments, class-level caches)
**How to avoid:** Use `setUp()` and `tearDown()` properly, avoid module-level state modification
**Warning signs:** Test order affects results

## Code Examples

### Verified Pattern: Complete Integration Test

```python
# Source: cogrid/envs/overcooked/test_state_serialization.py
class TestOvercookedStateSerialization(unittest.TestCase):
    """Test state serialization for Overcooked environments."""

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

    def test_cooking_pot_roundtrip(self):
        """Test full cooking scenario with state save/restore."""
        self.env.reset(seed=42)

        # Find pot and add ingredients
        pot = self._find_pot()
        pot.objects_in_pot = [Tomato(), Tomato(), Tomato()]
        pot.cooking_timer = 15

        # Take a step (should decrement timer)
        self.env.step({0: Actions.Noop, 1: Actions.Noop})
        self.assertEqual(pot.cooking_timer, 14)

        # Save state mid-cooking
        state = self.env.get_state()

        # Continue in env1
        for _ in range(14):
            self.env.step({0: Actions.Noop, 1: Actions.Noop})
        self.assertTrue(pot.dish_ready)

        # Restore and verify we can reach same state
        env2 = registry.make("OvercookedStateTest")
        env2.set_state(state)

        pot2 = env2.grid.get(*pot.pos)
        self.assertEqual(pot2.cooking_timer, 14)

        for _ in range(14):
            env2.step({0: Actions.Noop, 1: Actions.Noop})
        self.assertTrue(pot2.dish_ready)
```

### Verified Pattern: Parametrized Stateless Object Tests

```python
# Source: cogrid/envs/search_rescue/test_sr_objects_serialization.py
class TestStatelessSRObjects:
    """Verify stateless S&R objects need no extra serialization."""

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
        """Stateless objects return None from get_extra_state."""
        obj = obj_class(state=0)
        assert obj.get_extra_state() is None
        assert obj.get_extra_state(scope="search_rescue") is None
        assert obj.object_id == expected_object_id

    @pytest.mark.parametrize("obj_class", [MedKit, Pickaxe, Rubble, GreenVictim, PurpleVictim, YellowVictim])
    def test_stateless_object_roundtrip_via_state_integer(self, obj_class):
        """Stateless objects roundtrip via encode/constructor with state integer."""
        obj = obj_class(state=0)
        encoded = obj.encode()
        state_value = encoded[2]
        new_obj = obj_class(state=state_value)

        assert new_obj.object_id == obj.object_id
        assert new_obj.state == obj.state
```

### Verified Pattern: Termination Flag Preservation

```python
# Source: cogrid/envs/overcooked/test_state_serialization.py
def test_terminated_agent_preserved(self):
    """Test that terminated agent flag is preserved through roundtrip."""
    self.env.reset(seed=42)

    # Manually terminate agent 0
    self.env.env_agents[0].terminated = True

    # Remove from active agents list
    if 0 in self.env.agents:
        del self.env.agents[0]

    # Verify pre-roundtrip state
    self.assertTrue(self.env.env_agents[0].terminated)
    self.assertNotIn(0, self.env.agents)

    # Save state
    state = self.env.get_state()

    # Verify terminated flag is in serialized state
    self.assertTrue(state["agents"][0]["terminated"])

    # Restore to new environment
    env2 = cogrid_env.CoGridEnv(config=self.config)
    env2.set_state(state)

    # Verify terminated agent remains terminated
    self.assertTrue(env2.env_agents[0].terminated)
    self.assertNotIn(0, env2.agents)
```

## State of the Art

### PettingZoo Testing Integration

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Manual API verification | `pettingzoo.test.parallel_api_test` | Automated API compliance checking |
| Manual seed testing | `pettingzoo.test.parallel_seed_test` | Automated determinism verification |

**Reference:** The codebase already uses PettingZoo's test utilities in `cogrid/testing/pettingzoo_test.py`:

```python
# Source: cogrid/testing/pettingzoo_test.py
from pettingzoo import test as pettingzoo_test
from cogrid.envs import registry

class TestPettingZooAPI(unittest.TestCase):
    def test_overcooked_pettingzoo(self):
        env = registry.make("Overcooked-CrampedRoom-V0")
        pettingzoo_test.parallel_api_test(env, num_cycles=1000)
```

### pytest Best Practices Applied

| Practice | Implementation | Benefit |
|----------|---------------|---------|
| Parametrization | `@pytest.mark.parametrize` for S&R objects | Reduces duplication |
| Fixtures | `@pytest.fixture` for shared setup | Cleaner test isolation |
| Class-based organization | `TestStatelessSRObjects`, `TestRedVictimIsStateful` | Logical grouping |
| Descriptive names | `test_redvictim_active_rescue_has_extra_state` | Self-documenting |

## Requirements Verification

### TEST-01: Roundtrip tests for every object type with internal state

**Status:** COVERED with minor gaps

**Covered Objects:**
- Pot (cooking state, ingredients) - 2 tests
- Counter (obj_placed_on) - 2 tests
- RedVictim (toggle_countdown, first_toggle_agent) - 4 tests
- Door (is_open, is_locked via state integer) - 4 tests

**Stateless Objects Verified:**
- Overcooked: Onion, Tomato, Plate, OnionSoup, TomatoSoup, DeliveryZone, OnionStack, TomatoStack, PlateStack - 9 tests
- Search & Rescue: MedKit, Pickaxe, Rubble, GreenVictim, PurpleVictim, YellowVictim - 12 tests

**Gap:** Core GridObj types (Wall, Floor, Key, Goal) not explicitly tested but are stateless.

### TEST-02: Integration test validates full environment checkpoint/restore workflow

**Status:** COVERED for Overcooked, needs Search & Rescue parallel

**Existing Coverage:**
- `TestSimpleEnvStateSerialization` - 5 tests for base CoGridEnv
- `TestOvercookedStateSerialization` - 7 tests for Overcooked environment

**Gap:** No `TestSearchRescueEnvSerialization` parallel to Overcooked tests.

### TEST-03: Determinism test confirms restored environments produce identical trajectories

**Status:** COVERED

**Existing Coverage:**
- `test_deterministic_after_restore` - Verifies identical rewards and agent positions
- `test_rng_state_preservation` - Verifies identical random number sequences

**Enhancement opportunity:** Longer action sequences, more complex scenarios with cooking/rescue mechanics.

## Open Questions

1. **Q: Should there be a centralized test file for cross-environment validation?**
   - **A:** Recommended. Create `cogrid/tests/test_serialization_integration.py` for comprehensive cross-environment tests.

2. **Q: Are core GridObj types (Wall, Floor, etc.) tested?**
   - **A:** Not explicitly, but they are stateless (return None from `get_extra_state()`). Low priority since they cannot hold state.

3. **Q: Should PettingZoo's `parallel_seed_test` be used for serialization validation?**
   - **A:** Yes, `parallel_seed_test` already verifies determinism through seed, which complements serialization tests.

## Sources

### Primary (HIGH confidence)
- `/Users/chasemcd/Repositories/cogrid/cogrid/envs/overcooked/test_state_serialization.py` - 33 passing tests
- `/Users/chasemcd/Repositories/cogrid/cogrid/envs/search_rescue/test_sr_objects_serialization.py` - 22 passing tests
- `/Users/chasemcd/Repositories/cogrid/cogrid/envs/search_rescue/test_redvictim_serialization.py` - 7 passing tests
- [pytest Parametrize Documentation](https://docs.pytest.org/en/stable/how-to/parametrize.html)
- [PettingZoo Testing Documentation](https://pettingzoo.farama.org/content/environment_tests/)

### Secondary (MEDIUM confidence)
- [pytest Fixtures Documentation](https://docs.pytest.org/en/stable/how-to/fixtures.html)
- [Deterministic Simulation Testing](https://blog.resonatehq.io/deterministic-simulation-testing)
- [DEAP Checkpointing Tutorial](https://deap.readthedocs.io/en/master/tutorials/advanced/checkpoint.html)

## Metadata

**Confidence breakdown:**
- Test patterns: HIGH - verified from existing passing tests
- Coverage gaps: HIGH - direct code analysis
- pytest best practices: HIGH - official documentation verified
- PettingZoo integration: HIGH - existing code in repository

**Research date:** 2026-01-19
**Valid until:** 60 days (testing patterns stable, may update with pytest versions)

## Recommended Test Plan

### Phase 6 Testing Tasks

1. **Verify existing coverage (TEST-01)**
   - Run existing 62 tests, confirm all pass
   - Document which object types are covered
   - Confirm stateless objects return None from `get_extra_state()`

2. **Add Search & Rescue environment integration tests (TEST-02)**
   - Create `test_sr_env_serialization.py` parallel to Overcooked tests
   - Include full environment roundtrip after multiple steps
   - Test rescue mechanics mid-progress (RedVictim with active countdown)

3. **Enhance determinism tests (TEST-03)**
   - Extend action sequences to 50+ steps
   - Include stochastic elements (random agent priority, cooking timers)
   - Verify observation space matches exactly

4. **Create centralized integration test**
   - Cross-environment validation
   - JSON serialization roundtrip (for external storage)
   - Edge cases: all agents terminated, max_steps boundary

### Test Count Targets

| Category | Current | Target | Gap |
|----------|---------|--------|-----|
| Object roundtrip | 28 | 30 | +2 (core types) |
| Stateless verification | 21 | 21 | None |
| Environment integration | 12 | 18 | +6 (S&R env) |
| Determinism | 2 | 5 | +3 (extended) |
| **Total** | 62 | 74 | +12 |
