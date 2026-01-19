# Phase 5: Environment Serialization - Research

**Researched:** 2026-01-19
**Domain:** Environment-level state serialization (timestep, RNG, termination flags)
**Confidence:** HIGH

## Summary

Phase 5 research confirms that environment-level serialization is **already fully implemented** and working. The `CoGridEnv` class (in `cogrid/cogrid_env.py`) has complete `get_state()` and `set_state()` methods that handle all environment-level state including timestep, RNG state, and termination/truncation flags.

Existing tests in `cogrid/envs/overcooked/test_state_serialization.py` already verify that:
- Timestep (`t`) is preserved after roundtrip (ENVR-01)
- RNG state produces identical random sequences after restoration (ENVR-02)
- Deterministic replay works correctly

The only gap is explicit testing for termination/truncation flag preservation (ENVR-03). The `terminated` flag IS serialized (in `Agent.terminated`), but there's no test that explicitly verifies a terminated environment is still terminated after restoration.

**Primary recommendation:** This phase is verification-only. Add explicit tests for termination/truncation flag preservation (ENVR-03) to ensure complete coverage, but no new serialization code is needed.

## Standard Stack

### Core Implementation Locations

| Component | Location | Serialization Status |
|-----------|----------|---------------------|
| `CoGridEnv.get_state()` | `cogrid/cogrid_env.py:1141-1206` | COMPLETE |
| `CoGridEnv.set_state()` | `cogrid/cogrid_env.py:1208-1316` | COMPLETE |
| Timestep serialization | `state["timestep"]` | COMPLETE |
| RNG state serialization | `state["rng_state"]` | COMPLETE |
| Termination flag | `Agent.terminated` (per-agent) | COMPLETE |
| Truncation computation | `self.t >= self.max_steps` (dynamic) | N/A - computed |

### Key Insight: Termination vs Truncation

There are two distinct "done" conditions in CoGrid:

1. **`terminated`**: Per-agent flag stored in `Agent.terminated`. This IS serialized because `Agent.get_state()` includes it and `Agent.from_state()` restores it.

2. **`truncated`**: Computed dynamically by `get_terminateds_truncateds()` as `self.t >= self.max_steps`. This is NOT directly stored because it can be recomputed from `timestep` and `config["max_steps"]`.

This is correct design - storing only the minimal state needed and computing derived values.

## Architecture Patterns

### Existing Environment Serialization Pattern

**Location:** `cogrid/cogrid_env.py`, lines 1141-1316

The `CoGridEnv.get_state()` method serializes:
- `version`: State format version ("1.0")
- `config`: Full environment configuration
- `scope`: Object registry scope
- `timestep`: Current timestep (`self.t`)
- `cumulative_score`: Running score
- `current_layout_id`: Layout identifier
- `rng_state`: NumPy random state for determinism
- `grid`: Full grid state with object metadata
- `agents`: All agent states (including `terminated` flag)
- `spawn_points`: Remaining spawn points
- `prev_actions`: Previous actions taken
- `per_agent_reward`: Rewards per agent
- `per_component_reward`: Rewards per component

```python
# From cogrid/cogrid_env.py - already implemented
def get_state(self) -> dict:
    # Get RNG state for deterministic restoration
    rng_state = None
    if self._np_random is not None:
        rng_state = {"bit_generator": self._np_random.bit_generator.state}

    # Serialize grid (objects only, agents handled separately)
    grid_state = self.grid.get_state_dict(scope=self.scope)

    # Serialize all agents
    agents_state = {
        agent_id: agent.get_state(scope=self.scope)
        for agent_id, agent in self.env_agents.items()
    }

    state = {
        "version": "1.0",
        "config": self.config,
        "scope": self.scope,
        "timestep": self.t,  # <-- ENVR-01
        "cumulative_score": self.cumulative_score,
        "current_layout_id": self.current_layout_id,
        "rng_state": rng_state,  # <-- ENVR-02
        "grid": grid_state,
        "agents": agents_state,  # <-- Contains terminated flag (ENVR-03)
        "spawn_points": self.spawn_points.copy(),
        "prev_actions": ...,
        "per_agent_reward": ...,
        "per_component_reward": ...,
    }
    return state
```

### Termination Flag Flow

The `terminated` flag flows through the system as follows:

1. **Storage:** `Agent.terminated` is a boolean per-agent attribute
2. **Serialization:** `Agent.get_state()` includes `"terminated": self.terminated`
3. **Restoration:** `Agent.from_state()` sets `agent.terminated = state_dict["terminated"]`
4. **Environment update:** `CoGridEnv.set_state()` updates `self.agents` list based on restored `terminated` flags

```python
# From cogrid/cogrid_env.py:1310-1315 - already implemented
# Update agents list (currently active, non-terminated agents)
self.agents = [
    agent_id
    for agent_id in self.possible_agents
    if not self.env_agents[agent_id].terminated
]
```

## Don't Hand-Roll

| Problem | Don't Build | Already Exists |
|---------|-------------|----------------|
| Environment state serialization | Custom serialization | `CoGridEnv.get_state()` / `set_state()` |
| Timestep preservation | Manual timestep tracking | `state["timestep"]` |
| RNG state preservation | Custom RNG handling | `bit_generator.state` pattern |
| Termination flag preservation | Environment-level flags | `Agent.terminated` per-agent |

## Common Pitfalls

### Pitfall 1: Looking for Environment-Level Termination Flags

**What goes wrong:** Searching for a `CoGridEnv.terminated` attribute that doesn't exist.
**Why it happens:** Expecting termination to be environment-level rather than per-agent.
**How to avoid:** Understand that termination is per-agent in `Agent.terminated`, and the active agents list (`self.agents`) is derived from this.
**Warning signs:** Looking for termination/truncation in `get_state()` output directly.

### Pitfall 2: Expecting Truncation to be Stored

**What goes wrong:** Trying to find or add truncation flag serialization.
**Why it happens:** Not understanding that truncation is computed from `timestep >= max_steps`.
**How to avoid:** Recognize that `timestep` + `config["max_steps"]` is sufficient to reconstruct truncation state.
**Warning signs:** Adding `truncated` field to state dictionary.

### Pitfall 3: Testing Termination Without Actually Terminating an Agent

**What goes wrong:** Tests check flag exists but don't verify behavior with actually-terminated agents.
**Why it happens:** Most test scenarios use non-terminating environments.
**How to avoid:** Create explicit test that sets `agent.terminated = True`, saves/restores, verifies behavior.
**Warning signs:** Tests only check structure, not actual terminated-environment behavior.

## Code Examples

### Existing Timestep Serialization (Verified Working)

```python
# From cogrid/envs/overcooked/test_state_serialization.py
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
    self.assertEqual(self.env.t, 10)  # <-- ENVR-01 verified
```

### Existing RNG State Serialization (Verified Working)

```python
# From cogrid/envs/overcooked/test_state_serialization.py
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
        self.assertAlmostEqual(r1, r2, places=10)  # <-- ENVR-02 verified
```

### Existing Determinism Test (Verified Working)

```python
# From cogrid/envs/overcooked/test_state_serialization.py
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
        for agent_id in rew1.keys():
            self.assertAlmostEqual(rew1[agent_id], rew2[agent_id])
        for agent_id in agents1.keys():
            np.testing.assert_array_equal(agents1[agent_id].pos, agents2[agent_id].pos)
```

### Missing Test: Termination Flag Preservation (ENVR-03)

```python
# Recommended new test to add
def test_terminated_agent_preserved(self):
    """Test that terminated agents remain terminated after state restoration."""
    self.env.reset(seed=42)

    # Terminate an agent manually
    self.env.env_agents[0].terminated = True

    # Save state
    state = self.env.get_state()

    # Verify terminated flag is in serialized state
    self.assertTrue(state["agents"][0]["terminated"])

    # Restore to new environment
    env2 = cogrid_env.CoGridEnv(config=self.config)
    env2.set_state(state)

    # Verify agent is still terminated
    self.assertTrue(env2.env_agents[0].terminated)

    # Verify agent is not in active agents list
    self.assertNotIn(0, env2.agents)
```

### Missing Test: Truncation Behavior After Restore

```python
# Recommended new test to add
def test_truncation_after_restore_near_max_steps(self):
    """Test truncation is correctly computed after restoring near max_steps."""
    # Use environment with low max_steps for testing
    config = {**self.config, "max_steps": 10}
    env1 = cogrid_env.CoGridEnv(config=config)
    env1.reset(seed=42)

    # Step to just before max_steps
    for _ in range(9):
        env1.step({0: Actions.Noop, 1: Actions.Noop})

    # Save state at t=9 (one step from truncation)
    state = env1.get_state()
    self.assertEqual(state["timestep"], 9)

    # Restore and verify one more step causes truncation
    env2 = cogrid_env.CoGridEnv(config=config)
    env2.set_state(state)

    self.assertEqual(env2.t, 9)

    # One more step should trigger truncation
    _, _, terminateds, truncateds, _ = env2.step({0: Actions.Noop, 1: Actions.Noop})
    self.assertTrue(truncateds[0])
    self.assertTrue(truncateds[1])
```

## State of the Art

| Previous Understanding | Actual State | Impact |
|----------------------|--------------|--------|
| Termination flags need environment-level storage | Per-agent `Agent.terminated` already stored | No new storage needed |
| Truncation needs serialization | Computed from `timestep >= max_steps` | No storage needed |
| Environment serialization incomplete | `get_state()`/`set_state()` fully implemented | Phase is verification-only |

## Requirements Analysis

### ENVR-01: Environment timestep (`t`) serializes and restores

**Status:** ALREADY IMPLEMENTED AND TESTED

**Evidence:**
- `CoGridEnv.get_state()` includes `"timestep": self.t`
- `CoGridEnv.set_state()` restores `self.t = state["timestep"]`
- Test `test_roundtrip_after_steps` verifies `self.env.t == env2.t == 10`

**Recommendation:** No additional work needed.

### ENVR-02: RNG state (`np_random`) serializes and restores for determinism

**Status:** ALREADY IMPLEMENTED AND TESTED

**Evidence:**
- `CoGridEnv.get_state()` serializes `self._np_random.bit_generator.state`
- `CoGridEnv.set_state()` restores RNG via `self._np_random.bit_generator.state = ...`
- Test `test_rng_state_preservation` verifies identical random sequences after restore
- Test `test_deterministic_after_restore` verifies identical outcomes after restore

**Recommendation:** No additional work needed.

### ENVR-03: Termination/truncation flags serialize and restore

**Status:** IMPLEMENTED but needs explicit test coverage

**Evidence for termination:**
- `Agent.terminated` is serialized in `Agent.get_state()`
- `Agent.from_state()` restores `agent.terminated`
- `CoGridEnv.set_state()` updates `self.agents` list based on restored terminated flags

**Evidence for truncation:**
- Truncation is computed dynamically as `self.t >= self.max_steps`
- Restoring `timestep` is sufficient for truncation to work correctly

**Gap:** No test explicitly verifies a terminated environment is still terminated after restoration.

**Recommendation:** Add test `test_terminated_agent_preserved` to verify ENVR-03.

## Open Questions

None - all questions resolved through code analysis.

1. **Q: Is there an environment-level termination flag?**
   - **A:** No. Termination is per-agent in `Agent.terminated`. The environment computes `terminateds` dict from these per-agent flags.

2. **Q: Is truncation stored?**
   - **A:** No. Truncation is computed from `self.t >= self.max_steps`. Storing `timestep` is sufficient.

3. **Q: Does the `agents` list get restored correctly?**
   - **A:** Yes. `set_state()` rebuilds `self.agents` from non-terminated agents after restoration.

## Testing Approach

Since serialization is already implemented, testing should focus on:

1. **Explicit termination verification** - Test that `agent.terminated = True` persists through roundtrip
2. **Active agents list** - Verify `self.agents` excludes terminated agents after restore
3. **Truncation behavior** - Verify environment truncates at correct timestep after restore
4. **Edge cases** - All agents terminated, restore at max_steps, etc.

## Sources

### Primary (HIGH confidence)
- `/Users/chasemcd/Repositories/cogrid/cogrid/cogrid_env.py` - Environment with get_state/set_state (lines 1141-1316)
- `/Users/chasemcd/Repositories/cogrid/cogrid/core/agent.py` - Agent with terminated flag (line 31, 123, 162)
- `/Users/chasemcd/Repositories/cogrid/cogrid/envs/overcooked/test_state_serialization.py` - Existing tests (31 passing)

### Secondary (MEDIUM confidence)
- Test execution confirming all 31 tests pass

## Metadata

**Confidence breakdown:**
- Timestep serialization: HIGH - verified in source code and tested
- RNG state serialization: HIGH - verified in source code and tested
- Termination flag serialization: HIGH - verified in source code, needs explicit test
- Truncation computation: HIGH - verified in source code

**Research date:** 2026-01-19
**Valid until:** Indefinitely (core architecture unlikely to change)

## Conclusion

**Phase 5 is verification-only.** The serialization infrastructure for environment-level state is already complete:

1. `CoGridEnv.get_state()` and `CoGridEnv.set_state()` handle all environment state
2. Timestep (`self.t`) is serialized and restored correctly
3. RNG state is serialized and produces deterministic behavior after restore
4. Termination flags are serialized per-agent in `Agent.terminated`
5. Truncation is computed dynamically from `timestep >= max_steps`

**Recommended actions:**
1. Add test `test_terminated_agent_preserved` for ENVR-03 explicit verification
2. Add test `test_truncation_after_restore_near_max_steps` for truncation edge case
3. No implementation work required
