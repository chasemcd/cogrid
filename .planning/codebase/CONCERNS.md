# Codebase Concerns

**Analysis Date:** 2026-01-19

## Tech Debt

**Duplicated Build Directory:**
- Issue: The `/Users/chasemcd/Repositories/cogrid/build/lib/cogrid/` directory contains a complete copy of the source code, creating maintenance burden and potential version drift
- Files: `build/lib/cogrid/` (entire directory mirrors `cogrid/`)
- Impact: Changes to source must be synced to build directory; TODOs exist in both locations; confuses code analysis tools
- Fix approach: Remove build directory from version control, regenerate on build only

**Dead/Commented Code in feature_space.py:**
- Issue: Large block of commented-out code (lines 49-84) represents abandoned refactoring
- Files: `cogrid/feature_space/feature_space.py`
- Impact: Confuses readers about intended design; accumulates cruft
- Fix approach: Remove commented code or complete the refactoring

**Hardcoded Magic Numbers in Overcooked Features:**
- Issue: Max number of pots hardcoded to 2 in `OvercookedCollectedFeatures`, max_shape hardcoded to (11, 7) for layout
- Files: `cogrid/envs/overcooked/overcooked_features.py:56`, `cogrid/envs/overcooked/overcooked_features.py:86`
- Impact: Feature space will break silently or produce incorrect features for layouts with different configurations
- Fix approach: Derive these values from environment config or grid properties

**Unused Color Field in Object Encoding:**
- Issue: Object encoding includes a color field that is always 0 with a TODO to remove it
- Files: `cogrid/core/grid_object.py:193`
- Impact: Wastes encoding space; misleads about object representation format
- Fix approach: Remove the unused color field from encode() method and update all consumers

**Inventory Size Assumption:**
- Issue: Multiple places assume inventory_capacity=1 with TODO comments acknowledging the limitation
- Files: `cogrid/envs/overcooked/overcooked_grid_objects.py:175`, `cogrid/core/grid_object.py:366-368`, `cogrid/feature_space/feature_space.py:68`
- Impact: Multi-item inventories will cause silent bugs or assertion failures
- Fix approach: Generalize inventory handling or enforce single-item constraint explicitly

**PyGame Rendering Coupled to Environment:**
- Issue: Rendering logic mixed into core environment class instead of being modular
- Files: `cogrid/cogrid_env.py:81`, `cogrid/cogrid_env.py:1058`
- Impact: Cannot use environment without pygame import attempt; harder to test headlessly
- Fix approach: Extract rendering to separate module, inject visualizer if needed

## Known Bugs

**DistToOtherPlayers Feature Bug:**
- Symptoms: `other_agent_nums` increment is outside the for loop, causing incorrect distance encoding for >2 agents
- Files: `cogrid/envs/overcooked/overcooked_features.py:503`
- Trigger: Use environment with >2 agents
- Workaround: Only use with exactly 2 agents

**BFS Not Implemented for Distance Features:**
- Symptoms: Documentation claims BFS pathfinding but actually uses Manhattan/Euclidean distance
- Files: `cogrid/envs/overcooked/overcooked_features.py:269-271`, `cogrid/envs/overcooked/overcooked_features.py:341`, `cogrid/envs/overcooked/overcooked_features.py:392`
- Trigger: Layouts with obstacles between agent and target
- Workaround: None - features are incorrect for blocked paths

**OtherAgentVisibility Feature Incomplete:**
- Symptoms: Feature class exists but generate() raises NotImplementedError
- Files: `cogrid/feature_space/features.py:333`
- Trigger: Attempting to use "other_agent_visibility" feature
- Workaround: Do not use this feature

## Security Considerations

**No Input Validation on Config:**
- Risk: Config dictionary accepted without validation; malformed configs may cause crashes or unexpected behavior
- Files: `cogrid/cogrid_env.py:61-68`
- Current mitigation: Some assertions exist but incomplete
- Recommendations: Add schema validation for config; fail fast with clear error messages

## Performance Bottlenecks

**Grid Rendering Recomputes All Tiles:**
- Problem: Every render() call iterates all grid cells even when most haven't changed
- Files: `cogrid/core/grid.py:370-401`
- Cause: No dirty tile tracking
- Improvement path: Implement dirty tile tracking as noted in TODO at line 370

**fill_coords Uses Nested Python Loops:**
- Problem: Pixel-by-pixel iteration in Python for every rendered tile
- Files: `cogrid/visualization/rendering.py:30-42`
- Cause: Pure Python implementation without NumPy vectorization
- Improvement path: Vectorize using NumPy boolean masks

**Large Environment File (cogrid_env.py):**
- Problem: 1315 lines in single file makes navigation and maintenance difficult
- Files: `cogrid/cogrid_env.py`
- Cause: All environment logic in one class
- Improvement path: Extract rendering, serialization, and agent management to separate modules

## Fragile Areas

**Object Registry Global State:**
- Files: `cogrid/core/grid_object.py:31` (OBJECT_REGISTRY)
- Why fragile: Global mutable dictionary; registration order matters; scope conflicts possible
- Safe modification: Always check existing registrations before adding; test with fresh interpreter state
- Test coverage: No tests for registry collision handling

**Grid Agent Position Synchronization:**
- Files: `cogrid/cogrid_env.py:416-423` (update_grid_agents)
- Why fragile: `env_agents` and `grid.grid_agents` must stay synchronized; any modification path that skips update_grid_agents() will cause inconsistency
- Safe modification: Always call update_grid_agents() after any agent state change
- Test coverage: Some movement tests exist, but edge cases may not be covered

**State Serialization Version Handling:**
- Files: `cogrid/cogrid_env.py:1232-1233`
- Why fragile: Only version "1.0" supported; no migration path for state format changes
- Safe modification: Increment version and add migration logic when changing state format
- Test coverage: Basic roundtrip tests exist in `cogrid/envs/overcooked/test_state_serialization.py`

## Scaling Limits

**Feature Space Construction:**
- Current capacity: Feature shapes calculated at init time for fixed agent counts
- Limit: Cannot dynamically add/remove agents without reconstructing feature space
- Scaling path: Make feature generators dynamic or support agent set changes

**Pot Capacity:**
- Current capacity: Hardcoded to max 2 pots in feature encoding
- Limit: Layouts with >2 pots will have incomplete features
- Scaling path: Make pot count configurable or derive from grid analysis

## Dependencies at Risk

**PettingZoo Required:**
- Risk: Tests fail without pettingzoo installed; not in all dev environments
- Impact: CI/CD may pass but local testing fails
- Migration plan: Add to dev dependencies or document clearly

**Optional cv2 Handling:**
- Risk: Silent None assignment when cv2 unavailable, then crashes on actual use
- Files: `cogrid/visualization/rendering.py:7-10`, `cogrid/feature_space/features.py:11-14`
- Impact: Delayed failures when users attempt image operations
- Migration plan: Fail at import time with helpful message, or fully abstract cv2 dependency

## Missing Critical Features

**Action Masking Not Implemented:**
- Problem: `get_action_mask()` raises NotImplementedError
- Files: `cogrid/cogrid_env.py:1119-1120`
- Blocks: Safe RL with invalid action masking

**Reward Module Abstract Only:**
- Problem: `calculate_reward()` is abstract in base class, requires custom implementation
- Files: `cogrid/core/reward.py:50`
- Blocks: Using environment without defining reward functions

## Test Coverage Gaps

**No Tests Run Successfully:**
- What's not tested: All tests fail to import due to missing pettingzoo dependency
- Files: `cogrid/test_gridworld_env.py`, `cogrid/test_overcooked_env.py`, `cogrid/envs/overcooked/test_state_serialization.py`
- Risk: Regressions go undetected; cannot verify fixes
- Priority: High - need CI environment with all dependencies

**Commented-Out Tests:**
- What's not tested: Large sections of test_gridworld_env.py are commented out (lines 177-275, 404-489)
- Files: `cogrid/test_gridworld_env.py`
- Risk: View/observation tests not running; agent conflict tests not running
- Priority: Medium - tests existed but were disabled

**Feature Generators Lack Unit Tests:**
- What's not tested: Individual feature generators in `cogrid/feature_space/features.py`
- Files: `cogrid/feature_space/features.py`
- Risk: Feature encoding bugs affect downstream training without detection
- Priority: High - features are core to RL observation space

**Overcooked-Specific Objects:**
- What's not tested: Most overcooked grid objects have minimal test coverage
- Files: `cogrid/envs/overcooked/overcooked_grid_objects.py`
- Risk: Pot cooking logic, soup creation, delivery zone behavior may have edge case bugs
- Priority: Medium - some integration tests exist in test_overcooked_env.py

---

*Concerns audit: 2026-01-19*
