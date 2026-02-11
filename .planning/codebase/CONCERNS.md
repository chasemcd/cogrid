# Codebase Concerns

**Analysis Date:** 2026-02-10

## Tech Debt

**Rigid Inventory System:**
- Issue: Codebase assumes inventory size of 1 throughout, with hard assertions enforcing this limit. Multiple TODOs indicate this is a deliberate limitation that needs to be addressed.
- Files: `cogrid/envs/overcooked/overcooked_grid_objects.py:175`, `cogrid/core/grid_object.py:324-325`, `cogrid/core/grid_object.py:379-381`
- Impact: Cannot extend system to support agents carrying multiple items. Blocks multiplicity features.
- Fix approach: Remove inventory size assertions, refactor `pick_up_from()` and inventory management to handle variable-sized inventories. Update rendering logic at `cogrid/core/grid_object.py:374-396` to support more than 3 inventory items.

**State Encoding Incomplete:**
- Issue: Agent state only encodes inventory items but not agent roles or other attributes. Comment explicitly notes this gap.
- Files: `cogrid/core/grid_object.py:329-335`
- Impact: Agent state serialization loses role information, preventing proper state reconstruction. Affects reproducibility and state saving.
- Fix approach: Extend state integer to encode role_idx alongside inventory. May require expanding state encoding scheme to support both attributes simultaneously.

**Counter State Not Utilized:**
- Issue: Counter objects accept state parameter in `__init__` but don't use it. Comment indicates objects placed on counters should be encoded in state but aren't implemented.
- Files: `cogrid/core/grid_object.py:484-488`
- Impact: Cannot serialize/deserialize objects placed on counters. State is lost on environment reload.
- Fix approach: Implement state-to-object mapping in Counter initialization. Map state int to object via registry, then construct object and place it on counter.

**Rendering Tightly Coupled to Environment:**
- Issue: PyGame rendering logic mixed into CoGridEnv class alongside core simulation. TODO explicitly flags this.
- Files: `cogrid/cogrid_env.py:81`, `cogrid/cogrid_env.py:1058`
- Impact: Difficult to test environment without pygame. Violates separation of concerns. Makes headless execution brittle.
- Fix approach: Extract all pygame code (lines 1053-1115+) to separate `Renderer` class. Pass renderer as dependency to CoGridEnv.

**Inefficient Grid Rendering:**
- Issue: Grid renderer recomputes all tiles every step with nested loops. TODO mentions "dirty tiles" optimization not implemented.
- Files: `cogrid/core/grid.py:370-403`
- Impact: Performance bottleneck in high-resolution rendering. Unnecessary CPU usage on static cells.
- Fix approach: Implement dirty tile tracking. Only re-render tiles that changed position/state. Cache unchanged tiles.

## Performance Bottlenecks

**Feature Distance Calculations Use Euclidian Distance Comment But Manhattan Distance Implementation:**
- Problem: Comment in `ClosestObj.generate()` says "TODO: Use BFS here" but code only does (dy, dx) euclidian distance. This is misleading about what actually happens.
- Files: `cogrid/envs/overcooked/overcooked_features.py:269`
- Cause: BFS not implemented; fallback to simple coordinate deltas
- Improvement path: Implement actual BFS pathfinding to find true walkable distances respecting walls and obstacles.

**Pot Reachability Hardcoded to True:**
- Problem: Pot reachability feature always returns 1, claiming it should use BFS to calculate actual reachability.
- Files: `cogrid/envs/overcooked/overcooked_features.py:341`, `cogrid/envs/overcooked/overcooked_features.py:392`
- Cause: BFS search not implemented
- Improvement path: Calculate actual reachability using BFS from agent to pot. Return 0 if unreachable due to walls.

**Nested Loop Grid Encoding:**
- Problem: Grid encoding uses nested loops over height/width. No early termination or optimization.
- Files: `cogrid/core/grid.py:418-437`
- Cause: Straightforward implementation without optimization
- Improvement path: Vectorize with numpy operations or use numpy's flat iteration where possible. Profile before optimizing.

## Fragile Areas

**Global Object Registry:**
- Files: `cogrid/core/grid_object.py:31`
- Why fragile: Module-level mutable state shared globally across all scopes. No thread safety. Registration order matters for character uniqueness. Scope system is complex.
- Safe modification: Add thread locks if multi-threaded use planned. Document scope inheritance rules. Test character collisions across scopes.
- Test coverage: No explicit tests for registry collision handling. No tests for concurrent scope registration.

**Feature Space Shape Assumptions:**
- Files: `cogrid/envs/overcooked/overcooked_features.py:50-100`, `cogrid/feature_space/features.py:130-175`
- Why fragile: Feature shapes are hardcoded. Adding objects or changing pot count breaks shape contracts. `max_num_pots=2` is magical constant.
- Safe modification: Make feature shapes dynamic based on environment config. Pass max_pots through constructor.
- Test coverage: No tests validating feature shapes under different environment configurations.

**Inventory Length Assertions:**
- Files: `cogrid/core/grid_object.py:324-325`, `cogrid/core/grid_object.py:379-381`, `cogrid/envs/overcooked/overcooked_grid_objects.py:175`
- Why fragile: Hard assertions crash on inventory size > 1. No graceful degradation. Multiple places assume inventory[0] exists.
- Safe modification: Change assertions to validation. Return error codes or adjust rendering instead of crashing.
- Test coverage: No tests for inventory > 1 size. No boundary tests.

## Known Issues

**Bare Exception Handler:**
- Symptoms: cv2 import failure silently sets cv2=None, making some features unavailable without clear error message
- Files: `cogrid/visualization/rendering.py:7-10`, `cogrid/feature_space/features.py:11-14`
- Trigger: System without opencv-python installed. Features needing cv2 will fail at runtime with confusing error.
- Workaround: Install opencv-python explicitly. Check cv2 is not None before using image resize features.

**Bare Print Statement in Library Code:**
- Symptoms: Reward registry override prints warning to stdout instead of using logging. Mixed signal in warnings vs logging.
- Files: `cogrid/core/reward.py:58-61`
- Trigger: Registering duplicate reward ID
- Workaround: None - warning is printed regardless
- Fix approach: Use logging module instead of print() for warnings in library code.

## Test Coverage Gaps

**State Serialization Not Tested in Core:**
- What's not tested: GridObj state encoding/decoding round-trips. Agent state preservation.
- Files: `cogrid/core/grid_object.py:190-195`, `cogrid/core/grid_object.py:202-221`
- Risk: State changes break silently. Serialization format changes not caught. Cart before horse for state restoration.
- Priority: High - blocks reproducibility features

**Environment Reset Determinism:**
- What's not tested: Whether reset() with same seed produces identical state. Layout variations not seeded.
- Files: `cogrid/cogrid_env.py:201-250` (reset implementation)
- Risk: Non-deterministic behavior in training. Agents see different environments despite seeding.
- Priority: High - critical for RL training stability

**Feature Shape Contracts:**
- What's not tested: Features produce correct shapes under all environment configurations.
- Files: `cogrid/envs/overcooked/overcooked_features.py`, `cogrid/feature_space/features.py`
- Risk: Shape mismatches crash agents mid-training. Silent failures if shapes shrink.
- Priority: Medium - surfaces quickly but hard to debug

**Rendering Pipeline:**
- What's not tested: Rendering without pygame. Image encoding/decoding. POV rendering accuracy.
- Files: `cogrid/cogrid_env.py:1018-1115`, `cogrid/visualization/rendering.py`
- Risk: Rendering bugs affect visual debugging. PIL/image issues not caught early.
- Priority: Medium - affects usability not core function

## Scaling Limits

**Fixed Pot Count in Features:**
- Current capacity: max_num_pots = 2 hardcoded
- Limit: Adding 3rd pot breaks feature shape
- Scaling path: Make `max_num_pots` configurable, passed from environment to feature class. Update test fixtures to test with 3+ pots.

**Grid Encoding Memory:**
- Current capacity: Grid encodes as numpy array (H, W, 3) - 3 channels for each cell
- Limit: Large maps (50x50+) consume significant memory for feature computation. Repeated encoding expensive.
- Scaling path: Cache encoded grids per timestep. Only re-encode changed cells. Consider sparse representation for mostly-empty maps.

**Agent View Computation:**
- Current capacity: Default view_size = 7 (49-cell grids per agent)
- Limit: View computation for each agent runs separately. 10+ agents becomes expensive.
- Scaling path: Compute all views once per step if possible. Use shared visibility masks.

## Security Considerations

**No Input Validation on Config Dictionary:**
- Risk: CoGridEnv accepts arbitrary config dict. Missing keys crash at access time instead of initialization.
- Files: `cogrid/cogrid_env.py:84`, `cogrid/cogrid_env.py:94`, `cogrid/cogrid_env.py:97`
- Current mitigation: Code assumes keys exist (name, max_steps, num_agents, etc.)
- Recommendations: Add config schema validation at `__init__`. Provide helpful errors for missing keys. Document required config keys.

**Object Registry Scope Confusion:**
- Risk: Scope system allows registering same ID in different scopes. Easy to accidentally collide when mixing environments.
- Files: `cogrid/core/grid_object.py:40-51`, `cogrid/core/grid_object.py:60-82`
- Current mitigation: Global scope checked first, but scope inheritance not explicit
- Recommendations: Add warnings when registering objects with same char in different scopes. Document scope precedence.

## Missing Critical Features

**No State Serialization/Deserialization:**
- Problem: Cannot save/load environment state. Rollback not possible. Replay not fully supported.
- Blocks: Checkpointing during training, state inspection tools, deterministic replay for debugging
- Impact: Training is one-way. Can't pause and resume. Can't inspect intermediate states.

**No Vectorized Environment Wrapper:**
- Problem: All environment interaction is sequential. No batch environment support.
- Blocks: Using with standard RL libraries that expect vectorized environments (stable-baselines3, etc.)
- Impact: Inefficient training. Must roll own parallelization.

**No Configuration Validation:**
- Problem: Invalid configs cause cryptic errors deep in initialization.
- Blocks: User-friendly error messages. Early feedback on config mistakes.
- Impact: Hard to debug setup issues. Long feedback loop for users.

---

*Concerns audit: 2026-02-10*
