# Project Milestones: CoGrid

## v0.2.0 Determinism Audit (Shipped: 2026-01-20)

**Delivered:** Full environment determinism — same seed produces identical trajectories, restored states continue identically.

**Phases completed:** 1-4 (4 plans total)

**Key accomplishments:**

- Replaced agent move shuffle with deterministic sort for collision resolution
- Threaded np_random through layout functions for seeded layout selection
- Removed legacy RandomState fallback in sr_utils.py
- Added 4-test determinism verification suite (100-step trajectories, restored state continuation)

**Stats:**

- 6 files modified
- 10,546 lines of Python
- 4 phases, 4 plans
- 11 determinism tests passing
- Completed in 1 day

**Git range:** `fix(01-01)` → `docs(v0.2.0)`

**What's next:** TBD

---

## v0.1.0 State Serialization (Shipped: 2026-01-19)

**Delivered:** Complete state serialization for CoGrid environments with byte-perfect fidelity — any environment state can be saved and restored identically.

**Phases completed:** 1-6 (9 plans total)

**Key accomplishments:**

- Created audit script for GridObj serialization status (AST-based, 27 tests)
- Implemented RedVictim serialization and verified Door works via state integer
- Documented serialization pattern in GridObj docstrings
- Verified all Overcooked objects serialize correctly (Pot, Counter, stateless objects)
- Verified all Search & Rescue objects serialize correctly (6 stateless + RedVictim)
- Verified agent serialization with inventory contents preserved
- Verified environment-level state (timestep, RNG, termination flags)
- Created comprehensive integration tests with 50+ step determinism validation

**Stats:**

- 49 files created/modified
- 10,173 lines of Python
- 6 phases, 9 plans
- 76 tests passing
- Completed in 1 day

**Git range:** `feat(01-01)` → `docs(06)`

**What's next:** TBD — state serialization feature complete

---
