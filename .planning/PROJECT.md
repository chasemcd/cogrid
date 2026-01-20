# CoGrid State Serialization

## What This Is

Complete state serialization for CoGrid environments — ensuring `get_state()` and `set_state()` perfectly capture and restore any environment state for checkpointing, replay, and debugging. This is an enhancement to an existing multi-agent reinforcement learning framework built on PettingZoo.

## Core Value

Any environment state can be saved and restored with byte-perfect fidelity — the restored environment behaves identically to the original.

## Requirements

### Validated

- ✓ Multi-agent grid world environments (Overcooked, Search & Rescue, Goal Seeking) — existing
- ✓ PettingZoo ParallelEnv interface (step, reset, render, close) — existing
- ✓ Grid encoding/decoding with scoped object registries — existing
- ✓ Basic `get_state()`/`set_state()` on CoGridEnv — existing
- ✓ `Grid.get_state_dict()`/`set_state_dict()` for grid serialization — existing
- ✓ `Agent.get_state()`/`from_state()` for agent serialization — existing
- ✓ `GridObj.get_extra_state()`/`set_extra_state()` hooks — existing
- ✓ State version field ("1.0") for compatibility — existing
- ✓ All GridObj subclasses implement complete `get_extra_state()`/`set_extra_state()` — v0.1.0
- ✓ Pot object serializes: contents, cooking progress, ready status — v0.1.0
- ✓ Counter object serializes: held items with their full state — v0.1.0
- ✓ All Overcooked objects serialize completely — v0.1.0
- ✓ All Search & Rescue objects serialize completely — v0.1.0
- ✓ Agent inventory contents serialize with full object state — v0.1.0
- ✓ Environment state captures: timestep, RNG state, termination flags — v0.1.0
- ✓ Roundtrip test: get_state → set_state produces identical behavior — v0.1.0
- ✓ Clear pattern/documentation for future objects to follow — v0.1.0

### Active

(None — v0.1.0 complete, next milestone TBD)

### Out of Scope

- Rendering state (pygame window position, display buffers) — not needed for replay
- Feature space caching — regenerated on demand
- Cross-version state migration — v0.1.0 states only need to work with v0.1.0
- Goal Seeking environment serialization — lower priority domain, can add later

## Current State

**Shipped:** v0.1.0 State Serialization (2026-01-19)

**Codebase:**
- 10,173 lines of Python
- 76 serialization tests passing
- Supports Overcooked and Search & Rescue environments

**Tech stack:** Python 3.10+, NumPy, PettingZoo

## Constraints

- **Compatibility**: Must maintain existing `get_state()`/`set_state()` API signature
- **Tech stack**: Python 3.10+, NumPy for arrays, no new dependencies
- **Performance**: Serialization should not significantly slow down normal environment operation

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use existing `get_extra_state`/`set_extra_state` pattern | Already established, just needs complete implementation | ✓ Good |
| AST-based parsing for audit script | Reliable, doesn't require importing code | ✓ Good |
| GridAgent intentionally not serialized | Ephemeral, regenerated from Agent state each step | ✓ Good |
| Door uses state integer (no extra_state needed) | is_open/is_locked derived in __init__ | ✓ Good |
| Verification-only phases for most objects | Research found serialization already implemented | ✓ Good — saved implementation time |

---
*Last updated: 2026-01-19 after v0.1.0 milestone*
