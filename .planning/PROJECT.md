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

### Active

- [ ] All GridObj subclasses implement complete `get_extra_state()`/`set_extra_state()`
- [ ] Pot object serializes: contents, cooking progress, ready status
- [ ] Counter object serializes: held items with their full state
- [ ] All Overcooked objects serialize completely (Onion, Tomato, Plate, Soup, DeliveryZone, etc.)
- [ ] All Search & Rescue objects serialize completely (Victims, Rubble, Tools)
- [ ] Agent inventory contents serialize with full object state
- [ ] Environment state captures: timestep, RNG state, episode info
- [ ] Roundtrip test: get_state → set_state produces identical behavior
- [ ] Clear pattern/documentation for future objects to follow

### Out of Scope

- Rendering state (pygame window position, display buffers) — not needed for replay
- Feature space caching — regenerated on demand
- Cross-version state migration — v1.0 states only need to work with v1.0

## Context

**Existing serialization infrastructure:**
- `CoGridEnv.get_state()` and `set_state()` exist but may not capture all object internal state
- `GridObj` has `get_extra_state()` and `set_extra_state()` hooks but not all subclasses implement them
- `Grid.encode()` captures object type and basic state but complex objects (Pot with contents) need extra handling
- Tests exist in `cogrid/envs/overcooked/test_state_serialization.py` but coverage may be incomplete

**Key objects needing audit:**
- Overcooked: Pot (cooking state), Counter (held items), Plate, Soup, OnionStack, TomatoStack, DeliveryZone
- Search & Rescue: Victims (rescue state), Rubble, Tools (MedKit, Pickaxe)
- Global: Door (open/closed), Key, any object with internal state beyond position

## Constraints

- **Compatibility**: Must maintain existing `get_state()`/`set_state()` API signature
- **Tech stack**: Python 3.10+, NumPy for arrays, no new dependencies
- **Performance**: Serialization should not significantly slow down normal environment operation

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use existing `get_extra_state`/`set_extra_state` pattern | Already established, just needs complete implementation | — Pending |

---
*Last updated: 2026-01-19 after initialization*
