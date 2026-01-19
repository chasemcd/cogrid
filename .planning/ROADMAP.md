# Roadmap

**Project:** CoGrid State Serialization
**Created:** 2026-01-19
**Phases:** 6

## Overview

This roadmap delivers complete state serialization for CoGrid environments through six phases. The structure follows the natural requirement categories: establishing framework patterns first, then implementing object serialization by domain (Overcooked, Search & Rescue), then agent and environment-level serialization, and finally comprehensive testing. Each phase builds on the previous, with testing at the end to validate the complete system.

## Phases

### Phase 1: Framework Foundation

**Goal:** Establish the patterns and infrastructure for complete object serialization across all domains.
**Depends on:** Nothing (first phase)
**Requirements:** FRAM-01, FRAM-02, FRAM-03

**Success Criteria:**
1. Developer can run audit script/command and see complete list of GridObj subclasses with their serialization status (implemented/missing)
2. Objects containing other objects (e.g., Counter holding a Plate) correctly serialize their nested contents when `get_extra_state()` is called
3. Documentation exists showing a developer how to implement `get_extra_state()`/`set_extra_state()` for a new object type

**Plans:** 3 plans

Plans:
- [x] 01-01-PLAN.md — Create audit script for GridObj serialization status
- [x] 01-02-PLAN.md — Implement RedVictim serialization + verify Door
- [x] 01-03-PLAN.md — Document serialization pattern in docstrings

---

### Phase 2: Overcooked Objects

**Goal:** Verify all Overcooked domain objects serialize their complete internal state (most already implemented or stateless).
**Depends on:** Phase 1 (recursive serialization pattern must exist)
**Requirements:** OVER-01, OVER-02, OVER-03, OVER-04, OVER-05, OVER-06

**Success Criteria:**
1. Pot object roundtrips correctly: cooking_tick value, objects_in_pot list, is_ready flag, and is_cooking flag all match after get_state/set_state
2. Counter with an object on it roundtrips correctly: the object_on_counter is restored with its full state
3. Plate holding soup roundtrips correctly: soup contents are preserved
4. OnionStack and TomatoStack roundtrip correctly: remaining count matches after restore
5. All Overcooked objects can be individually roundtripped without data loss

**Plans:** 1 plan

Plans:
- [ ] 02-01-PLAN.md — Verify all Overcooked objects via comprehensive roundtrip tests

---

### Phase 3: Search & Rescue Objects

**Goal:** All Search & Rescue domain objects serialize their complete internal state.
**Depends on:** Phase 1 (recursive serialization pattern must exist)
**Requirements:** SRCH-01, SRCH-02, SRCH-03, SRCH-04

**Success Criteria:**
1. Victim objects (Green, Yellow, Red) roundtrip correctly: rescue state preserved after restore
2. Rubble objects roundtrip correctly: cleared/uncleared state preserved
3. Tool objects (MedKit, Pickaxe) roundtrip correctly: any ownership or usage state preserved
4. All Search & Rescue objects can be individually roundtripped without data loss

**Plans:** (created by /gsd:plan-phase)

---

### Phase 4: Agent Serialization

**Goal:** Agents serialize their complete state including inventory contents with full object state.
**Depends on:** Phase 2, Phase 3 (object serialization must work for inventory contents)
**Requirements:** AGNT-01, AGNT-02

**Success Criteria:**
1. Agent holding an object (e.g., Onion, Plate with soup) roundtrips correctly: inventory item restored with complete state
2. OvercookedAgent-specific state roundtrips correctly: any domain-specific fields preserved
3. Agent serialization works regardless of what object type is being held

**Plans:** (created by /gsd:plan-phase)

---

### Phase 5: Environment Serialization

**Goal:** Environment-level state (timestep, RNG, termination flags) serializes completely for deterministic replay.
**Depends on:** Phase 4 (all lower-level serialization must work)
**Requirements:** ENVR-01, ENVR-02, ENVR-03

**Success Criteria:**
1. Environment timestep (`t`) is identical after get_state/set_state roundtrip
2. RNG state restoration produces identical random sequences: same actions after restore yield same outcomes as before checkpoint
3. Termination and truncation flags are preserved: a terminated environment restored from checkpoint is still terminated

**Plans:** (created by /gsd:plan-phase)

---

### Phase 6: Testing

**Goal:** Comprehensive test coverage validates complete state serialization system.
**Depends on:** Phase 5 (all serialization must be implemented)
**Requirements:** TEST-01, TEST-02, TEST-03

**Success Criteria:**
1. Test suite exists with roundtrip tests for each object type that could have internal state
2. Full environment test exists: run partial episode, checkpoint, restore, continue, verify no behavior difference
3. Determinism test exists: restored environment produces identical trajectories when given identical action sequences
4. All tests pass in CI/pytest run

**Plans:** (created by /gsd:plan-phase)

---

## Progress

| Phase | Status | Completed |
|-------|--------|-----------|
| 1 - Framework Foundation | Complete | 2026-01-19 |
| 2 - Overcooked Objects | Planned | - |
| 3 - Search & Rescue Objects | Not started | - |
| 4 - Agent Serialization | Not started | - |
| 5 - Environment Serialization | Not started | - |
| 6 - Testing | Not started | - |

---

*Roadmap for milestone: v1.0*
