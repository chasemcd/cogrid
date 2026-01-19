# Requirements: CoGrid State Serialization

**Defined:** 2026-01-19
**Core Value:** Any environment state can be saved and restored with byte-perfect fidelity — the restored environment behaves identically to the original.

## v1 Requirements

### Framework

- [ ] **FRAM-01**: Audit all GridObj subclasses and identify those missing `get_extra_state()`/`set_extra_state()`
- [ ] **FRAM-02**: Implement recursive object serialization — objects holding other objects serialize their contents fully
- [ ] **FRAM-03**: Document the serialization pattern for future object authors

### Overcooked Objects

- [ ] **OVER-01**: Pot serializes cooking_tick, objects_in_pot (recursively), is_ready, is_cooking flags
- [ ] **OVER-02**: Counter serializes object_on_counter with full object state
- [ ] **OVER-03**: Plate serializes soup contents
- [ ] **OVER-04**: Soup serializes ingredients list
- [ ] **OVER-05**: OnionStack/TomatoStack serialize remaining count
- [ ] **OVER-06**: DeliveryZone and other Overcooked objects audited and serialized as needed

### Search & Rescue Objects

- [ ] **SRCH-01**: Victim objects (Green/Yellow/Red) serialize rescue state
- [ ] **SRCH-02**: Rubble serializes cleared state
- [ ] **SRCH-03**: Tools (MedKit, Pickaxe) serialize ownership/usage state
- [ ] **SRCH-04**: All other S&R objects audited and serialized as needed

### Agent Serialization

- [ ] **AGNT-01**: Agent inventory serializes held objects with full state (not just type)
- [ ] **AGNT-02**: Domain-specific agent state (OvercookedAgent) serializes completely

### Environment Serialization

- [ ] **ENVR-01**: Environment timestep (`t`) serializes and restores
- [ ] **ENVR-02**: RNG state (`np_random`) serializes and restores for determinism
- [ ] **ENVR-03**: Termination/truncation flags serialize and restore

### Testing

- [ ] **TEST-01**: Roundtrip tests for each object type (create -> get_state -> set_state -> verify)
- [ ] **TEST-02**: Full environment roundtrip test (run episode, checkpoint, restore, verify)
- [ ] **TEST-03**: Determinism test (restored env produces identical trajectories)

## v2 Requirements

(None identified — scope is well-defined)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Rendering state serialization | Not needed for replay — pygame windows regenerated |
| Feature space caching | Regenerated on demand from grid state |
| Cross-version state migration | v1.0 states only need to work with v1.0 code |
| Goal Seeking environment | Lower priority domain, can add later if needed |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FRAM-01 | Phase 1 - Framework Foundation | Pending |
| FRAM-02 | Phase 1 - Framework Foundation | Pending |
| FRAM-03 | Phase 1 - Framework Foundation | Pending |
| OVER-01 | Phase 2 - Overcooked Objects | Pending |
| OVER-02 | Phase 2 - Overcooked Objects | Pending |
| OVER-03 | Phase 2 - Overcooked Objects | Pending |
| OVER-04 | Phase 2 - Overcooked Objects | Pending |
| OVER-05 | Phase 2 - Overcooked Objects | Pending |
| OVER-06 | Phase 2 - Overcooked Objects | Pending |
| SRCH-01 | Phase 3 - Search & Rescue Objects | Pending |
| SRCH-02 | Phase 3 - Search & Rescue Objects | Pending |
| SRCH-03 | Phase 3 - Search & Rescue Objects | Pending |
| SRCH-04 | Phase 3 - Search & Rescue Objects | Pending |
| AGNT-01 | Phase 4 - Agent Serialization | Pending |
| AGNT-02 | Phase 4 - Agent Serialization | Pending |
| ENVR-01 | Phase 5 - Environment Serialization | Pending |
| ENVR-02 | Phase 5 - Environment Serialization | Pending |
| ENVR-03 | Phase 5 - Environment Serialization | Pending |
| TEST-01 | Phase 6 - Testing | Pending |
| TEST-02 | Phase 6 - Testing | Pending |
| TEST-03 | Phase 6 - Testing | Pending |

**Coverage:**
- v1 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0

---
*Requirements defined: 2026-01-19*
*Last updated: 2026-01-19 after roadmap creation*
