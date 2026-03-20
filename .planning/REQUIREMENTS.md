# Requirements: LocalView API Simplification

**Defined:** 2026-03-18
**Core Value:** A new user can create a custom LocalView subclass with minimal boilerplate — just plain Python methods, no framework ceremony.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### API Simplification

- [x] **API-01**: Subclass authors override a single instance method `extra_channels(self, state, H, W) -> list[ndarray]` instead of the classmethod `build_extra_channel_fn`
- [x] **API-02**: Subclass authors declare `n_extra_channels` as a class attribute instead of overriding the `extra_n_channels()` classmethod
- [x] **API-03**: Subclass authors only need `from cogrid.feature_space import LocalView` — no decorator, registry, or `xp` imports required
- [ ] **API-04**: OvercookedLocalView is updated to use the new simplified API
- [ ] **API-05**: Deprecated classmethods (`extra_n_channels`, `build_extra_channel_fn`) are removed from the public API

### Helpers

- [x] **HELP-01**: Base class provides a `_scatter_to_grid(grid, positions, values)` helper that hides JAX/NumPy branching from subclass authors

### Safety

- [x] **SAFE-01**: Golden output tests capture exact tensor baselines for LocalView and OvercookedLocalView BEFORE any code changes
- [x] **SAFE-02**: Runtime assertion validates that `n_extra_channels` matches the length of `extra_channels()` output
- [ ] **SAFE-03**: Clear, actionable error messages for common subclassing mistakes (missing `n_extra_channels`, wrong return type, channel count mismatch)
- [x] **SAFE-04**: JAX backend tracing verification — instance method approach works correctly under JAX JIT

### Compatibility

- [ ] **COMPAT-01**: Observation arrays produced by the refactored code are identical to pre-refactor (same shape, dtype, channel order)
- [x] **COMPAT-02**: All existing tests in `test_features.py` pass without modification (except import path changes)

## v2 Requirements

### API Enhancements

- **API-06**: Auto-registration via `__init_subclass__` — hide `@register_feature_type` decorator entirely
- **API-07**: Channel introspection / debugging utilities for inspecting channel layout
- **API-08**: Declarative channel specification as an alternative to method override

### Testing

- **TEST-01**: `focal_only` composition test to validate attribute effect on feature composition

## Out of Scope

| Feature | Reason |
|---------|--------|
| Changes to Feature base class | Scoped to LocalView layer only — Feature/registry stays as-is |
| Changes to `generic_local_view_feature()` internals | Only the public subclass API changes, not the core computation |
| New observation channels or features | Purely an API cleanup, not a feature addition |
| Changes to channel layout or semantics | Same outputs, simpler authoring |
| Backward compatibility shims | Breaking changes OK — OvercookedLocalView is the only subclass |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| API-01 | Phase 2 | Complete |
| API-02 | Phase 2 | Complete |
| API-03 | Phase 2 | Complete |
| API-04 | Phase 3 | Pending |
| API-05 | Phase 3 | Pending |
| HELP-01 | Phase 2 | Complete |
| SAFE-01 | Phase 1 | Complete |
| SAFE-02 | Phase 2 | Complete |
| SAFE-03 | Phase 3 | Pending |
| SAFE-04 | Phase 2 | Complete |
| COMPAT-01 | Phase 3 | Pending |
| COMPAT-02 | Phase 1 | Complete |

**Coverage:**
- v1 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0

---
*Requirements defined: 2026-03-18*
*Last updated: 2026-03-18 after roadmap creation*
