# Roadmap: LocalView API Simplification

## Overview

This refactoring transforms cogrid's LocalView subclass API from a classmethod-heavy, closure-returning pattern into a beginner-friendly instance-method interface. The work follows a strict safety-first sequence: capture golden output baselines before any code changes, refactor the LocalView base class with the new instance-method bridge, then update OvercookedLocalView and clean up the deprecated API. Every phase is independently verifiable and the output tensor remains bit-identical throughout.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Safety Baseline** - Capture golden output tests and confirm existing test suite passes before any code changes (completed 2026-03-19)
- [ ] **Phase 2: Base Class Refactoring** - Transform LocalView to use instance methods, class attributes, and the closure-bridge pattern
- [ ] **Phase 3: Subclass Update and Cleanup** - Update OvercookedLocalView, remove deprecated classmethods, add error messages, verify full parity

## Phase Details

### Phase 1: Safety Baseline
**Goal**: A provable baseline exists so any code change that alters observation output is immediately detected
**Depends on**: Nothing (first phase)
**Requirements**: SAFE-01, COMPAT-02
**Success Criteria** (what must be TRUE):
  1. Golden output test fixtures exist that capture exact tensor values (shape, dtype, values) for both LocalView and OvercookedLocalView on a deterministic env state
  2. Running the full existing test suite (`test_features.py`, `test_overcooked_features.py`) produces zero failures, confirming the pre-refactor baseline is clean
**Plans:** 1/1 plans complete

Plans:
- [ ] 01-01-PLAN.md — Create golden output tests for LocalView and OvercookedLocalView, verify existing test suite baseline

### Phase 2: Base Class Refactoring
**Goal**: LocalView exposes a beginner-friendly subclass API -- instance methods and class attributes -- while preserving the Feature/registry contract under the hood
**Depends on**: Phase 1
**Requirements**: API-01, API-02, API-03, HELP-01, SAFE-02, SAFE-04
**Success Criteria** (what must be TRUE):
  1. A new LocalView subclass can be authored by overriding `extra_channels(self, state)` and setting `n_extra_channels` as a class attribute -- no classmethods, closures, or decorator knowledge required
  2. `from cogrid.feature_space import LocalView` is the only import a subclass author needs
  3. `_scatter_to_grid` helper is available on the base class and abstracts JAX/NumPy backend branching
  4. A runtime assertion fires with a clear message when `n_extra_channels` does not match the length of `extra_channels()` output
  5. The refactored LocalView (with no extra channels) produces bit-identical output to the golden baseline, and the code works correctly under JAX JIT tracing
**Plans:** 2 plans

Plans:
- [ ] 02-01-PLAN.md — Create test scaffold for new API + add import re-exports
- [ ] 02-02-PLAN.md — Refactor LocalView base class with new instance-method API

### Phase 3: Subclass Update and Cleanup
**Goal**: OvercookedLocalView uses the new simplified API, deprecated classmethods are removed, and the full refactoring is verified end-to-end
**Depends on**: Phase 2
**Requirements**: API-04, API-05, SAFE-03, COMPAT-01
**Success Criteria** (what must be TRUE):
  1. OvercookedLocalView is implemented using only `__init__`, `extra_channels(self, state)`, and `n_extra_channels = 8` -- no classmethods remain in the subclass
  2. Deprecated classmethods (`extra_n_channels`, `build_extra_channel_fn`) are removed from the LocalView public API
  3. Common subclassing mistakes (missing `n_extra_channels`, wrong return type, channel count mismatch) produce clear, actionable error messages
  4. Golden output tests pass with exact equality -- observation arrays are bit-identical to the pre-refactor baseline for both LocalView and OvercookedLocalView
  5. The full existing test suite passes without modification (except import path changes)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Safety Baseline | 1/1 | Complete    | 2026-03-19 |
| 2. Base Class Refactoring | 0/2 | Not started | - |
| 3. Subclass Update and Cleanup | 0/0 | Not started | - |
