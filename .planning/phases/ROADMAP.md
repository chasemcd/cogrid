# v0.2.0 Determinism Audit — Roadmap

## Goal

Ensure all CoGrid environments are fully deterministic: same seed → identical trajectories, and restored states produce identical behavior.

## Phases

### Phase 1: Fix Step Dynamics Determinism ✓

**Status:** Complete (2026-01-20)

**Goal:** Remove randomness from step() — agent move priority must be deterministic.

**Scope:**
- [cogrid_env.py:493](../cogrid/cogrid_env.py#L493): Replace `np_random.shuffle(agents_to_move)` with deterministic ordering (sort by agent ID)

**Deliverable:** Step dynamics produce identical results for identical inputs.

---

### Phase 2: Fix Unseeded Layout Randomization ✓

**Status:** Complete (2026-01-20)

**Goal:** Ensure `Overcooked-RandomizedLayout-V0` uses the environment's seeded RNG.

**Scope:**
- [envs/__init__.py:128](../cogrid/envs/__init__.py#L128): Replace `random.choice()` with seeded selection
- Update `layout_fn` signature to receive `np_random` from environment
- Thread `np_random` through layout selection

**Deliverable:** RandomizedLayout variant produces identical layouts for identical seeds.

---

### Phase 3: Fix sr_utils Legacy RandomState ✓

**Status:** Complete (2026-01-20)

**Goal:** Ensure Search & Rescue grid generation uses the environment's RNG.

**Scope:**
- [envs/search_rescue/sr_utils.py:20-21](../cogrid/envs/search_rescue/sr_utils.py#L20): Remove fallback `RandomState`, require `np_random` argument
- Verify all callers pass `np_random`

**Deliverable:** S&R grid generation is deterministic via environment seed.

---

### Phase 4: Determinism Verification Tests

**Goal:** Add tests that verify determinism guarantees.

**Plans:** 1 plan

Plans:
- [ ] 04-01-PLAN.md — Extend trajectory test to 100 steps, add restored state continuation test

**Scope:**
- Test: Same seed → identical 100-step trajectory
- Test: Restored state → identical continuation
- Test: Agent collision resolution is deterministic
- Test: RandomizedLayout produces same layout for same seed

**Deliverable:** Test suite that catches determinism regressions.

---

## Success Criteria

1. `reset(seed=X)` followed by identical actions always produces identical states
2. `set_state(state)` followed by identical actions produces identical states
3. No uses of unseeded `random` module
4. No uses of `np.random` global state
5. All randomness flows through `self.np_random` (seeded at reset)

## Out of Scope

- Demo scripts (run_interactive.py, run_state_demo.py) — external to env
- Policy action sampling — user responsibility
