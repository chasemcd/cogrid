# CoGrid -- Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-16)

**Core value:** Minimal code paths, maximal clarity. One functional simulation core that works identically whether xp is numpy or jax.numpy.
**Current focus:** Phase 27 -- Documentation Migration

## Current Position

Phase: 27 of 28 (Documentation Migration)
Plan: 1 of 3 in current phase
Status: Executing
Last activity: 2026-02-16 -- Completed 27-01 (MkDocs infrastructure)

Progress: [██████░░░░] 57%

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 7min
- Total execution time: 0.4 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 25-packaging-consolidation | 2 | 24min | 12min |
| 26-ci-cd-foundation | 1 | 2min | 2min |
| 27-documentation-migration | 1 | 2min | 2min |

**Recent Trend:**
- Last 5 plans: 25-01 (4min), 25-02 (20min), 26-01 (2min), 27-01 (2min)
- Trend: improving

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v1.0-v1.4: Built functional array-based engine with component API and code clarity pass
- v1.5 start: MkDocs over Sphinx, pyproject.toml over setup.py, GitHub Actions CI
- 25-01: Removed license classifier due to PEP 639 conflict with SPDX expression in setuptools >=68
- 25-01: pyproject.toml is now sole packaging config with dynamic version from cogrid.__version__
- 25-02: Used TYPE_CHECKING + future annotations for forward references instead of string annotations
- 25-02: All docstrings follow Google convention (concise, no redundant Args/Returns sections)
- 26-01: CI uses parallel jobs (no needs dependencies) for fastest feedback
- 26-01: astral-sh/ruff-action@v3 for lint (pre-built binary, PR annotations)
- 26-01: JAX tests on Python 3.12 only (JAX >= 0.5.x requires Python >= 3.11)
- 27-01: mkdocstrings paths: [.] since cogrid/ is at repo root; literate-nav only for API reference
- 27-01: docs/reference/ gitignored; gen_ref_pages.py regenerates at build time

### Pending Todos

None yet.

### Blockers/Concerns

- NumPy 2.0 compatibility unknown -- current pin is numpy==1.26.4, need to test against numpy 2.x in Phase 26
- PyPI Trusted Publisher requires manual configuration on pypi.org before Phase 28 publish workflow will work
- ~~Mixed docstring styles (Sphinx, Google, NumPy) -- need normalization before mkdocstrings renders correctly in Phase 27~~ RESOLVED in 25-02: all docstrings now Google convention

## Session Continuity

Last session: 2026-02-16
Stopped at: Completed 27-01-PLAN.md
Resume file: None
