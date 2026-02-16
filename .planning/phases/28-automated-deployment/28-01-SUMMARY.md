---
phase: 28-automated-deployment
plan: 01
subsystem: infra
tags: [ci, coverage, pytest-cov, mkdocs, github-actions]

requires:
  - phase: 26-ci-cd-foundation
    provides: "CI workflow with lint, test, test-jax jobs"
  - phase: 27-documentation-migration
    provides: "mkdocs site with Material theme and mkdocstrings"
provides:
  - "docs validation CI job (mkdocs build --strict)"
  - "coverage reporting CI job (pytest-cov + py-cov-action)"
  - "coverage config in pyproject.toml with relative_files"
  - "mkdocs validation config for link/nav checking"
affects: [28-02]

tech-stack:
  added: [pytest-cov, py-cov-action/python-coverage-comment-action]
  patterns: ["coverage config in pyproject.toml [tool.coverage.*]", "mkdocs validation section for strict mode"]

key-files:
  created: []
  modified:
    - ".github/workflows/ci.yml"
    - "pyproject.toml"
    - "mkdocs.yml"
    - "cogrid/envs/overcooked/config.py"
    - "docs/index.md"
    - "docs/getting-started.md"

key-decisions:
  - "py-cov-action/python-coverage-comment-action@v3 for PR coverage comments"
  - "coverage.run.relative_files=true for CI path mapping compatibility"
  - "mkdocs validation warns on broken links/anchors rather than ignoring them"

patterns-established:
  - "All CI jobs run in parallel (no needs dependencies) for fastest feedback"
  - "Docs job uses .[docs] extra, coverage job uses .[dev] extra"

duration: 5min
completed: 2026-02-16
---

# Phase 28 Plan 01: Docs Validation & Coverage CI Summary

**CI extended with mkdocs strict-mode validation and pytest-cov coverage reporting using py-cov-action for PR comments**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-16T23:05:57Z
- **Completed:** 2026-02-16T23:11:30Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- pytest-cov added as dev dependency with coverage config (source, omit, relative_files) in pyproject.toml
- mkdocs validation section added to catch broken links, anchors, and nav issues in strict mode
- Two new CI jobs (docs, coverage) running in parallel with existing lint/test/test-jax jobs
- Coverage job uses py-cov-action to post coverage summaries as PR comments

## Task Commits

Each task was committed atomically:

1. **Task 1: Add coverage config and pytest-cov dependency** - `3805f56` (feat)
2. **Task 2: Add docs and coverage jobs to CI workflow** - `7b306c6` (feat)

## Files Created/Modified

- `.github/workflows/ci.yml` - Added docs and coverage jobs (5 total parallel jobs)
- `pyproject.toml` - Added pytest-cov dep, [tool.coverage.run], [tool.coverage.report]
- `mkdocs.yml` - Added validation: section for nav/link checking
- `cogrid/envs/overcooked/config.py` - Reformatted module docstring to avoid griffe warnings
- `docs/index.md` - Fixed API Reference link to explicit index.md path
- `docs/getting-started.md` - Fixed API Reference link to explicit index.md path

## Decisions Made

- Used py-cov-action/python-coverage-comment-action@v3 for automatic PR coverage comments (badge-ready)
- Set relative_files=true in coverage config for correct path mapping in CI
- mkdocs validation uses warn level for all checks, which becomes errors under --strict

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed mkdocs build --strict warnings**
- **Found during:** Task 1 (mkdocs build --strict verification)
- **Issue:** 7 warnings: 2 unrecognized `reference/` links in index.md and getting-started.md, 5 griffe docstring parsing warnings in overcooked/config.py
- **Fix:** Changed `reference/` links to `reference/cogrid/index.md` (explicit path to generated page); reformatted config.py module docstring from `Functions:` (parsed as Google-style section) to plain bullet list
- **Files modified:** docs/index.md, docs/getting-started.md, cogrid/envs/overcooked/config.py
- **Verification:** mkdocs build --strict exits 0 with no warnings
- **Committed in:** 3805f56 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Fix was necessary for mkdocs build --strict to pass. No scope creep.

## Issues Encountered

None beyond the deviation documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- CI now has 5 parallel jobs: lint, test, test-jax, docs, coverage
- Ready for 28-02 (PyPI publish workflow) which will add the 6th CI job
- PyPI Trusted Publisher still requires manual configuration on pypi.org (noted in STATE.md blockers)

## Self-Check: PASSED

- All 7 files verified present on disk
- Both task commits (3805f56, 7b306c6) verified in git log

---
*Phase: 28-automated-deployment*
*Completed: 2026-02-16*
