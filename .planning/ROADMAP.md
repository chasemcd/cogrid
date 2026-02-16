# Roadmap: CoGrid v1.5 -- Docs & Packaging

## Overview

Transform CoGrid from a working but unpublished library into a proper open-source project. Four phases move sequentially through packaging modernization (pyproject.toml), CI validation (GitHub Actions), documentation migration (MkDocs Material replacing Sphinx), and automated deployment (PyPI publishing, GitHub Pages). Everything depends on pyproject.toml, so packaging comes first.

## Milestones

- v1.0 MVP -- Phases 1-4 (shipped)
- v1.1 Single Code Path -- Phases 5-8 (shipped)
- v1.2 Component API -- Phases 9-12 (shipped)
- v1.3 Array Feature System -- Phases 13-16 (shipped)
- v1.4 Developer Experience -- Phases 20-24 (shipped)
- **v1.5 Docs & Packaging** -- Phases 25-28 (in progress)

## Phases

**Phase Numbering:**
- Integer phases (25, 26, 27, 28): Planned milestone work
- Decimal phases (e.g., 26.1): Urgent insertions (marked with INSERTED)

- [x] **Phase 25: Packaging Consolidation** - Single pyproject.toml replaces all legacy packaging files *(completed 2026-02-16)*
- [x] **Phase 26: CI/CD Foundation** - GitHub Actions validates tests, linting, and multi-backend on every push *(completed 2026-02-16)*
- [x] **Phase 27: Documentation Migration** - MkDocs Material site with guides, tutorials, and API reference *(completed 2026-02-16)*
- [ ] **Phase 28: Automated Deployment** - Docs deploy to GitHub Pages, packages publish to PyPI, old infrastructure removed

## Phase Details

### Phase 25: Packaging Consolidation
**Goal**: Users can install CoGrid with modern tooling and optional extras
**Depends on**: Nothing (first phase of v1.5)
**Requirements**: PKG-01, PKG-02, PKG-03, PKG-04, PKG-05, PKG-06, PKG-07, PKG-08
**Success Criteria** (what must be TRUE):
  1. Running `pip install -e .` from a clean checkout installs CoGrid successfully using pyproject.toml
  2. Running `pip install -e ".[jax]"` installs JAX backend; `pip install -e ".[dev]"` installs test/lint tools; `pip install -e ".[docs]"` installs MkDocs stack
  3. `python -c "import cogrid; print(cogrid.__version__)"` prints the current version
  4. No setup.py, setup.cfg, MANIFEST.in, requirements.txt, or stale build artifacts remain in the repository
  5. `ruff check cogrid/` and `ruff format --check cogrid/` run cleanly using pyproject.toml configuration
**Plans:** 2 plans

Plans:
- [x] 25-01-PLAN.md -- pyproject.toml, __version__, README conversion, legacy file removal
- [x] 25-02-PLAN.md -- Ruff format + lint fix across all 66 Python files (zero violations)

### Phase 26: CI/CD Foundation
**Goal**: Every push and PR is automatically validated for correctness across Python versions and backends
**Depends on**: Phase 25
**Requirements**: CI-01, CI-02, CI-03, CI-04, CI-08
**Success Criteria** (what must be TRUE):
  1. Pushing a commit to main or opening a PR triggers a GitHub Actions workflow that runs the full test suite
  2. Tests pass on Python 3.10, 3.11, and 3.12 with the numpy backend
  3. A separate CI job runs JAX-specific tests on Python 3.12 (Linux/macOS)
  4. Ruff lint and format checks run in CI and fail the build on violations
  5. CI runs complete within a reasonable time using pip dependency caching
**Plans:** 1 plan

Plans:
- [x] 26-01-PLAN.md -- CI workflow with lint, numpy test matrix, and JAX test jobs

### Phase 27: Documentation Migration
**Goal**: Users can learn CoGrid's concepts, follow tutorials, and browse API reference on a MkDocs Material site
**Depends on**: Phase 25 (mkdocstrings needs the package importable)
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04, DOC-05, DOC-06, DOC-07, DOC-08, DOC-09
**Success Criteria** (what must be TRUE):
  1. Running `mkdocs serve` renders a complete documentation site with navigation, search, and Material theme
  2. A Getting Started page walks through installation, creating an environment, and running a step -- using the current array-based API
  3. A Custom Environment tutorial explains how to create a new environment using GridObject subclasses, ArrayReward, and layout files
  4. API reference pages auto-generated via mkdocstrings display public classes and functions for all core modules
  5. Architecture/concepts page explains the xp backend system, EnvState, component API, and step pipeline
**Plans:** 3 plans

Plans:
- [x] 27-01-PLAN.md -- MkDocs Material infrastructure, mkdocs.yml, gen_ref_pages.py, API reference auto-generation
- [x] 27-02-PLAN.md -- Getting Started, Architecture/Concepts, Custom Environment, JAX Backend content pages
- [x] 27-03-PLAN.md -- Environment gallery pages (Overcooked, SearchRescue, GoalSeeking), Contributing guide

### Phase 28: Automated Deployment
**Goal**: Documentation and packages publish automatically -- no manual steps after merging to main or creating a release
**Depends on**: Phase 26, Phase 27
**Requirements**: CI-05, CI-06, CI-07, CI-09, DOC-10
**Success Criteria** (what must be TRUE):
  1. Pushing to main automatically builds and deploys documentation to GitHub Pages at https://chasemcd.github.io/cogrid
  2. Creating a GitHub Release triggers automated PyPI publishing via OIDC trusted publishers (no API tokens)
  3. CI includes `mkdocs build --strict` that catches broken links and missing references before merge
  4. Coverage reporting runs in CI and a coverage badge is available for the README
  5. Old Sphinx configuration files and RST source files are removed from the repository
**Plans:** 2 plans

Plans:
- [ ] 28-01-PLAN.md -- CI docs validation (mkdocs build --strict) and coverage reporting (pytest-cov + py-cov-action)
- [ ] 28-02-PLAN.md -- Docs deploy workflow, PyPI publish workflow, Sphinx cleanup, README update

## Progress

**Execution Order:**
Phases execute in numeric order: 25 -> 26 -> 27 -> 28

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 25. Packaging Consolidation | 2/2 | ✓ Complete | 2026-02-16 |
| 26. CI/CD Foundation | 1/1 | ✓ Complete | 2026-02-16 |
| 27. Documentation Migration | 3/3 | ✓ Complete | 2026-02-16 |
| 28. Automated Deployment | 0/2 | Not started | - |

---
*Roadmap created: 2026-02-16*
*Milestone: v1.5 Docs & Packaging*
