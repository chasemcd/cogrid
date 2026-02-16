# Phase 25: Packaging Consolidation - Research

**Researched:** 2026-02-16
**Domain:** Python packaging (pyproject.toml, PEP 621) and linting (ruff)
**Confidence:** HIGH

## Summary

Phase 25 replaces CoGrid's legacy packaging setup (setup.py, setup.cfg, MANIFEST.in, requirements.txt) with a single PEP 621-compliant pyproject.toml, adds ruff linting/formatting configuration, and fixes all existing violations. The codebase currently has 66 Python files under `cogrid/`, with 440 total ruff violations against the target rule set (E+F+I+UP+D with Google convention at 100-char line length). Of these, ~97 are auto-fixable. Formatting violations affect 45 of 66 files.

A critical finding is that `cogrid/__init__.py` does not currently define `__version__`. It must be added as a string literal for setuptools' `attr:` dynamic version to parse it via `ast.literal_eval`.

The user decision to allow `numpy>=1.26,<3.0` works well since PettingZoo 1.24.3 and Gymnasium 1.0.0 are already the installed versions. The JAX extra needs careful version bounds: current JAX 0.9.x requires Python >=3.11, while the project targets Python >=3.10. The `[jax]` extra must use bounds that allow pip's resolver to find compatible versions on both Python 3.10 (jax<=0.4.38) and Python 3.11+ (jax 0.5+).

**Primary recommendation:** Use setuptools as the build backend (it is already configured and the project is pure-Python with no special needs). Write the full pyproject.toml first, then run `ruff format` and `ruff check --fix` for auto-fixable issues, then manually fix remaining violations file by file.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Dependency version policy
- Compatible ranges for all core dependencies (e.g. `numpy>=1.26,<3.0`)
- Allow numpy 2.x now -- optimistic approach; CI in Phase 26 will catch breakage
- Minimum floors only for dev dependencies (e.g. `pytest>=7.0`) -- flexible for contributors
- Python requires `>=3.10` -- matches Phase 26 CI matrix (3.10, 3.11, 3.12)

#### Optional extras design
- Four extras groups: `[jax]`, `[dev]`, `[docs]`, `[all]`
- `[jax]` -- jax + jaxlib with compatible ranges (e.g. `jax>=0.4.20,<1.0`)
- `[dev]` -- test + lint + docs (includes everything a contributor needs)
- `[docs]` -- mkdocs stack only (standalone for doc-only work)
- `[all]` -- union of jax + dev + docs
- No separate `[test]` group -- `[dev]` covers it

#### Version management
- Canonical version in `cogrid/__init__.py` as `__version__ = "0.0.16"`
- pyproject.toml reads version dynamically from `__init__.py`
- Keep current version `0.0.16` -- bump to 1.5.0 when full milestone ships
- Package name on PyPI: `cogrid`

#### Linting & formatting rules
- Line length: 100 characters
- Ruff rule set: E (errors) + F (pyflakes) + I (isort) + UP (pyupgrade) + D (docstrings)
- Import sorting handled by ruff (I rules) -- no separate isort
- Enforce Google-style docstrings via ruff D rules -- normalize now ahead of Phase 27
- Fix all existing violations as part of Phase 25 -- clean slate from day one
- Ruff handles both linting (`ruff check`) and formatting (`ruff format`)

### Claude's Discretion
- Build backend choice (setuptools, hatchling, flit, etc.)
- pyproject.toml structure and section ordering
- Exact ruff rule codes within the selected categories
- Migration strategy for removing legacy files
- Ruff per-file-ignores if needed for specific edge cases during violation fixing

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.

</user_constraints>

## Standard Stack

### Core

| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| setuptools | >=68 | Build backend | Already configured; stable, well-understood; handles `attr:` dynamic versioning from `__init__.py` |
| ruff | >=0.9.0 | Lint + format | Single tool replaces flake8+isort+black+pyupgrade+pydocstyle; fast; fully pyproject.toml-configured |

### Supporting (runtime dependencies)

| Library | Constraint | Purpose | Notes |
|---------|-----------|---------|-------|
| numpy | >=1.26,<3.0 | Array backend | 1.26.4 currently pinned; allows numpy 2.x per user decision |
| gymnasium | >=0.29 | RL environment API | Currently installed: 1.0.0; no upper bound needed (PettingZoo constrains) |
| pettingzoo | >=1.24 | Multi-agent API | Currently installed: 1.24.3; latest is 1.25.0 |

### Supporting (optional extras)

| Group | Libraries | Constraints | Notes |
|-------|-----------|-------------|-------|
| `[jax]` | jax, jaxlib | `jax>=0.4.20,<1.0` | JAX 0.9.x needs Python >=3.11; on Python 3.10, pip resolves to jax 0.4.x which still works |
| `[docs]` | mkdocs, mkdocs-material, mkdocstrings[python] | `mkdocs>=1.5`, `mkdocs-material>=9.0`, `mkdocstrings[python]>=0.24` | Standard MkDocs stack for Python API docs |
| `[dev]` | pytest, ruff, + all docs deps | `pytest>=7.0`, `ruff>=0.9.0` | Single group for contributors |
| `[all]` | union of jax+dev+docs | (references other extras) | Convenience combo |

### Build Backend Recommendation (Claude's Discretion)

**Recommendation: Keep setuptools.**

Rationale:
- The project already uses setuptools (pyproject.toml has `build-backend = "setuptools.build_meta"`).
- CoGrid is a pure-Python package with no extension modules.
- setuptools handles `attr:` dynamic version from `__init__.py` natively.
- Hatchling offers advantages (smarter file inclusion, better defaults) but switching backends adds migration risk with no practical benefit for this project.
- setuptools >=68 supports all PEP 621 metadata.

If this were a fresh project, hatchling would be a fine choice. For an existing setuptools project, migrating backends adds complexity with negligible gain.

## Architecture Patterns

### Target pyproject.toml Structure

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cogrid"
dynamic = ["version"]
description = "..."
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
authors = [{name = "Chase McDonald", email = "chasemcd@cmu.edu"}]
classifiers = [...]
dependencies = [
    "numpy>=1.26,<3.0",
    "gymnasium>=0.29",
    "pettingzoo>=1.24",
]

[project.optional-dependencies]
jax = ["jax>=0.4.20,<1.0"]
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.24",
]
dev = [
    "pytest>=7.0",
    "ruff>=0.9.0",
    "cogrid[docs]",
]
all = ["cogrid[jax]", "cogrid[dev]"]

[project.urls]
Homepage = "https://github.com/chasemcd/cogrid"

[tool.setuptools.dynamic]
version = {attr = "cogrid.__version__"}

[tool.setuptools.packages.find]
include = ["cogrid*"]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "D"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.format]
docstring-code-format = true
```

### Pattern: Dynamic Version via `attr:`

**What:** `__version__` is the single source of truth in `cogrid/__init__.py`; pyproject.toml reads it at build time via `[tool.setuptools.dynamic] version = {attr = "cogrid.__version__"}`.

**Critical requirement:** The `__init__.py` must define `__version__` as a **simple string literal** (e.g., `__version__ = "0.0.16"`). Setuptools uses `ast.literal_eval` to parse it. Any dynamic construction (f-strings, function calls, imports) will fail.

**Current state:** `cogrid/__init__.py` has NO `__version__` -- it must be added.

**Example `cogrid/__init__.py` after edit:**
```python
__version__ = "0.0.16"


def make(environment_id: str, **kwargs):
    """Create a pre-registered CoGrid environment by name.

    Triggers registration of all built-in environments on first call.
    """
    import cogrid.envs  # noqa: F401 -- trigger registration
    from cogrid.envs import registry

    return registry.make(environment_id, **kwargs)
```

### Pattern: Self-Referencing Optional Extras

**What:** The `[dev]` extra can reference `cogrid[docs]` to include docs dependencies. Similarly, `[all]` references `cogrid[jax]` and `cogrid[dev]`.

**Why:** Avoids duplicating dependency lists across groups.

**Example:**
```toml
[project.optional-dependencies]
jax = ["jax>=0.4.20,<1.0"]
docs = ["mkdocs>=1.5", "mkdocs-material>=9.0", "mkdocstrings[python]>=0.24"]
dev = ["pytest>=7.0", "ruff>=0.9.0", "cogrid[docs]"]
all = ["cogrid[jax]", "cogrid[dev]"]
```

### Anti-Patterns to Avoid

- **Pinning exact versions in library dependencies:** Use compatible ranges (`>=X,<Y`), never `==X.Y.Z`. Exact pins cause dependency conflicts for downstream users.
- **Leaving both old and new packaging files:** After migration, remove setup.py/setup.cfg/MANIFEST.in/requirements.txt from git. Leaving them creates confusion about which is authoritative.
- **Importing package modules in `__init__.py` before `__version__`:** The `attr:` directive uses AST parsing. If `__init__.py` has imports that fail (circular, missing deps), it can break the build. Put `__version__` first, before any imports.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Import sorting | Manual import ordering | ruff I rules | ruff handles isort-compatible sorting natively |
| Code formatting | Manual style enforcement | `ruff format` | Consistent, deterministic, fast |
| Docstring convention enforcement | Manual review | ruff D rules + `convention = "google"` | Automatic, catches 300+ violations |
| Version syncing between files | Custom scripts | setuptools `attr:` dynamic version | Standard, maintained, AST-based |

**Key insight:** ruff replaces 5+ separate tools (flake8, isort, black, pyupgrade, pydocstyle) with a single binary. There is no reason to install or configure any of those tools separately.

## Common Pitfalls

### Pitfall 1: `__version__` Not Parseable by `ast.literal_eval`

**What goes wrong:** Build fails with `ValueError: malformed node or string` when pip tries to determine the package version.
**Why it happens:** `attr:` uses `ast.literal_eval`, which cannot handle f-strings, function calls, or imported values.
**How to avoid:** Use a bare string literal: `__version__ = "0.0.16"`. Nothing else on that line. Place it at the very top of `__init__.py`, before any imports.
**Warning signs:** `pip install -e .` fails with a version-related error.

### Pitfall 2: Ruff D Rules Without Setting Convention

**What goes wrong:** Incompatible rules (D203 vs D211, D212 vs D213) both fire, producing contradictory lint errors.
**Why it happens:** Without `convention = "google"`, ruff enables all D rules including mutually exclusive ones.
**How to avoid:** Always set `[tool.ruff.lint.pydocstyle] convention = "google"` when using D rules.
**Warning signs:** Ruff emits "incompatible rules" warnings; lint errors suggest adding and removing blank lines simultaneously.

### Pitfall 3: Forgetting to Remove Legacy Files

**What goes wrong:** `pip install -e .` silently uses setup.py instead of pyproject.toml, or users get confused about which file to edit.
**Why it happens:** pip and setuptools have fallback behavior to legacy files.
**How to avoid:** Delete setup.py, setup.cfg, MANIFEST.in, requirements.txt, and docs/requirements.txt in the same commit (or immediately after) creating the full pyproject.toml. Verify by running `pip install -e .` with only pyproject.toml present.
**Warning signs:** `pip install -e .` output mentions `setup.py` instead of pyproject.toml.

### Pitfall 4: Ruff Auto-Fix Breaks Code Semantics

**What goes wrong:** `ruff check --fix` removes an import that was used for side-effect registration (common with `noqa: F401` comments).
**Why it happens:** Auto-fix for F401 (unused imports) removes imports not referenced in the file, but some imports trigger module-level side effects (e.g., `@register_object_type` decorators).
**How to avoid:** Run `ruff check --fix` and then immediately run the test suite. Existing `# noqa: F401` comments in the codebase already protect these imports -- verify they are preserved.
**Warning signs:** Tests fail after auto-fix with "not registered" errors.

### Pitfall 5: D100/D101/D102/D103 Docstring Requirements on Tests

**What goes wrong:** Hundreds of "undocumented-public-*" errors on test files, which conventionally don't need docstrings.
**Why it happens:** Ruff D rules apply to all Python files by default.
**How to avoid:** Use `per-file-ignores` to exempt test files from docstring requirements:
```toml
[tool.ruff.lint.per-file-ignores]
"cogrid/tests/*" = ["D100", "D101", "D102", "D103", "D104"]
"cogrid/test_*.py" = ["D100", "D101", "D102", "D103", "D104"]
```
**Warning signs:** Lint output dominated by test file docstring errors.

### Pitfall 6: Real Bugs Exposed by F821/F841

**What goes wrong:** Ruff finds actual undefined names and unused variables, not just style issues.
**Why it happens:** These are genuine code bugs that were never caught.
**Current state:** 4 F821 (undefined name) errors exist:
- `cogrid/core/layout_parser.py:69` -- `EnvState` used as type annotation but not imported (needs string annotation or import)
- `cogrid/run_interactive.py:17,24,65` -- `ort` (onnxruntime) and `special` (scipy.special) referenced but never imported
**How to avoid:** Fix these manually (add missing imports or remove dead code). Do not blindly auto-fix.

## Code Examples

### Complete pyproject.toml (recommended)

Source: Verified against [setuptools pyproject.toml docs](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) and [PEP 621](https://peps.python.org/pep-0621/)

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cogrid"
dynamic = ["version"]
description = "A library for creating multi-agent grid-world environments for reinforcement learning."
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
authors = [
    {name = "Chase McDonald", email = "chasemcd@cmu.edu"},
]
keywords = ["reinforcement-learning", "multi-agent", "gridworld", "gymnasium", "pettingzoo"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "numpy>=1.26,<3.0",
    "gymnasium>=0.29",
    "pettingzoo>=1.24",
]

[project.optional-dependencies]
jax = [
    "jax>=0.4.20,<1.0",
]
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.24",
]
dev = [
    "pytest>=7.0",
    "ruff>=0.9.0",
    "cogrid[docs]",
]
all = [
    "cogrid[jax]",
    "cogrid[dev]",
]

[project.urls]
Homepage = "https://github.com/chasemcd/cogrid"
Repository = "https://github.com/chasemcd/cogrid"

[tool.setuptools.dynamic]
version = {attr = "cogrid.__version__"}

[tool.setuptools.packages.find]
include = ["cogrid*"]

# ---------------------------------------------------------------------------
# Ruff
# ---------------------------------------------------------------------------

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "D"]

[tool.ruff.lint.per-file-ignores]
"cogrid/tests/*" = ["D100", "D101", "D102", "D103", "D104", "D107"]
"cogrid/test_*.py" = ["D100", "D101", "D102", "D103", "D104", "D107"]
"cogrid/benchmarks/*" = ["D100", "D101", "D102", "D103", "D104"]
"examples/*" = ["D"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.format]
docstring-code-format = true

# ---------------------------------------------------------------------------
# pytest
# ---------------------------------------------------------------------------

[tool.pytest.ini_options]
testpaths = ["cogrid/tests"]
```

### Ruff Per-File-Ignores (recommended)

Source: [Ruff settings docs](https://docs.astral.sh/ruff/settings/)

```toml
[tool.ruff.lint.per-file-ignores]
# Test files don't need docstrings
"cogrid/tests/*" = ["D100", "D101", "D102", "D103", "D104", "D107"]
"cogrid/test_*.py" = ["D100", "D101", "D102", "D103", "D104", "D107"]
# Benchmarks don't need docstrings
"cogrid/benchmarks/*" = ["D100", "D101", "D102", "D103", "D104"]
# Examples are informal
"examples/*" = ["D"]
```

### Ruff Fix Workflow (recommended order)

```bash
# Step 1: Format all files (deterministic, safe)
ruff format cogrid/

# Step 2: Auto-fix safe lint violations
ruff check cogrid/ --fix

# Step 3: Check remaining violations
ruff check cogrid/ --statistics

# Step 4: Fix remaining violations manually

# Step 5: Final verification
ruff format --check cogrid/
ruff check cogrid/
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| setup.py + setup.cfg | pyproject.toml (PEP 621) | PEP 621 accepted June 2021; setuptools support since v61 (2022) | Single file replaces 2-4 config files |
| flake8 + isort + black | ruff (lint + format) | ruff format stable since Dec 2023 | 10-100x faster; single tool; single config |
| requirements.txt for deps | pyproject.toml `[project.optional-dependencies]` | PEP 621 | Dependencies live next to metadata |
| setup.py `version=` | `dynamic = ["version"]` with `attr:` | setuptools v46.4+ (2020) | Version lives in code, not config |

**Deprecated/outdated:**
- `setup.py` as primary config: Replaced by pyproject.toml. Still functional but not recommended for new/migrated projects.
- `setup.cfg [metadata]`: Superseded by PEP 621 `[project]` table.
- `MANIFEST.in`: setuptools now uses package discovery from pyproject.toml; not needed for pure-Python.
- `description-file = README.rst` in setup.cfg: Use `readme = "README.md"` in `[project]`.

## Codebase-Specific Findings

### Current Violation Counts (at target config: 100-char, E+F+I+UP+D, Google convention)

| Rule | Count | Auto-fixable | Notes |
|------|-------|-------------|-------|
| D102 (undocumented method) | 104 | No | Largest category; many in test files (use per-file-ignores) |
| I001 (unsorted imports) | 67 | Yes | All auto-fixable by ruff |
| D101 (undocumented class) | 50 | No | Many in test files |
| E501 (line too long) | 47 | No | Manual rewrap needed (at 100-char limit, much fewer than 88-char default) |
| D107 (undocumented init) | 26 | No | Many in test files |
| D103 (undocumented function) | 23 | No | Some in test files |
| D415 (missing terminal punctuation) | 23 | No | Add periods to existing docstrings |
| D100 (undocumented module) | 20 | No | Most are `__init__.py` or test files |
| D205 (blank line after summary) | 14 | No | Manual fix |
| D212 (multi-line summary first line) | 14 | Yes | Auto-fixable |
| F841 (unused variable) | 11 | No | Manual review needed |
| D104 (undocumented package) | 11 | No | `__init__.py` files; per-file-ignores or add docstrings |
| D202 (blank after function) | 8 | Yes | Auto-fixable |
| F821 (undefined name) | 4 | No | Real bugs (see Pitfall 6) |
| Others | <10 each | Mixed | UP008, UP034, E402, E722, F401, F402, D200, D209, D402, D403, D416, D417 |
| **Total** | **440** | **~97** | -- |
| **Format violations** | **45 files** | Yes | `ruff format` fixes all |

### Per-File-Ignores Will Significantly Reduce Counts

With per-file-ignores exempting test/benchmark files from D100-D104/D107, the D-rule count drops substantially. Rough estimate: ~100-150 fewer violations (tests have ~60% of undocumented-public-* errors).

### Files to Remove (Legacy Packaging)

1. `/Users/chasemcd/Repositories/cogrid/setup.py` -- replaced by pyproject.toml
2. `/Users/chasemcd/Repositories/cogrid/setup.cfg` -- replaced by pyproject.toml
3. `/Users/chasemcd/Repositories/cogrid/MANIFEST.in` -- not needed for pure-Python with setuptools package discovery
4. `/Users/chasemcd/Repositories/cogrid/requirements.txt` -- replaced by `[project.dependencies]`
5. `/Users/chasemcd/Repositories/cogrid/docs/requirements.txt` -- replaced by `[project.optional-dependencies.docs]`
6. `/Users/chasemcd/Repositories/cogrid/update_pypi.sh` -- uses `setup.py sdist bdist_wheel`; needs rewrite or removal
7. `/Users/chasemcd/Repositories/cogrid/.readthedocs.yaml` -- references Sphinx; will be replaced in docs phase

### Build Artifacts to Clean (not in git, but on disk)

- `build/` directory (stale bdist/lib artifacts)
- `dist/` directory (old cogrid-0.1.2 wheel/tarball)
- `cogrid.egg-info/` directory (stale PKG-INFO)

These are already in `.gitignore` so they won't be committed, but the success criteria mentions cleaning stale build artifacts.

### README Conversion (PKG-07)

Current README is RST (`README.rst`). Requirements say "README converted from RST to Markdown with project badges." The existing RST content is short (58 lines) and contains:
- Logo image reference
- Brief description
- Installation instructions
- Citation block

Conversion to Markdown is straightforward. Badges for PyPI, CI, and docs should be added (though CI and docs URLs may not exist yet; placeholder or conditional).

## Open Questions

1. **JAX version bounds vs Python 3.10**
   - What we know: JAX >=0.5.0 requires Python >=3.11. JAX 0.4.38 was the last 0.4.x release (Dec 2024), likely the last to support Python 3.10. `jax>=0.4.20,<1.0` will work because pip will resolve to 0.4.x on Python 3.10 and 0.5+/0.9.x on Python 3.11+.
   - What's unclear: Whether `jax>=0.4.20,<1.0` or `jax>=0.4.20` (no upper bound) is better. The `<1.0` cap was in the user's example.
   - Recommendation: Use `jax>=0.4.20,<1.0` as the user specified. This is safe; JAX is still pre-1.0 (latest is 0.9.0.1). If JAX releases 1.0, the bound can be relaxed.

2. **update_pypi.sh replacement**
   - What we know: Current script uses `python setup.py sdist bdist_wheel && twine upload dist/*`. After migration, `python -m build && twine upload dist/*` is the modern equivalent.
   - What's unclear: Whether to keep a publishing script or defer to CI/CD (Phase 26).
   - Recommendation: Remove `update_pypi.sh` in this phase. Publishing automation belongs in CI/CD (Phase 26). If needed, a developer can run `python -m build && twine upload dist/*` manually.

3. **`.readthedocs.yaml` removal timing**
   - What we know: It references Sphinx, which will be replaced by MkDocs in a later phase.
   - What's unclear: Whether to remove it now or in the docs phase.
   - Recommendation: Remove it in this phase since it references `docs/requirements.txt` which is being deleted. The docs phase can create a new one if needed.

4. **D rules on `cogrid/run_interactive.py` and `cogrid/test_overcooked_env.py`**
   - What we know: `run_interactive.py` has real bugs (undefined names `ort`, `special`). `test_overcooked_env.py` is a test file located outside `cogrid/tests/`.
   - Recommendation: Fix the undefined name bugs in `run_interactive.py`. For `test_overcooked_env.py`, either move it to `cogrid/tests/` or add it to per-file-ignores. The `"cogrid/test_*.py"` pattern in per-file-ignores already covers it.

## Sources

### Primary (HIGH confidence)
- [setuptools pyproject.toml documentation](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) -- full pyproject.toml configuration reference
- [PEP 621](https://peps.python.org/pep-0621/) -- project metadata specification
- [Ruff settings reference](https://docs.astral.sh/ruff/settings/) -- all ruff configuration options
- [Ruff configuration guide](https://docs.astral.sh/ruff/configuration/) -- pyproject.toml integration
- Local codebase analysis -- ruff violation counts, file inventory, dependency inspection

### Secondary (MEDIUM confidence)
- [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) -- pyproject.toml best practices
- [JAX versioning policy](https://docs.jax.dev/en/latest/jep/9419-jax-versioning.html) -- version compatibility rules
- [JAX deprecation policy](https://docs.jax.dev/en/latest/deprecation.html) -- Python version support timelines
- [NumPy endoflife.date](https://endoflife.date/numpy) -- Python version support per NumPy release
- PyPI index queries (local `pip index versions`) -- confirmed latest versions of all dependencies

### Tertiary (LOW confidence)
- JAX 0.4.38 as "last version supporting Python 3.10" -- inferred from 0.5.0/0.9.x requiring >=3.11, but not directly verified via PyPI metadata (PyPI challenge page blocked fetch). Functional test: the user's local environment runs Python 3.11 with JAX 0.4.38 successfully.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- setuptools and ruff are well-documented; all versions verified via `pip index versions`
- Architecture: HIGH -- pyproject.toml structure follows official setuptools documentation; ruff config verified against docs
- Pitfalls: HIGH -- violation counts measured by running ruff on the actual codebase; edge cases identified from real files
- Dependency versions: MEDIUM -- JAX Python 3.10 cutoff version is inferred, not directly confirmed

**Research date:** 2026-02-16
**Valid until:** 2026-03-16 (stable domain; ruff and setuptools change slowly)
