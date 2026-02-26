# Contributing

Thank you for your interest in contributing to CoGrid! This guide walks through
the complete workflow: fork, install, develop, test, and submit.

## Getting Started

1. **Fork** the repository on GitHub:
   [chasemcd/cogrid](https://github.com/chasemcd/cogrid)

2. **Clone** your fork:

    ```bash
    git clone https://github.com/YOUR_USERNAME/cogrid.git
    cd cogrid
    ```

3. **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

4. **Install development dependencies:**

    ```bash
    pip install -e ".[dev]"
    ```

    This installs the package in editable mode along with test tools (pytest,
    ruff), documentation tools (mkdocs, mkdocstrings), and all core
    dependencies. See the [Getting Started](https://cogrid.readthedocs.io) guide for
    basic installation options.

## Development Workflow

1. Create a feature branch:

    ```bash
    git checkout -b feature/your-feature
    ```

2. Make your changes.

3. Run the test and lint checks (see sections below).

4. Commit with a descriptive message and push:

    ```bash
    git add your_files.py
    git commit -m "feat: add your feature description"
    git push origin feature/your-feature
    ```

5. Open a pull request against `main` on GitHub.

## Running Tests

CoGrid uses [pytest](https://docs.pytest.org/) for testing. The test suite
covers both the NumPy and JAX backends.

```bash
# Run all tests
pytest tests/ -v

# Run only NumPy tests (default, works on Python 3.10+)
pytest tests/

# Run only JAX tests (requires JAX: pip install cogrid[jax])
pytest tests/ -k jax
```

> **Tip: JAX requirement**
> JAX tests require Python 3.11+ and JAX >= 0.5.x. Install the JAX extras
> with `pip install -e ".[jax]"` before running JAX-specific tests.

## Code Style

CoGrid uses [Ruff](https://docs.astral.sh/ruff/) for both linting and
formatting. The configuration lives in `pyproject.toml`.

```bash
# Check for lint issues
ruff check cogrid/

# Auto-fix lint issues
ruff check --fix cogrid/

# Check formatting
ruff format --check cogrid/

# Auto-format
ruff format cogrid/
```

**Style conventions:**

- **Line length:** 100 characters
- **Docstrings:** Google convention (see
  [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#381-docstrings))
- **All public classes, methods, and functions** must have docstrings

## Submitting Changes

1. Ensure all tests pass: `pytest tests/ -v`
2. Ensure lint passes: `ruff check cogrid/`
3. Ensure formatting passes: `ruff format --check cogrid/`
4. Push to your fork: `git push origin feature/your-feature`
5. Open a pull request against `main`

CI runs automatically on every pull request:

- **Tests** on Python 3.10, 3.11, and 3.12
- **Lint** via `ruff check`
- **JAX tests** on Python 3.12

## Previewing Documentation

To preview documentation changes locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) to see the live preview.
Changes to Markdown files are hot-reloaded automatically.

## Creating a New Environment

New environments live in `cogrid/envs/your_env/` and follow the same pattern
as the built-in Overcooked environment.

The component API provides two registration decorators and a subclassing pattern:

- `@register_object_type` -- define grid objects with interaction behavior
- `@register_feature_type` -- define observation features
- `Reward` subclasses -- define reward functions (instantiated and passed in `config["rewards"]`)

For a complete walkthrough of building an environment from scratch, see the
[Custom Environment Tutorial](https://cogrid.readthedocs.io).

> **Note: No environment-specific logic in core**
> All environment-specific behavior must be implemented through the component
> API. The core engine (`cogrid/core/` and `cogrid/cogrid_env.py`) must remain
> environment-agnostic.
