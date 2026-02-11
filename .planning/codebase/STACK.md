# Technology Stack

**Analysis Date:** 2026-02-10

## Languages

**Primary:**
- Python 3.10+ - Primary language for entire codebase

## Runtime

**Environment:**
- Python 3.10+ (as specified in `setup.py`)

**Package Manager:**
- pip - Python package manager
- setuptools 68.2.2 - Build and packaging tool
- Lockfile: No lockfile present (requirements.txt available)

## Frameworks

**Core:**
- Gymnasium - RL environment API for single/multi-agent environments, replaces OpenAI Gym
- PettingZoo 1.24.3 - Multi-agent environment framework, provides `ParallelEnv` base class
- NumPy 1.26.4 - Numerical computation and array operations

**Documentation:**
- Sphinx - Documentation generation (config: `docs/conf.py`)
- sphinx-rtd-theme 2.0.0 - Read the Docs theme for documentation
- Sphinx extensions: autodoc, viewcode, todo

**Optional:**
- PyGame - Optional rendering for environment visualization (imported with try/except in `cogrid/cogrid_env.py`)

## Key Dependencies

**Critical:**
- numpy 1.26.4 - Required for numerical operations, grid representations, agent state
- gymnasium - Required for environment API compliance and observation/action spaces
- pettingzoo 1.24.3 - Required for parallel multi-agent environment interface (`CoGridEnv` inherits from `pettingzoo.ParallelEnv`)

**Development/Documentation:**
- sphinx-rtd-theme 2.0.0 - For documentation site generation
- setuptools 68.2.2 - For package building and distribution

## Configuration

**Environment:**
- No `.env` files detected - configuration via Python config dictionaries
- Python version specified in:
  - `setup.py`: `python_requires=">=3.10"`
  - `.readthedocs.yaml`: `python: "3.10"`

**Build:**
- `pyproject.toml` - Minimal PEP 517 build configuration using setuptools
- `setup.py` - Main package configuration with dependencies and metadata
- `setup.cfg` - Metadata configuration for package description
- `MANIFEST.in` - Specifies files to include in source distribution

**Documentation:**
- `.readthedocs.yaml` - Read the Docs build configuration
- `docs/conf.py` - Sphinx configuration
- `docs/requirements.txt` - Documentation-specific dependencies

## Platform Requirements

**Development:**
- Python 3.10 or higher
- Standard system libraries (numpy requirements)
- Optional: PyGame for visualization (soft dependency)

**Production:**
- Python 3.10+
- PyPI distribution available at https://pypi.org/project/cogrid/
- Cross-platform: OS Independent (specified in `setup.py` classifiers)

## Distribution

**PyPI Package:**
- Name: cogrid
- Current version: 0.0.16 (as of Feb 2026)
- Distribution status: Pre-Alpha (Development Status :: 2 - Pre-Alpha)
- License: Apache Software License

---

*Stack analysis: 2026-02-10*
