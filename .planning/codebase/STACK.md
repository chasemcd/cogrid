# Technology Stack

**Analysis Date:** 2026-01-19

## Languages

**Primary:**
- Python 3.10+ (required by `setup.py`)

**Secondary:**
- None

## Runtime

**Environment:**
- Python 3.10+ (development tested on 3.11.7)
- No containerization detected

**Package Manager:**
- pip with setuptools
- Lockfile: Not present (only `requirements.txt` with pinned versions)

## Frameworks

**Core:**
- PettingZoo 1.24.3 - Multi-agent reinforcement learning environment API
- Gymnasium 0.29.1 - Reinforcement learning environment abstraction (spaces, etc.)
- NumPy 1.26.4 - Numerical computing and array operations

**Testing:**
- pytest - Test runner (`.pytest_cache/` directory exists)
- No formal test configuration file detected

**Build/Dev:**
- setuptools 68.2.2+ - Package building
- wheel - Distribution format
- twine - PyPI upload (in `update_pypi.sh`)

**Documentation:**
- Sphinx - Documentation generator
- sphinx-rtd-theme 2.0.0 - ReadTheDocs theme
- ReadTheDocs - Documentation hosting

## Key Dependencies

**Critical:**
- `pettingzoo==1.24.3` - Core multi-agent environment interface; `CoGridEnv` inherits from `pettingzoo.ParallelEnv`
- `gymnasium` - Provides `spaces.Dict`, `spaces.Discrete` for observation/action spaces
- `numpy==1.26.4` - Grid representation, array operations throughout

**Optional (Runtime):**
- `pygame` - Interactive visualization (`render_mode="human"`), imported conditionally
- `cv2` (opencv-python) - Text rendering on images, imported conditionally
- `onnxruntime` - ONNX model inference for AI agents (commented out in `run_interactive.py`)
- `scipy` - Softmax for action sampling (commented out in `run_interactive.py`)

**Infrastructure:**
- `twine` - PyPI package upload
- `sphinx` - Documentation building

## Configuration

**Environment:**
- No `.env` files used
- Configuration passed via Python dictionaries to `CoGridEnv.__init__(config={})`
- Key config keys: `name`, `max_steps`, `num_agents`, `action_set`, `features`, `rewards`, `grid`

**Build:**
- `pyproject.toml`: Build system config (setuptools backend)
- `setup.py`: Package metadata, version (0.0.16), dependencies
- `setup.cfg`: Metadata file pointer
- `MANIFEST.in`: Include files for distribution
- `.readthedocs.yaml`: Documentation build config

## Platform Requirements

**Development:**
- Python 3.10 or higher
- pip for package management
- Optional: pygame for interactive visualization

**Production:**
- PyPI package: `pip install cogrid`
- Minimal dependencies (numpy, gymnasium, pettingzoo)

## Version Management

**Current Version:** 0.0.16 (defined in `setup.py`)

**PyPI Publishing:**
- Manual via `update_pypi.sh`:
  ```bash
  rm -rf dist/
  python setup.py sdist bdist_wheel
  twine upload dist/*
  ```

**No CI/CD pipeline detected** - No `.github/` directory or CI configuration files.

---

*Stack analysis: 2026-01-19*
