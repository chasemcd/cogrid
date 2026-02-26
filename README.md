[![PyPI version](https://img.shields.io/pypi/v/cogrid)](https://pypi.org/project/cogrid/)
[![Python](https://img.shields.io/pypi/pyversions/cogrid)](https://pypi.org/project/cogrid/)
[![License](https://img.shields.io/github/license/chasemcd/cogrid)](https://github.com/chasemcd/cogrid/blob/main/LICENSE)
[![Coverage](https://raw.githubusercontent.com/chasemcd/cogrid/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/chasemcd/cogrid/blob/python-coverage-comment-action-data/htmlcov/index.html)

# CoGrid

![CoGrid Logo](docs/assets/images/cogrid_logo_nobg.png)

CoGrid is a library for creating multi-agent grid-world environments for reinforcement learning research. It features a functional array-based simulation core, pluggable components (rewards, features, objects), and dual NumPy/JAX backend support.

CoGrid utilizes the parallel [PettingZoo](https://pettingzoo.farama.org/) API to standardize the multi-agent environment interface.

![Example](docs/assets/images/overcooked_grid.png)

## Installation

Install from PyPI:

```bash
pip install cogrid
```

To install with JAX backend support:

```bash
pip install cogrid[jax]
```

For development (includes test, lint, and docs tools):

```bash
pip install cogrid[dev]
```

## Citation

If you use CoGrid in your research, please cite the following paper:

```bibtex
@article{mcdonald2024cogrid,
  author  = {McDonald, Chase and Gonzalez, Cleotilde},
  title   = {CoGrid and Interactive Gym: A Framework for Multi-Agent Experimentation},
  year    = {forthcoming},
}
```
