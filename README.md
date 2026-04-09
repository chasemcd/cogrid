[![PyPI version](https://img.shields.io/pypi/v/cogrid)](https://pypi.org/project/cogrid/)
[![Python](https://img.shields.io/pypi/pyversions/cogrid)](https://pypi.org/project/cogrid/)
[![License](https://img.shields.io/github/license/chasemcd/cogrid)](https://github.com/chasemcd/cogrid/blob/main/LICENSE)
[![Coverage](https://raw.githubusercontent.com/chasemcd/cogrid/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/chasemcd/cogrid/blob/python-coverage-comment-action-data/htmlcov/index.html)

<p align="center">
  <img src="docs/assets/images/cogrid_logo_clean.png" alt="CoGrid Logo" width="300">
</p>

CoGrid is a library for creating multi-agent grid-world environments for reinforcement learning research. It features a functional array-based simulation core, pluggable components (rewards, features, objects), and dual NumPy/JAX backend support.

CoGrid utilizes the parallel [PettingZoo](https://pettingzoo.farama.org/) API to standardize the multi-agent environment interface. The JAX API is similar to that of [JaxMARL](https://github.com/FLAIROx/JaxMARL).

CoGrid is designed to offer an approachable API for environment customization, compatibility with standard tooling, and pre-build benchmark environments. Full documentation is available at [cogrid.readthedocs.io](https://cogrid.readthedocs.io).

<p align="center">
  <img src="docs/assets/images/v2_layouts.png" alt="OvercookedV2 Layouts" width="60%">
</p>

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

> [!IMPORTANT]
> CoGrid has gone through a major overhaul and the API has changed significantly. If you need the previous version, you can install it with `pip install cogrid==0.0.16`.


## Citation

If you use CoGrid in your research, please cite the following paper:

```bibtex
@article{mcdonald2024cogrid,
  author  = {McDonald, Chase and Gonzalez, Cleotilde},
  title   = {CoGrid and the Multi-User Gymnasium: A Framework for Multi-Agent Experimentation},
  year    = {forthcoming},
}
```


## Acknowledgements

This work builds on the invaluable efforts of many others:

```bibtex
@article{carroll2019utility,
  title={On the utility of learning about humans for human-ai coordination},
  author={Carroll, Micah and Shah, Rohin and Ho, Mark K and Griffiths, Tom and Seshia, Sanjit and Abbeel, Pieter and Dragan, Anca},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}

@article{rutherford2024jaxmarl,
  title={Jaxmarl: Multi-agent rl environments and algorithms in jax},
  author={Rutherford, Alexander and Ellis, Benjamin and Gallici, Matteo and Cook, Jonathan and Lupu, Andrei and Ingvarsson, Gar{\dh}ar and Willi, Timon and Hammond, Ravi and Khan, Akbir and de Witt, Christian S and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={50925--50951},
  year={2024}
}


@article{gessler2025overcookedv2,
  title={Overcookedv2: Rethinking overcooked for zero-shot coordination},
  author={Gessler, Tobias and Dizdarevic, Tin and Calinescu, Ani and Ellis, Benjamin and Lupu, Andrei and Foerster, Jakob Nicolaus},
  journal={arXiv preprint arXiv:2503.17821},
  year={2025}
}

```