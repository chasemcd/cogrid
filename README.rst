CoGrid
===============================================

.. image:: https://readthedocs.org/projects/example-sphinx-basic/badge/?version=latest
    :target: https://example-sphinx-basic.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   content/getting_started
   content/examples
   content/cogrid_env
   content/custom_env



CoGrid is a library that extends Minigrid to support multi-agent environments. 
It is built on top of the Minigrid library, which is a minimalistic gridworld environment developed
originally by Chevalier-Boisvert et al. (2023) (https://arxiv.org/abs/2306.13831). CoGrid has several
differentiating factors from Minigrid:

1. Multi-agent support. CoGrid supports multiple agents in the same environment, each with their own
   observation space and action space. Whereas Minigrid's environment logic is centered around a single 
   agent interacting with a ``Grid`` of ``WorldObj``s, CoGrid's environment logic also tracks ``Agent``s abs
   unique objects, allowing an arbitrary number to exist in the environment.
2. ``Reward`` modularization. CoGrid allows for the creation of custom ``Reward``s that can be added to the
   environment. These ``Reward``s are used to calculate the reward for each agent at each step, and can be
   used to create complex reward functions that depend on the state of the environment, the actions of
   other agents, etc.
3. ``Feature`` modularization. Similar to rewards, CoGrid allows for the creation of custom ``Feature``s that can be added to the
   environment. These ``Feature``s are used to construct the agent's observation
   space, such as the location of other agents, an image of the environment, etc. 

CoGrid utilizes the PettingZoo API to standardize the multi-agent environment interface.

Installation
------------
To install CoGrid, you can use the PyPi distribution:

    .. code-block:: bash

        pip install cogrid

Or directly from the master branch:

    .. code-block:: bash

        pip install git+https://www.github.com/chasemcd/cogrid.git