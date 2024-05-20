Welcome to CoGrid!
===============================================

.. raw:: html

   <div style="display: flex; justify-content: center; padding-left: 40%;">
      <a href="https://www.github.com/chasemcd/cogrid">
         <img src="_static/images/cogrid_logo_nobg.png" alt="CoGrid Logo" style="width: 50%; display: block;">
      </a>
   </div>


.. image:: https://img.shields.io/github/stars/chasemcd/cogrid
   :target: https://github.com/chasemcd/cogrid
   :alt: GitHub Repo stars


CoGrid is a library that extends Minigrid to support multi-agent environments. 
It is built on top of the Minigrid library, which is a minimalistic gridworld environment developed
originally by Chevalier-Boisvert et al. (2023) (https://arxiv.org/abs/2306.13831). CoGrid has several
differentiating factors from Minigrid:

1. Multi-agent support. CoGrid supports multiple agents in the same environment, each with their own
   observation space and action space. Whereas Minigrid's environment logic is centered around a single 
   agent interacting with a ``Grid`` of ``WorldObj`` objects, CoGrid's environment logic also tracks ``Agent`` objects as
   unique objects, allowing an arbitrary number to exist in the environment.
2. ``Reward`` modularization. CoGrid allows for the creation of custom ``Reward`` objects that can be added to the
   environment. Each ``Reward`` is used to calculate the reward for each agent at each step, and can be
   used to create complex reward functions that depend on the state of the environment, the actions of
   other agents, etc.
3. ``Feature`` modularization. Similar to rewards, CoGrid allows for the creation of custom ``Feature`` objects that can be added to the
   environment. These ``Feature`` classes are used to construct each agent's observation
   space, such as the location of other agents, an image of the environment, etc. 

CoGrid utilizes the PettingZoo API to standardize the multi-agent environment interface.


CoGrid is a library that extends Minigrid to support multi-agent environments. 
It is built on top of the Minigrid library, which is a minimalistic gridworld environment developed
originally by Chevalier-Boisvert et al. (2023) (https://arxiv.org/abs/2306.13831). CoGrid has several
differentiating factors from Minigrid:

1. Multi-agent support. CoGrid supports multiple agents in the same environment, each with their own
   observation space and action space. Whereas Minigrid's environment logic is centered around a single 
   agent interacting with a ``Grid`` of ``WorldObj`` objects, CoGrid's environment logic also tracks ``Agent`` objects as
   unique objects, allowing an arbitrary number to exist in the environment.
2. ``Reward`` modularization. CoGrid allows for the creation of custom ``Reward`` objects that can be added to the
   environment. Each ``Reward`` is used to calculate the reward for each agent at each step, and can be
   used to create complex reward functions that depend on the state of the environment, the actions of
   other agents, etc.
3. ``Feature`` modularization. Similar to rewards, CoGrid allows for the creation of custom ``Feature`` objects that can be added to the
   environment. These ``Feature`` classes are used to construct each agent's observation
   space, such as the location of other agents, an image of the environment, etc. 

CoGrid utilizes the PettingZoo API to standardize the multi-agent environment interface.


.. raw:: html

   <div style="display: flex; justify-content: center; margin-top: 20px;">
      <img src="_static/images/sr_example.gif" alt="Example GIF" style="width: 75%; display: block;">
   </div>

Installation
------------

Install directly from the master branch (``pip install cogrid`` coming soon!):

    .. code-block:: bash

        pip install git+https://www.github.com/chasemcd/cogrid.git


Citation
---------

If you use CoGrid in your research, please cite the following paper:

    .. code-block:: bash

        @article{mcdonald2024cogrid,
         author  = {McDonald, Chase and Gonzalez, Cleotilde},
         title   = {CoGrid and Interactive Gym: A Framework for Multi-Agent Experimentation},
         year    = {forthcoming},
         }


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   content/getting_started
   content/examples
   content/cogrid_env
   content/custom_env
