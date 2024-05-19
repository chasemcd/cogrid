CoGrid
=======

.. .. raw:: html

..    <div style="text-align: center;">
..        <a href="https://www.github.com/chasemcd/cogrid">
..            <img src="docs/_static/images/cogrid_logo_nobg.png" alt="CoGrid Logo" style="width: 25%; display: block; margin: 0 auto;">
..        </a>
..    </div>

.. raw:: html

   <table style="width:100%; border:0;">
       <tr>
           <td style="text-align: center;">
               <a href="https://www.github.com/chasemcd/cogrid">
                   <img src="docs/_static/images/cogrid_logo_nobg.png" alt="CoGrid Logo" style="width: 50%;">
               </a>
           </td>
       </tr>
   </table>


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

Installation
------------
.. To install CoGrid, you can use the PyPi distribution:

   ..  .. code-block:: bash

   ..      pip install cogrid

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