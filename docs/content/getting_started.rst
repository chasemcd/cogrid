Overview
=========

CoGrid Environment
-------------------

CoGrid Configs
---------------

Much of the environment is configured through the ``config`` object that is passed to the environment. It has 
several key entries that are expected in the base class, but any additional entries can be added as needed
in custom environments:

.. code-block:: python

    env_config = {
        "name": ENV_NAME,  # the name of the environment from the registry
        "num_agents": 2,  # number of agents to place in the environment
        "action_set": ACTION_SET,  # the action space for the agents
        "features": [FEATURE_NAME],  # a list of features to include in the observation. Pass a dictioanry to specify per-agent.
        "rewards": ["delivery_reward"],  # a list of the reward functions to include in the environment.
        "grid_gen_kwargs": {"load": ENV_LAYOUT_NAME,},  # the layout of the environment if registered in the grid registry
        "max_steps": MAX_T,  # the number of timesteps before terminating
    }

Grid
------------


Agents
------------


Features
------------




Rewards
------------


