Overview
=========

CoGrid Environment
-------------------

The ``CoGridEnv`` class is the base class for all environments in the CoGrid framework. All logic for handling grid interactions is included 
in the base class, and custom environments can be created by subclassing this class and adding additional logic via the environment hooks. 
For instance ``CoGridEnv.on_step`` is called at each timestep, and can be overridden to add custom logic to the environment, such as that for actions
beyond the provided set. 

CoGrid Configs
---------------

Much of the environment is configured through the ``config`` object that is passed to the environment. It has 
several key entries that are expected in the base class, but any additional entries can be added as needed
in custom environments:

.. code-block:: python

    env_config = {
        "name": ENV_NAME,  # the name of the environment from the registry
        "num_agents": NUM_AGENTS,  # number of agents to place in the environment
        "action_set": ACTION_SET,  # the set of available actions to the agents
        "features": [FEATURE_NAME],  # a list of features to include in the observation. Pass a dictioanry to specify per-agent.
        "rewards": [REWARD_NAME],  # a list of the reward functions to include in the environment.
        "grid_gen_kwargs": {"load": ENV_LAYOUT_NAME,},  # the layout of the environment if registered in the grid registry
        "max_steps": MAX_T,  # the number of timesteps before terminating
    }

Grid
------------

Each ``CoGridEnv`` has a ``grid`` attribute that is used to store the state of the environment. The ``CoGridEnv.grid`` is a ``Grid`` instance, which tracks
all of the ``GridObj`` instances in the environment. Every object, including agents, is a subclass of a ``GridObj``, which defines the interaction logic for each 
unique object in the environment. 


Agents
------------

There are two kinds of agents in ``cogrid``. The first is a ``GridObj`` agent, which is used to represent the agent in the ``Grid``. The second is the ``Agent`` class, which is represented
in the ``CoGridEnv`` and defines any custom logic that agents need to interact with the environment. In simple environments, there is no need to subclass the ``Agent``; however,
in some instances it may be necessary to add additional logic to the agent, e.g., implementing conditions that specify when an agent can take particular actions. For an example, see the ``OvercookedAgent`` class.


Features
------------

``Feature`` objects define a particular piece of information that is include in each agent's observation. 
All ``Feature`` objects should be registered using te ``register_feature`` function by passing in the class and feature name. 
One that has been done, features names can be passed to environment configurations in one of three formats (1) as a single 
feature name that will be utilized for all agents, (2) a list of feature names that will all be used for all agents, or (3)
a dictionary keyed by agent ID with values of lists of feature names:

.. code-block:: python
    
    {
        "features": "feature_name_1" | ["feature_name_1, feature_name_2"] | {0: ["feature_name_1"], 1: ["feature_name_2"]
    }


All features will be returned in a dictionary observation space.


Rewards
------------

``Reward`` objects define the logic of the form ``R(s_t, a_t, s_{t+1}) -> r_t``. Each ``Reward`` utilizes the ``Reward.calculate_reward`` function,
which takes as input the previous and current state (as ``Grid`` objects) and the actions of all agents. Each returns a dictionary keyed by agent ID with values 
that represent the reward calculated for that agent. Given a list of ``Reward`` objects in the configuration, each are added at each step in the reward returned
by ``CoGridEnv.step()``. Importantly, ``Reward`` objects are not the only way to define rewards. Environment hooks (e.g., ``CoGridEnv.on_toggle``) can be overwritten
to include logic that will add to ``CoGridEnv.reward_this_step``, a dictionary keyed by agent IDs with additional rewards at each step. These rewards
will be added to those calculated by each ``Reward`` at ever step. 

Similar to features, ``Reward`` objects should be registered using ``register_reward``, then they can be passed into the configuration as a list. 


