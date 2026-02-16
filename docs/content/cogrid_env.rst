CoGrid Environments
===============================================

CoGrid environment classes are all based on ``CoGridEnv`` class, which contains all of the environment logic. Most important
are the ``step`` and ``reset`` methods, as in any Gymnasium-style environment. The ``reset`` method provides the initial state and ``step`` takes as input a dictionary of agent 
actions and conducts the state transition (e.g., ``T(S, A) -> S'``) and returns the new observations, reward, and whether the
episode is done (either terminated or truncated), alongside any reported information in the ``infos`` return dictionaries.

.. automodule:: cogrid.cogrid_env
    :members:


.. automodule:: cogrid.core.grid
    :members:


.. automodule:: cogrid.core.grid_object
    :members:


.. automodule:: cogrid.core.agent
    :members:


.. automodule:: cogrid.core.reward
    :members:


