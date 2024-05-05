Environments
===============================================

Here we overview the environments provided in CoGrid.

Overcooked
-----------

.. automodule:: cogrid.envs.overcooked.overcooked
    :members:

To illustrate the ease of building environments, we reimplement the Overcooked-AI environment by Carroll et al. (2019)
in CoGrid. The Overcooked-AI environment is a cooperative multi-agent environment where agents must work together to
combine ingredients to make dishes and deliver them earn a reward. 

We provide an implementation of the Cramped Room layout, pictured below. 

.. image:: ../_static/images/overcooked_grid.png
    :align: center


Search & Rescue
----------------


.. automodule:: cogrid.envs.search_rescue.search_rescue
    :members:

We also provide an implementation of the Search & Rescue environment, where agents must work together to find and rescue
victims in the environment. This environment is inspired by the work by the Minimap task (Nguyen & Gonzalez, 2023). 

In this environment, two agents must work together to find and rescue three types of victims: red, yellow, and green.
Green victims are rescued simply by selecting the toggle action when adjacent to them. Yellow victims require
that the agent equip a medical kit before rescuing them. Lastly, red victims require that one agent equip the medical kit
and the other agent also must be adjacentq to the victim. There are also obstacles: locked doors and rubble. Agents must 
find keys to unlock doors and use pickaxes to clear rubble. This task includes both sequential (clear obstacles, finding items)
and parallel (rescuing victims) collaboration. 

.. image:: ../_static/images/search_rescue_grid.png
    :align: center