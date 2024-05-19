Examples
==============================

In this section, we'll walk through building a custom environment by implementing the Search & Rescue environment. This task 
is derived from Nguyen & Gonzalez (2023), mimicking the multi-agent adaptation in McDonald et al. (in review). 
 
The Search & Rescue environment is one where multiple agents work together to rescue victims. They do so by using tools
that grant them abilities: a ``PickAxe`` can be used to clear rubble, a ``MedKit`` can be used to rescue a ``RedVictim`` or ``YellowVictim``,
and a ``Key`` can be used to unlock doors. Here is all the logic we must implement:

1. Create grid objects for ``Rubble``, ``PickAxe``, ``MedKit``, ``RedVictim``, and ``YellowVictim`` (note that ``Key`` and ``Door`` are provided in CoGrid).
2. Define the interaction logic:
    - Agents can save a ``GreenVictim`` by using the ``toggle`` action when facing them. 
    - Agents can save a ``YellowVictim`` if they have a ``MedKit`` in their inventory and use the ``toggle`` action when facing them.
    - Agents can save a ``RedVictim`` if one agent has a ``MedKit`` in their inventory, uses the ``toggle`` action when facing them, and the other agent also uses the ``toggle`` action within 30 timesteps of the first toggle.
    - Agents can clear ``Rubble`` if they have a ``PickAxe`` in their inventory and use the ``toggle`` action when facing the ``Rubble``.
3. Define the reward logic:
    - Agents receive a reward of 1.0 for saving a ``GreenVictim``.
    - Agents receive a reward of 2.0 for saving a ``YellowVictim``.
    - Agents receive a reward of 3.0 for saving a ``RedVictim``.
4. Define the ``Feature`` objects that the agents will use for their observation space:
    - Encoding of their partially observed view, included in CoGrid.


Environment
------------

Grid Objects
^^^^^^^^^^^^^

We'll start by defining the grid objects that will be used in the environment. We'll create classes for ``Rubble``, ``PickAxe``, ``MedKit``, ``GreenVictim``, ``RedVictim``, and ``YellowVictim``.
Here we provide an example for ``Rubble`` and ``RedVictim``, all other classes can be implemented similarly and are available in the source code.

.. code-block:: python

    class Rubble(grid_object.GridObj):
        object_id = "rubble"
        color = Colors.Brown
        char = "X"

        def see_behind(self) -> bool:
            """Rubble blocks vision"""
            return False

        def toggle(self, env: typing.EnvType, agent: typing.AgentType) -> bool:
            """Rubble can be toggled by an agent with PickAxe and is removed 
            from the grid if successfully toggled.
            """
            toggling_agent_has_pickaxe = (
                any([isinstance(obj, Pickaxe) for obj in agent.inventory])
            )

            if toggling_agent_has_pickaxe:
                self._remove_from_grid(env.grid)

            return toggling_agent_has_pickaxe

        def render(self, tile_img):
            """Render the Rubble object as three circles in the tile."""
            c = COLORS[self.color]
            fill_coords(tile_img, point_in_circle(cx=0.25, cy=0.3, r=0.2), c)
            fill_coords(tile_img, point_in_circle(cx=0.75, cy=0.3, r=0.2), c)
            fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.7, r=0.2), c)


    # Register the grid object
    grid_object.register_object(Rubble.object_id, Rubble)


    class RedVictim(grid_object.GridObj):
        name = "red_victim"
        color = Colors.Red
        char = "R"

        def __init__(self, state=0):
            super().__init__(
                state=state,
            )
            self.toggle_countdown = 0
            self.first_toggle_agent_id: typing.AgentID = None

        def tick(self):
            """At each timestep, decrement toggle countdown and set the count as the state."""
            if self.toggle_countdown > 0:
                self.toggle_countdown -= 1
            self.state = self.toggle_countdown

        def toggle(self, env, agent) -> bool:
            """A RedVictim can be rescued if a Medic (or agent carrying MedKit) is the adjacent toggling agent
            and then another agent toggles within 30 timesteps."""

            if self.toggle_countdown == 0:

                toggling_agent_has_medkit = any(
                    [isinstance(obj, MedKit) for obj in agent.inventory]
                )

                if toggling_agent_has_medkit:
                    self.first_toggle_agent = agent.agent_id
                    self.toggle_countdown = 30

                return True

            if (
                self.toggle_countdown > 0
                and agent.agent_id != self.first_toggle_agent
            ):
                self._remove_from_grid(env.grid)
                return True

            return False

        def render(self, tile_img):
            """Render as a red circle."""
            c = COLORS[self.color]
            fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), c)


    grid_object.register_object(RedVictim.object_id, RedVictim)



``CoGridEnv`` Class
^^^^^^^^^^^^^^^^^^^^

The environment class requires that we implement the method for determining when an episode is complete. In our setting,
it's when no victims remain. For this, we just implement the ``get_terminateds_truncateds`` method, which sets the ``terminated``
attribute for all agents. The `super()` call will determine if we're truncating according to the ``max_steps`` passed to the environment ``config``.

.. code-block:: python

    class SearchRescueEnv(cogrid_env.CoGridEnv):
        def get_terminateds_truncateds(self) -> tuple[dict[typing.AgentID, bool], dict[typing.AgentID, bool]]:
            """
            Set done only when all targets have been located.
            """
            green_targets_in_grid = any(
                [
                    isinstance(obj, search_rescue_grid_objects.GreenVictim)
                    for obj in self.grid.grid
                ]
            )
            yellow_targets_in_grid = any(
                [   
                    isinstance(obj, search_rescue_grid_objects.YellowVictim)
                    for obj in self.grid.grid
                ]
            )
            red_targets_in_grid = any(
                [
                    isinstance(obj, search_rescue_grid_objects.RedVictim)
                    for obj in self.grid.grid
                ]
            )

            all_targets_reached = (
                not green_targets_in_grid
                and not yellow_targets_in_grid
                and not red_targets_in_grid
            )

            if all_targets_reached:
                for agent in self.agents.values():
                    agent.terminated = True

            return super().get_terminateds_truncateds()


Features
---------

The necessary feature encoding for the Search & Rescue environment is the (encoded) partially observed view of the agents. This is already implemented in CoGrid, but we provide and walk through the implementation below. 
Rather than observing RGB images, we encode the view into an integer representation of the grid objects and corresponding object states. 

.. code-block:: python

    class FoVEncoding(feature.Feature):
        """The Field of View (FoV) encoding feature, which encodes the agent's partially observed view."""
        def __init__(self, view_len, **kwargs):
            super().__init__(
                low=0,
                high=np.inf,
                shape=(view_len, view_len, 3),
                name="fov_encoding",
                **kwargs
            )

        def generate(self, env, player_id, **kwargs):
            """Generate the FoV encoding for the agent."""
            # Generate a slice of the grid around the agent
            agent_grid, _ = env.gen_obs_grid(agent_id=player_id)

            # Encode that slice as a 3D array of integers
            encoded_agent_grid = agent_grid.encode(encode_char=False)

            return encoded_agent_grid

Rewards
--------

Next, we define a reward function via a ``Reward`` class. This will calculate the number of victims saved
and provide a common reward to all agents according to their types. 

.. code-block:: python

    class RescueReward(reward.Reward):
        def __init__(self, agent_ids: list[str | int], **kwargs):
            super().__init__(
                name="rescue_reward", agent_ids=agent_ids, coefficient=1.0, **kwargs
            )

        def calculate_reward(
            self,
            state: Grid,
            agent_actions: dict[typing.AgentID, typing.ActionType],
            new_state: Grid,
        ) -> dict[typing.AgentID, float]:
            """Calcaute the reward for delivering a soup dish.

            :param state: The previous state of the grid.
            :type state: Grid
            :param actions: Actions taken by each agent in the previous state of the grid.
            :type actions: dict[int  |  str, int  |  float]
            :param new_state: The new state of the grid.
            :type new_state: Grid
            """
            prev_num_green = state.get_obj_count(
                search_rescue_grid_objects.GreenVictim
            )
            prev_num_yellow = state.get_obj_count(
                search_rescue_grid_objects.YellowVictim
            )
            prev_num_red = state.get_obj_count(
                search_rescue_grid_objects.RedVictim
            )

            new_num_green = new_state.get_obj_count(
                search_rescue_grid_objects.GreenVictim
            )
            new_num_yellow = new_state.get_obj_count(
                search_rescue_grid_objects.YellowVictim
            )
            new_num_red = new_state.get_obj_count(
                search_rescue_grid_objects.RedVictim
            )

            green_reward = prev_num_green - new_num_green
            yellow_reward = (prev_num_yellow - new_num_yellow) * 2
            red_reward = (prev_num_red - new_num_red) * 3

            reward_dict = {
                agent_id: green_reward + yellow_reward + red_reward
                for agent_id in self.agent_ids
            }

            return reward_dict


Environment Layout
-------------------

Finally, we define the layout of the environment. This includes the grid size, the grid objects, the agents, and the features.
We can do this using the grid object ``char`` attributes in a list of strings. In this example, ``#`` denotes a ``Wall``, ``S`` a spawn position,
``G`` a ``GreenVictim``, ``Y`` a ``YellowVictim``, ``R`` a ``RedVictim``, ``P`` a ``PickAxe``, ``M`` a ``MedKit``, ``K`` a ``Key``, and ``D`` a ``Door``.

.. code-block:: python

    layout = [
        "##########",
        "#SS      #",
        "#        #",
        "#        #",
        "#       G#",
        "#        #",
        "#P     K #",
        "#XX M ##D#",
        "#GX Y #GG#",
        "##########",
    ]

    states = np.zeros((10, 10), dtype=int)

We must also specify the states of all objects as a NumPy array, which defaults to 0 if unspecified. 

Environment Registration
-------------------------


We can now register an example of the ``SearchRescueEnv`` environment with the rewards and layout:

.. code-block:: python

    search_rescue_config = {
        "num_agents": 2,
        "features": ["fov_encoding"],
        "rewards": ["rescue_reward"],
        "grid_gen_kwargs": {"load": (layout, states)},
        "max_steps": 1000,
    }


    registry.register(
        "Search-Rescue-Example-V0",
        functools.partial(SearchRescueEnv, config=search_rescue_config),
    )


Finally, to use the environment, we can call it as follows:

.. code-block:: python

    env = registry.make("Search-Rescue-Example-V0")

    # Reset the environment
    obs = env.reset()

    terminateds = truncateds = {"__all__": False}
    while not terminateds["__all__"] and not truncateds["__all__"]:
        # Take a step in the environment
        obs, reward, terminateds, truncateds, infos = env.step(
            {agent_id: env.action_space.sample() for agent_id in env.agents}
        )
