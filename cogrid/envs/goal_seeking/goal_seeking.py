"""Goal-seeking grid environment."""

import operator

from cogrid.cogrid_env import CoGridEnv
from cogrid.envs.goal_seeking.agent import GoalSeekingAgent


class GoalSeeking(CoGridEnv):
    """Goal Seeking GridWorld task. A remake with multi-agent capability. For reference, see:.

    https://www.cmu.edu/dietrich/sds/ddmlab/papers/2020ICCM_Ngoc-CameraReady.pdf
    https://www.cmu.edu/dietrich/sds/ddmlab/papers/NguyenGonzalez2020.pdf
    """

    def __init__(self, grid_path, config):
        """Initialize the goal-seeking environment from config."""
        super().__init__(grid_path=grid_path, agent_class=GoalSeekingAgent, config=config)

        self.target_values = self.grid_data["values"]
        self.optimal_path_length = self.grid_data["optimal_path_length"]

        self.ma_spawns = (
            [tuple(pos) for pos in self.grid_data["ma_spawns"]]
            if config["num_agents"] > 1
            else None
        )

        self.targets = {}
        for row in range(self.base_grid.shape[0]):
            for col in range(self.base_grid.shape[1]):
                if self.base_grid[row, col] in self.target_values.keys():
                    self.targets[self.base_grid[row, col]] = (row, col)

        self.pref_target_loc = self.targets[
            max(self.target_values.items(), key=operator.itemgetter(1))[0]
        ]

        self.setup_agents()

    def select_spawn_point(self, random_spawn=True) -> tuple:
        """Select an available spawn point for a new agent."""
        curr_pos = [agent.pos for agent in self.env_agents.values()]

        if (
            "gen_random_spawn" in self.config["env"].keys()
            and self.config["env"]["gen_random_spawn"]
        ):
            self.np_random.shuffle(self.free_spaces)
            # Random spawn excludes pre-defined spawn points so that
            # random positions are distinct from map-specified ones
            # (required for SoU transfer experiments).
            available = [
                sp for sp in self.free_spaces if sp not in curr_pos and sp not in self.spawns
            ]
            return available[0]

        if random_spawn:
            self.np_random.shuffle(self.spawns)

        if self.config["num_agents"] > 1:
            all_spawns = self.ma_spawns
        else:
            all_spawns = self.spawns

        selected_spawn = None
        for spawn in all_spawns:
            if spawn not in curr_pos:
                selected_spawn = spawn
                break
        assert selected_spawn is not None, (
            "There are not enough spawn points in the map for the specified number of agents."
        )
        return selected_spawn

    def custom_reset(self):
        """Reset goal positions and broadcast agent positions."""
        self.add_goals()
        for agent in self.env_agents.values():
            agent.all_agent_pos = self.agent_pos

    def add_goals(self):
        """Place goal targets on the world map."""
        for target, pos in self.targets.items():
            self.world_map[pos] = target
