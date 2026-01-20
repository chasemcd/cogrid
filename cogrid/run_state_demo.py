"""Interactive demo for state serialization.

Controls:
- Arrow keys: Move agent
- Q: Toggle (interact with objects)
- W: Pickup/Drop
- Space: Noop
- Enter: Save state, destroy env, restore from saved state
- Backspace: Reset environment
- Escape: Quit
"""
from __future__ import annotations

import json

from cogrid.core.actions import Actions
from cogrid.cogrid_env import CoGridEnv
from cogrid.envs import registry
from cogrid.core import typing

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None
    raise ImportError("Must `pip install pygame` to use interactive visualizer!")


KEY_TO_ACTION = {
    "left": Actions.MoveLeft,
    "right": Actions.MoveRight,
    "up": Actions.MoveUp,
    "down": Actions.MoveDown,
    "q": Actions.Toggle,
    "w": Actions.PickupDrop,
    "space": Actions.Noop,
}


class StateDemoPlay:
    def __init__(
        self,
        env_creator: callable,
        human_agent_id: typing.AgentID = None,
        seed: int = None,
    ) -> None:
        self.env_creator = env_creator
        self.env = env_creator()
        self.seed = seed
        self.closed = False
        self.human_agent_id = human_agent_id
        self.obs = None
        self.cumulative_reward = 0
        self.saved_state = None

    def run(self):
        self.reset(self.seed)
        while not self.closed:
            actions = {agent_id: Actions.Noop for agent_id in self.env.agent_ids}

            for a_id, obs in self.obs.items():
                if a_id == self.human_agent_id:
                    continue
                actions[a_id] = self.env.action_spaces[a_id].sample()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    return
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))

                    if event.key == "escape":
                        self.env.close()
                        return

                    if event.key == "backspace":
                        self.reset(self.seed)
                        continue

                    if event.key == "return":
                        self.save_destroy_restore()
                        continue

                    if (
                        self.human_agent_id is not None
                        and event.key in KEY_TO_ACTION.keys()
                    ):
                        actions[self.human_agent_id] = KEY_TO_ACTION[event.key]
                    else:
                        print(f"Invalid action: {event.key}")

            self.step(actions)

    def save_destroy_restore(self):
        """Save state, destroy environment, create new one, restore state."""
        print("\n" + "=" * 60)
        print("SAVE STATE")
        print("=" * 60)

        # 1. Get current state
        state = self.env.get_state()
        state_json = json.dumps(state, indent=2)
        print(f"State captured at timestep {state['timestep']}")
        print(f"State size: {len(state_json)} bytes")
        print(f"Agents: {list(state['agents'].keys())}")

        # Show agent positions and inventories
        for agent_id, agent_state in state["agents"].items():
            inv_str = (
                ", ".join(obj["object_id"] for obj in agent_state["inventory"])
                if agent_state["inventory"]
                else "empty"
            )
            print(f"  {agent_id}: pos={agent_state['pos']}, inventory=[{inv_str}]")

        # 2. Destroy environment
        print("\n" + "-" * 60)
        print("DESTROYING ENVIRONMENT")
        print("-" * 60)
        self.env.close()
        del self.env
        print("Environment destroyed.")

        # 3. Create new environment
        print("\n" + "-" * 60)
        print("CREATING NEW ENVIRONMENT")
        print("-" * 60)
        self.env = self.env_creator()
        self.env.reset(seed=self.seed)
        print("New environment created and reset.")

        # 4. Restore state
        print("\n" + "-" * 60)
        print("RESTORING STATE")
        print("-" * 60)
        self.env.set_state(state)
        print(f"State restored to timestep {self.env.t}")

        # 5. Verify
        restored_state = self.env.get_state()
        if state == restored_state:
            print("Verification: States match exactly!")
        else:
            print("WARNING: States differ after restore!")

        print("=" * 60 + "\n")

        # Render restored state
        self.env.render()

    def step(self, actions: dict[str, Actions]):
        self.obs, rewards, terminateds, truncateds, _ = self.env.step(actions)
        self.cumulative_reward += list(rewards.values())[0]

        if not self.env.agents:
            print("All agents done!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed):
        self.obs, _ = self.env.reset(seed=seed)
        self.cumulative_reward = 0
        self.env.render()
        print(f"Environment reset (seed={seed})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-id",
        type=str,
        help="environment to load",
        default="Overcooked-CrampedRoom-V0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed",
        default=42,
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default=512,
        help="pygame window size",
    )
    parser.add_argument(
        "--human-agent",
        type=int,
        default=1,
        help="which agent to control (0 or 1)",
    )

    args = parser.parse_args()

    def env_creator() -> CoGridEnv:
        return registry.make(
            args.env_id,
            highlight=False,
            render_mode="human",
            screen_size=args.screen_size,
        )

    print(__doc__)

    demo = StateDemoPlay(
        env_creator,
        human_agent_id=args.human_agent,
        seed=args.seed,
    )
    demo.run()
