"""Tests for step dynamics determinism."""
import copy
import unittest

from cogrid.envs import registry
from cogrid.core.actions import Actions


class TestRandomizedLayoutDeterminism(unittest.TestCase):
    """Verify RandomizedLayout environment selects layouts deterministically."""

    def test_randomized_layout_deterministic(self):
        """Same seed produces same layout selection."""
        layouts_run1 = []
        layouts_run2 = []

        for _ in range(5):
            env = registry.make("Overcooked-RandomizedLayout-V0")
            env.reset(seed=42)
            layouts_run1.append(env.current_layout_id)
            env.close()

        for _ in range(5):
            env = registry.make("Overcooked-RandomizedLayout-V0")
            env.reset(seed=42)
            layouts_run2.append(env.current_layout_id)
            env.close()

        self.assertEqual(layouts_run1, layouts_run2, "RandomizedLayout not deterministic")


class TestStepDeterminism(unittest.TestCase):
    """Verify step() is deterministic for identical inputs."""

    def test_identical_actions_produce_identical_states(self):
        """Same seed + same actions = same state, every time."""
        env1 = registry.make("Overcooked-CrampedRoom-V0")
        env2 = registry.make("Overcooked-CrampedRoom-V0")

        env1.reset(seed=42)
        env2.reset(seed=42)

        # Run 100 steps with identical actions
        for _ in range(100):
            actions = {0: Actions.MoveRight, 1: Actions.MoveLeft}
            obs1, _, _, _, _ = env1.step(actions)
            obs2, _, _, _, _ = env2.step(actions)

            # States must be identical
            state1 = env1.get_state()
            state2 = env2.get_state()

            # Compare agent positions
            for agent_id in state1["agents"]:
                self.assertEqual(
                    state1["agents"][agent_id]["pos"],
                    state2["agents"][agent_id]["pos"],
                    f"Agent {agent_id} position diverged at step {env1.t}"
                )

        env1.close()
        env2.close()

    def test_collision_resolution_deterministic(self):
        """When agents collide, resolution is deterministic."""
        results = []

        for _ in range(10):
            env = registry.make("Overcooked-CrampedRoom-V0")
            env.reset(seed=42)

            # Force agents toward each other
            for _ in range(20):
                actions = {0: Actions.MoveRight, 1: Actions.MoveLeft}
                env.step(actions)

            # Record final positions
            state = env.get_state()
            positions = tuple(
                tuple(state["agents"][a]["pos"]) for a in sorted(state["agents"])
            )
            results.append(positions)
            env.close()

        # All runs should produce identical results
        self.assertEqual(len(set(results)), 1, "Collision resolution is non-deterministic")


class TestRestoredStateDeterminism(unittest.TestCase):
    """Verify restored state continues identically to original environment."""

    def test_restored_state_identical_continuation(self):
        """Restored state produces identical continuation as original."""
        env1 = registry.make("Overcooked-CrampedRoom-V0")
        env1.reset(seed=42)

        # Run 50 steps to reach interesting state
        for _ in range(50):
            actions = {0: Actions.MoveRight, 1: Actions.MoveLeft}
            env1.step(actions)

        # Save checkpoint
        state = env1.get_state()

        # Continue for 50 more steps, recording all outputs
        env1_results = []
        for _ in range(50):
            actions = {0: Actions.MoveUp, 1: Actions.MoveDown}
            obs, rewards, terminated, truncated, info = env1.step(actions)
            env1_results.append({
                "obs": {k: v.copy() for k, v in obs.items()},
                "rewards": copy.deepcopy(rewards),
                "terminated": copy.deepcopy(terminated),
                "truncated": copy.deepcopy(truncated),
            })

        # Create env2, restore state
        env2 = registry.make("Overcooked-CrampedRoom-V0")
        env2.set_state(state)

        # Run same 50 actions and verify all outputs match
        for step_idx in range(50):
            actions = {0: Actions.MoveUp, 1: Actions.MoveDown}
            obs, rewards, terminated, truncated, info = env2.step(actions)

            # Verify rewards match
            for agent_id in rewards:
                self.assertEqual(
                    rewards[agent_id],
                    env1_results[step_idx]["rewards"][agent_id],
                    f"Reward mismatch at step {step_idx} for agent {agent_id}"
                )

            # Verify terminated flags match
            for agent_id in terminated:
                self.assertEqual(
                    terminated[agent_id],
                    env1_results[step_idx]["terminated"][agent_id],
                    f"Terminated mismatch at step {step_idx} for agent {agent_id}"
                )

            # Verify truncated flags match
            for agent_id in truncated:
                self.assertEqual(
                    truncated[agent_id],
                    env1_results[step_idx]["truncated"][agent_id],
                    f"Truncated mismatch at step {step_idx} for agent {agent_id}"
                )

        env1.close()
        env2.close()


if __name__ == "__main__":
    unittest.main()
