"""
Integration tests for state serialization across CoGrid environments.

These tests verify extended determinism (50+ step sequences) and cross-environment
validation for the get_state/set_state serialization system.
"""

import copy

import numpy as np
import pytest

from cogrid.core.actions import Actions
from cogrid.envs import registry


class TestDeterminismExtended:
    """Extended determinism tests with longer action sequences."""

    def test_overcooked_extended_determinism(self):
        """Test 50-step determinism for Overcooked environment."""
        env1 = registry.make("Overcooked-CrampedRoom-V0")
        env1.reset(seed=42)

        # Run 50 steps to reach interesting state
        actions_list = [{0: Actions.MoveRight, 1: Actions.MoveLeft}] * 25
        actions_list += [{0: Actions.MoveUp, 1: Actions.MoveDown}] * 25
        for actions in actions_list:
            env1.step(actions)

        # Save checkpoint
        state = env1.get_state()

        # Continue 50 more steps, recording results
        test_actions = [{0: Actions.MoveDown, 1: Actions.MoveUp}] * 50
        env1_results = []
        for actions in test_actions:
            obs, rewards, terminated, truncated, info = env1.step(actions)
            env1_results.append(
                {
                    "rewards": copy.deepcopy(rewards),
                    "obs": {k: v.copy() for k, v in obs.items()},
                    "terminated": copy.deepcopy(terminated),
                    "truncated": copy.deepcopy(truncated),
                }
            )

        # Restore and run same sequence
        env2 = registry.make("Overcooked-CrampedRoom-V0")
        env2.set_state(state)

        for i, actions in enumerate(test_actions):
            obs, rewards, terminated, truncated, info = env2.step(actions)
            # Verify rewards match
            for agent_id in rewards:
                assert rewards[agent_id] == env1_results[i]["rewards"][agent_id], (
                    f"Reward mismatch at step {i} for agent {agent_id}"
                )
            # Verify termination/truncation flags match
            for agent_id in terminated:
                assert (
                    terminated[agent_id] == env1_results[i]["terminated"][agent_id]
                ), f"Terminated mismatch at step {i} for agent {agent_id}"
            for agent_id in truncated:
                assert truncated[agent_id] == env1_results[i]["truncated"][agent_id], (
                    f"Truncated mismatch at step {i} for agent {agent_id}"
                )

    def test_search_rescue_extended_determinism(self):
        """Test 50-step determinism for Search & Rescue environment."""
        env1 = registry.make("SearchRescue-Test-V0")
        env1.reset(seed=42)

        # Run 50 steps to reach interesting state
        actions_list = [{0: Actions.MoveRight, 1: Actions.MoveLeft}] * 25
        actions_list += [{0: Actions.MoveUp, 1: Actions.MoveDown}] * 25
        for actions in actions_list:
            env1.step(actions)

        # Save checkpoint
        state = env1.get_state()

        # Continue 50 more steps, recording results
        test_actions = [{0: Actions.MoveDown, 1: Actions.MoveUp}] * 50
        env1_results = []
        for actions in test_actions:
            obs, rewards, terminated, truncated, info = env1.step(actions)
            env1_results.append(
                {
                    "rewards": copy.deepcopy(rewards),
                    "obs": {k: v.copy() for k, v in obs.items()},
                    "terminated": copy.deepcopy(terminated),
                    "truncated": copy.deepcopy(truncated),
                }
            )

        # Restore and run same sequence
        env2 = registry.make("SearchRescue-Test-V0")
        env2.set_state(state)

        for i, actions in enumerate(test_actions):
            obs, rewards, terminated, truncated, info = env2.step(actions)
            # Verify rewards match
            for agent_id in rewards:
                assert rewards[agent_id] == env1_results[i]["rewards"][agent_id], (
                    f"Reward mismatch at step {i} for agent {agent_id}"
                )
            # Verify termination/truncation flags match
            for agent_id in terminated:
                assert (
                    terminated[agent_id] == env1_results[i]["terminated"][agent_id]
                ), f"Terminated mismatch at step {i} for agent {agent_id}"
            for agent_id in truncated:
                assert truncated[agent_id] == env1_results[i]["truncated"][agent_id], (
                    f"Truncated mismatch at step {i} for agent {agent_id}"
                )


class TestObservationSpaceMatch:
    """Tests for observation array value matching after restore."""

    def test_observation_arrays_match_after_restore(self):
        """Verify actual observation array values match, not just shapes."""
        env1 = registry.make("Overcooked-CrampedRoom-V0")
        env1.reset(seed=42)

        # Run some steps
        for _ in range(10):
            env1.step({0: Actions.MoveRight, 1: Actions.MoveLeft})

        # Save state
        state = env1.get_state()

        # Continue and record observations
        # obs[agent_id] is a dict like {'overcooked_features': array([...])}
        test_actions = [{0: Actions.MoveUp, 1: Actions.MoveDown}] * 20
        env1_observations = []
        for actions in test_actions:
            obs, _, _, _, _ = env1.step(actions)
            # Deep copy each agent's observation dict
            env1_observations.append(
                {
                    agent_id: {k: v.copy() for k, v in agent_obs.items()}
                    for agent_id, agent_obs in obs.items()
                }
            )

        # Restore and verify observations match exactly
        env2 = registry.make("Overcooked-CrampedRoom-V0")
        env2.set_state(state)

        for i, actions in enumerate(test_actions):
            obs, _, _, _, _ = env2.step(actions)
            for agent_id in obs:
                for feature_name in obs[agent_id]:
                    np.testing.assert_array_equal(
                        obs[agent_id][feature_name],
                        env1_observations[i][agent_id][feature_name],
                        err_msg=f"Observation mismatch at step {i} for agent {agent_id}, feature {feature_name}",
                    )
