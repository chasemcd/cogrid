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


class TestCrossEnvironmentValidation:
    """Tests for serialization across different registered environments."""

    @pytest.mark.parametrize(
        "env_name",
        [
            "Overcooked-CrampedRoom-V0",
            "SearchRescue-Test-V0",
        ],
    )
    def test_environment_supports_serialization(self, env_name):
        """Verify environment supports get_state/set_state roundtrip."""
        env = registry.make(env_name)
        env.reset(seed=42)

        # Take some steps
        actions = {i: Actions.Noop for i in env.agent_ids}
        for _ in range(5):
            env.step(actions)

        # Record state before roundtrip
        original_t = env.t

        # Roundtrip
        state = env.get_state()
        env2 = registry.make(env_name)
        env2.set_state(state)

        # Verify timestep matches
        assert env.t == env2.t, f"Timestep mismatch: {env.t} != {env2.t}"
        assert env2.t == original_t, f"Timestep not preserved: {env2.t} != {original_t}"

        # Verify agent ids match
        assert set(env.agent_ids) == set(env2.agent_ids), "Agent IDs mismatch"

        # Verify agent positions match
        for agent_id in env.agent_ids:
            np.testing.assert_array_equal(
                env.env_agents[agent_id].pos,
                env2.env_agents[agent_id].pos,
                err_msg=f"Agent {agent_id} position mismatch",
            )


class TestEdgeCaseSerialization:
    """Tests for edge cases and boundary conditions in serialization."""

    def test_max_steps_boundary(self):
        """Test serialization at max_steps boundary triggers truncation correctly."""
        # Create env with short max_steps
        from cogrid.envs.overcooked import overcooked

        config = {
            "name": "overcooked",
            "num_agents": 2,
            "action_set": "cardinal_actions",
            "features": ["overcooked_features"],
            "rewards": ["delivery_reward"],
            "grid": {"layout": "overcooked_cramped_room_v0"},
            "max_steps": 10,
            "scope": "overcooked",
        }
        env1 = overcooked.Overcooked(config=config)
        env1.reset(seed=42)

        # Step to t=9 (one before max_steps)
        actions = {0: Actions.Noop, 1: Actions.Noop}
        for _ in range(9):
            obs, rewards, terminated, truncated, info = env1.step(actions)
            # Should not be truncated yet
            assert not any(truncated.values()), f"Premature truncation at step {env1.t}"

        assert env1.t == 9, f"Expected t=9, got t={env1.t}"

        # Save state at t=9
        state = env1.get_state()

        # Restore to new env
        env2 = overcooked.Overcooked(config=config)
        env2.set_state(state)
        assert env2.t == 9, f"Expected restored t=9, got t={env2.t}"

        # Next step should trigger truncation
        obs, rewards, terminated, truncated, info = env2.step(actions)
        assert env2.t == 10, f"Expected t=10, got t={env2.t}"
        assert all(truncated.values()), "Truncation should have been triggered at max_steps"

    def test_grid_state_preserved(self):
        """Test that grid state including floor cells is preserved."""
        env1 = registry.make("Overcooked-CrampedRoom-V0")
        env1.reset(seed=42)

        # Run some steps to change state
        for _ in range(10):
            env1.step({0: Actions.MoveRight, 1: Actions.MoveLeft})

        # Get grid encoding before save
        original_encoding = env1.grid.encode(encode_char=False, scope=env1.scope)

        # Save and restore
        state = env1.get_state()
        env2 = registry.make("Overcooked-CrampedRoom-V0")
        env2.set_state(state)

        # Get grid encoding after restore
        restored_encoding = env2.grid.encode(encode_char=False, scope=env2.scope)

        # Verify grid encodings match
        np.testing.assert_array_equal(
            original_encoding,
            restored_encoding,
            err_msg="Grid encoding mismatch after restore",
        )
