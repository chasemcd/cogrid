"""Determinism regression tests for the numpy backend.

Movement priority and initial agent directions must be derived from the
state's RNG key, not from a fresh unseeded Generator. Two runs with the
same seed and the same action sequence must produce identical rollouts.
"""

import numpy as np
import pytest

from cogrid.backend import set_backend
from cogrid.backend._dispatch import _reset_backend_for_testing


@pytest.fixture(autouse=True)
def _numpy_backend():
    _reset_backend_for_testing()
    set_backend("numpy")


def _rollout(seed, n_steps=25):
    """Reset with ``seed``, replay a fixed action sequence, record the trace."""
    from cogrid.envs import registry

    env = registry.make("Overcooked-CrampedRoom-V0")
    obs, _ = env.reset(seed=seed)

    agent_ids = list(obs.keys())
    n_actions = env.action_space(agent_ids[0]).n
    trace = [{aid: np.asarray(obs[aid]).copy() for aid in agent_ids}]
    positions = [np.asarray(env._env_state.agent_pos).copy()]

    for t in range(n_steps):
        # Fixed, agent-dependent action sequence -- no RNG involved.
        actions = {aid: (t + i) % n_actions for i, aid in enumerate(agent_ids)}
        obs, rewards, terminateds, truncateds, _ = env.step(actions)
        trace.append({aid: np.asarray(obs[aid]).copy() for aid in agent_ids})
        positions.append(np.asarray(env._env_state.agent_pos).copy())
        if all(terminateds.values()) or all(truncateds.values()):
            break

    return trace, positions


def test_same_seed_same_rollout():
    """Identical seed + identical actions produce identical observations."""
    trace_a, pos_a = _rollout(seed=7)
    trace_b, pos_b = _rollout(seed=7)

    assert len(trace_a) == len(trace_b)
    for t, (step_a, step_b) in enumerate(zip(trace_a, trace_b)):
        assert step_a.keys() == step_b.keys()
        for aid in step_a:
            np.testing.assert_array_equal(
                step_a[aid], step_b[aid], err_msg=f"obs diverged at t={t} for {aid}"
            )
    for t, (p_a, p_b) in enumerate(zip(pos_a, pos_b)):
        np.testing.assert_array_equal(p_a, p_b, err_msg=f"positions diverged at t={t}")


def test_snapshot_resumes_the_original_rng_stream():
    """A snapshot restored into a fresh env continues the original trajectory.

    ``step()`` draws each step's RNG key from ``self.np_random``, so the
    snapshot must carry that generator's state, not just ``rng_key``.
    """
    import json

    from cogrid.envs import registry

    def _continue(env, agent_ids, n_actions, start, stop):
        trace = []
        for t in range(start, stop):
            env.step({aid: (t + i) % n_actions for i, aid in enumerate(agent_ids)})
            trace.append(np.asarray(env._env_state.agent_pos).copy())
        return np.stack(trace)

    env = registry.make("Overcooked-CrampedRoom-V0")
    obs, _ = env.reset(seed=7)
    agent_ids = list(obs.keys())
    n_actions = env.action_space(agent_ids[0]).n

    _continue(env, agent_ids, n_actions, 0, 5)
    # Round-trip through JSON -- the snapshot is documented as serializable.
    snapshot = json.loads(json.dumps(env.get_state()))
    original = _continue(env, agent_ids, n_actions, 5, 20)

    # Fresh env on a different seed, i.e. a different np_random stream.
    restored_env = registry.make("Overcooked-CrampedRoom-V0")
    restored_env.reset(seed=999)
    restored_env.set_state(snapshot)
    restored = _continue(restored_env, agent_ids, n_actions, 5, 20)

    np.testing.assert_array_equal(original, restored)


def test_set_state_tolerates_snapshot_without_rng_stream():
    """Snapshots predating ``np_random_state`` still restore without error."""
    from cogrid.envs import registry

    env = registry.make("Overcooked-CrampedRoom-V0")
    obs, _ = env.reset(seed=7)
    agent_ids = list(obs.keys())
    n_actions = env.action_space(agent_ids[0]).n
    env.step({aid: 0 for aid in agent_ids})

    snapshot = env.get_state()
    del snapshot["np_random_state"]

    env.set_state(snapshot)
    env.step({aid: (1 + i) % n_actions for i, aid in enumerate(agent_ids)})


def test_movement_priority_is_not_constant():
    """Priority must vary across steps, not lock one agent in front forever.

    Guards against the degenerate fix where the RNG key is never advanced
    and every step re-derives the same permutation from the same seed.
    """
    from cogrid.core.pipeline.step import _backend_rng

    key = 42
    perms = []
    for _ in range(20):
        key, priority = _backend_rng(key, "permutation", 2)
        assert key is not None
        perms.append(tuple(int(x) for x in priority))

    assert len(set(perms)) > 1, f"priority never changed across steps: {perms[:5]}"


def test_directions_rng_returns_live_key():
    """The 'directions' mode must return an advanced key, not None."""
    from cogrid.core.pipeline.step import _backend_rng

    key, dirs = _backend_rng(42, "directions", 2)
    assert key is not None, "reset must leave a usable rng_key on the state"
    assert key != 42
    assert dirs.shape == (2,)
    assert np.all((dirs >= 0) & (dirs < 4))

    # Same input key reproduces the same directions.
    key2, dirs2 = _backend_rng(42, "directions", 2)
    assert key2 == key
    np.testing.assert_array_equal(dirs, dirs2)
