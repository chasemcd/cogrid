"""Collision resolution edge-case tests for the unified move_agents() function.

Tests verify the vectorized pairwise conflict detection, priority masking, swap
detection, and cascade blocking in cogrid.core.movement.move_agents().

All tests use the numpy backend with direct array construction (no env needed).
"""

import numpy as np
import pytest

from cogrid.backend import set_backend

set_backend("numpy")

from cogrid.core.movement import move_agents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wall_map(H, W):
    """Create an HxW wall map with walls on the border only."""
    wm = np.zeros((H, W), dtype=np.int32)
    wm[0, :] = 1
    wm[-1, :] = 1
    wm[:, 0] = 1
    wm[:, -1] = 1
    return wm


def _empty_otm(H, W):
    return np.zeros((H, W), dtype=np.int32)


def _can_overlap(n_types=10):
    co = np.ones(n_types, dtype=np.int32)
    return co


# ---------------------------------------------------------------------------
# Test 1: Head-on collision
# ---------------------------------------------------------------------------


def test_head_on_collision():
    """Two agents moving toward the same cell from opposite sides.

    Agent 0 at (2,1) facing Right, Agent 1 at (2,3) facing Left.
    Both target (2,2). Priority determines winner.
    """
    H, W = 5, 5
    wall_map = _make_wall_map(H, W)
    otm = _empty_otm(H, W)
    can_overlap = _can_overlap()

    agent_pos = np.array([[2, 1], [2, 3]], dtype=np.int32)
    agent_dir = np.array([0, 2], dtype=np.int32)
    actions = np.array([3, 2], dtype=np.int32)

    # Priority [0, 1]: Agent 0 wins
    priority = np.array([0, 1], dtype=np.int32)
    new_pos, _ = move_agents(
        agent_pos, agent_dir, actions, wall_map, otm, can_overlap,
        priority, "cardinal",
    )
    assert np.array_equal(new_pos[0], [2, 2])
    assert np.array_equal(new_pos[1], [2, 3])
    assert not np.array_equal(new_pos[0], new_pos[1])

    # Priority [1, 0]: Agent 1 wins
    priority = np.array([1, 0], dtype=np.int32)
    new_pos, _ = move_agents(
        agent_pos, agent_dir, actions, wall_map, otm, can_overlap,
        priority, "cardinal",
    )
    assert np.array_equal(new_pos[1], [2, 2])
    assert np.array_equal(new_pos[0], [2, 1])


# ---------------------------------------------------------------------------
# Test 2: Swap detection
# ---------------------------------------------------------------------------


def test_swap_detection():
    """Two adjacent agents trying to swap positions are both reverted."""
    H, W = 5, 5
    wall_map = _make_wall_map(H, W)
    otm = _empty_otm(H, W)
    can_overlap = _can_overlap()

    agent_pos = np.array([[2, 2], [2, 3]], dtype=np.int32)
    agent_dir = np.array([0, 2], dtype=np.int32)
    actions = np.array([3, 2], dtype=np.int32)
    priority = np.array([0, 1], dtype=np.int32)

    new_pos, _ = move_agents(
        agent_pos, agent_dir, actions, wall_map, otm, can_overlap,
        priority, "cardinal",
    )

    assert np.array_equal(new_pos[0], [2, 2])
    assert np.array_equal(new_pos[1], [2, 3])


# ---------------------------------------------------------------------------
# Test 3: Moving into a staying agent
# ---------------------------------------------------------------------------


def test_into_staying_agent():
    """Agent tries to move into a non-moving agent's cell -- blocked."""
    H, W = 5, 5
    wall_map = _make_wall_map(H, W)
    otm = _empty_otm(H, W)
    can_overlap = _can_overlap()

    agent_pos = np.array([[2, 2], [2, 3]], dtype=np.int32)
    agent_dir = np.array([0, 0], dtype=np.int32)
    actions = np.array([3, 6], dtype=np.int32)  # MoveRight, Noop
    priority = np.array([0, 1], dtype=np.int32)

    new_pos, _ = move_agents(
        agent_pos, agent_dir, actions, wall_map, otm, can_overlap,
        priority, "cardinal",
    )

    assert np.array_equal(new_pos[0], [2, 2])
    assert np.array_equal(new_pos[1], [2, 3])


# ---------------------------------------------------------------------------
# Test 4: No conflict, independent movement
# ---------------------------------------------------------------------------


def test_no_conflict_independent_movement():
    """Two agents moving to different cells both succeed."""
    H, W = 5, 5
    wall_map = _make_wall_map(H, W)
    otm = _empty_otm(H, W)
    can_overlap = _can_overlap()

    agent_pos = np.array([[1, 1], [3, 3]], dtype=np.int32)
    agent_dir = np.array([0, 2], dtype=np.int32)
    actions = np.array([3, 2], dtype=np.int32)
    priority = np.array([0, 1], dtype=np.int32)

    new_pos, _ = move_agents(
        agent_pos, agent_dir, actions, wall_map, otm, can_overlap,
        priority, "cardinal",
    )

    assert np.array_equal(new_pos[0], [1, 2])
    assert np.array_equal(new_pos[1], [3, 2])


# ---------------------------------------------------------------------------
# Test 5: Wall blocked
# ---------------------------------------------------------------------------


def test_wall_blocked():
    """Agent tries to move into a wall and stays in place."""
    H, W = 5, 5
    wall_map = _make_wall_map(H, W)
    otm = _empty_otm(H, W)
    can_overlap = _can_overlap()

    agent_pos = np.array([[1, 1], [3, 3]], dtype=np.int32)
    agent_dir = np.array([3, 0], dtype=np.int32)  # Up, Right
    actions = np.array([0, 6], dtype=np.int32)  # MoveUp, Noop
    priority = np.array([0, 1], dtype=np.int32)

    new_pos, _ = move_agents(
        agent_pos, agent_dir, actions, wall_map, otm, can_overlap,
        priority, "cardinal",
    )

    assert np.array_equal(new_pos[0], [1, 1])


# ---------------------------------------------------------------------------
# Test 6: No-overlap invariant (100 random steps)
# ---------------------------------------------------------------------------


def test_no_overlap_invariant():
    """Run 100 random steps and verify no two agents ever overlap."""
    H, W = 7, 7
    wall_map = _make_wall_map(H, W)
    otm = _empty_otm(H, W)
    can_overlap = _can_overlap()

    rng = np.random.default_rng(42)
    agent_pos = np.array([[2, 2], [4, 4]], dtype=np.int32)
    agent_dir = np.array([0, 2], dtype=np.int32)

    for step in range(100):
        actions = rng.integers(0, 7, size=2).astype(np.int32)
        priority = rng.permutation(2).astype(np.int32)

        new_pos, new_dir = move_agents(
            agent_pos, agent_dir, actions, wall_map, otm, can_overlap,
            priority, "cardinal",
        )

        if np.array_equal(new_pos[0], new_pos[1]):
            pytest.fail(
                f"Step {step}: agents overlap at {new_pos[0]} "
                f"(actions={actions}, priority={priority})"
            )

        agent_pos = new_pos
        agent_dir = new_dir


# ---------------------------------------------------------------------------
# Test 7: Priority determines winner
# ---------------------------------------------------------------------------


def test_priority_determines_winner():
    """Verify the agent first in priority array wins contested cell."""
    H, W = 5, 5
    wall_map = _make_wall_map(H, W)
    otm = _empty_otm(H, W)
    can_overlap = _can_overlap()

    agent_pos = np.array([[2, 1], [2, 3]], dtype=np.int32)
    agent_dir = np.array([0, 2], dtype=np.int32)
    actions = np.array([3, 2], dtype=np.int32)

    # Agent 0 higher priority
    priority = np.array([0, 1], dtype=np.int32)
    new_pos, _ = move_agents(
        agent_pos, agent_dir, actions, wall_map, otm, can_overlap,
        priority, "cardinal",
    )
    assert np.array_equal(new_pos[0], [2, 2])
    assert np.array_equal(new_pos[1], [2, 3])

    # Agent 1 higher priority
    priority = np.array([1, 0], dtype=np.int32)
    new_pos, _ = move_agents(
        agent_pos, agent_dir, actions, wall_map, otm, can_overlap,
        priority, "cardinal",
    )
    assert np.array_equal(new_pos[1], [2, 2])
    assert np.array_equal(new_pos[0], [2, 1])
