# External Integrations

**Analysis Date:** 2026-01-19

## APIs & External Services

**None detected.**

This is a standalone Python library for multi-agent reinforcement learning environments. It does not integrate with external APIs, cloud services, or third-party web services.

## Data Storage

**Databases:**
- None - All state is in-memory

**File Storage:**
- Local filesystem only
- No persistent data storage

**Caching:**
- In-memory tile rendering cache: `Grid.tile_cache` in `cogrid/core/grid.py`

## Authentication & Identity

**Auth Provider:**
- None - Standalone library without authentication

## Monitoring & Observability

**Error Tracking:**
- None

**Logs:**
- Print statements for debugging (e.g., registry overwrites)
- No formal logging framework

## CI/CD & Deployment

**Hosting:**
- PyPI (`pip install cogrid`)
- GitHub repository: https://github.com/chasemcd/cogrid

**CI Pipeline:**
- None detected (no `.github/` directory)

**Documentation Hosting:**
- ReadTheDocs (configured in `.readthedocs.yaml`)
- Build config: `docs/conf.py`

## Environment Configuration

**Required env vars:**
- None

**Secrets location:**
- Not applicable

**Runtime Configuration:**
- All configuration via Python dictionaries passed to environment constructors
- Example config structure in `cogrid/cogrid_env.py`:
  ```python
  config = {
      "name": "environment_name",
      "max_steps": 100,
      "num_agents": 2,
      "action_set": "cardinal_actions",  # or "rotation_actions"
      "features": ["feature_name"],
      "rewards": ["reward_name"],
      "grid": {
          "layout": "layout_name",
          # or "layout_fn": callable
      }
  }
  ```

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Third-Party Library Integrations

**PettingZoo:**
- Location: `cogrid/cogrid_env.py`
- `CoGridEnv` inherits from `pettingzoo.ParallelEnv`
- Implements parallel multi-agent API: `reset()`, `step()`, `render()`, `close()`
- Type definitions from `pettingzoo.utils.env` in `cogrid/core/typing.py`

**Gymnasium:**
- Location: `cogrid/cogrid_env.py`, `cogrid/feature_space/feature_space.py`
- Uses `gymnasium.spaces.Discrete` for action spaces
- Uses `gymnasium.spaces.Dict` for observation spaces

**PyGame (Optional):**
- Location: `cogrid/cogrid_env.py`, `cogrid/run_interactive.py`
- Conditionally imported for human-mode rendering
- Handles keyboard input for interactive play
- Error raised if not installed when `render_mode="human"`

**OpenCV (Optional):**
- Location: `cogrid/visualization/rendering.py`
- Conditionally imported for `add_text_to_image()` function
- Used for text overlay on rendered frames

**ONNX Runtime (Commented Out):**
- Location: `cogrid/run_interactive.py`
- Intended for loading trained RL policies
- Currently disabled (commented import)

## Registry Patterns

The codebase uses internal registries to manage extensibility:

**Environment Registry:**
- Location: `cogrid/envs/registry.py`
- `registry.register(env_id, env_class)` to add environments
- `registry.make(env_id, **kwargs)` to instantiate

**Reward Registry:**
- Location: `cogrid/core/reward.py`
- `register_reward(reward_id, reward_class)` to add rewards
- `make_reward(reward_id, **kwargs)` to instantiate

**Feature Registry:**
- Location: `cogrid/feature_space/feature_space.py`
- `register_feature(feature_id, feature_class)` to add features
- `make_feature_generator(feature_id, **kwargs)` to instantiate

**Object Registry:**
- Location: `cogrid/core/grid_object.py`
- `register_object(object_id, obj_class, scope)` to add grid objects
- `make_object(object_id, scope, **kwargs)` to instantiate
- Scoped registries: `"global"`, `"overcooked"`, `"search_rescue"`

---

*Integration audit: 2026-01-19*
