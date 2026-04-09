# Interactions

When an agent performs `PickupDrop` or `Toggle`, the step pipeline runs a list of interaction functions — called **branches** — to determine what happens. Each branch independently decides whether it should fire and what state changes to apply.

## Step Pipeline Context

Interactions run after tick functions and movement:

**tick → movement → interactions → observations → rewards**

`PickupDrop` ("pick up / put down") and `Toggle` ("activate / use") are semantically different actions. Built-in branches for pickup, drop, and place only fire on `PickupDrop`. Custom branches can check `ctx.action` against `ctx.action_id` to distinguish which action the agent chose.

## Branch Function Signature

Every interaction function has the same interface:

```python
def my_branch(ctx: InteractionContext) -> tuple[bool, dict]:
    should_apply = ...  # bool: should this interaction fire?
    changes = {...}     # dict: array name -> updated array
    return should_apply, changes
```

- `should_apply` — boolean. If false, `changes` is ignored.
- `changes` — dict mapping state array names (e.g. `"object_type_map"`, `"agent_inv"`) to their new values.

The pipeline applies changes via `xp.where`: for each key in `changes`, the new value replaces the old one where `should_apply` is true. When multiple branches fire for the same agent, later branches overwrite earlier ones for overlapping keys.

## InteractionContext

The pipeline builds a frozen `InteractionContext` dataclass before calling branches. Each agent gets its own context per step.

??? api "`InteractionContext`"

    ::: cogrid.core.pipeline.context.InteractionContext
        options:
          heading_level: 4
          members: false

### Standard fields

| Field | Type | Description |
|-------|------|-------------|
| `can_interact` | bool | `True` if this agent performed `PickupDrop` or `Toggle` and no other agent blocks the cell ahead. |
| `action` | int | Raw action index this agent chose this step. |
| `action_id` | `ActionID` | Named indices for all actions. Use `ctx.action_id.pickup_drop`, `ctx.action_id.toggle`, etc. Actions not in the action set have index `-1`. |
| `facing_row` | int | Row of the cell the agent is facing. |
| `facing_col` | int | Column of the cell the agent is facing. |
| `facing_type` | int | Type ID of the object in the faced cell (0 = empty). |
| `agent_index` | int | Which agent (0, 1, ...) is acting. |
| `held_item` | int | Type ID of item the agent holds. `-1` = empty hands. |
| `type_ids` | dict | Maps object names to integer type IDs. `ctx.type_ids["goal"]` returns the goal's type ID. |
| `object_type_map` | `(H, W)` int array | Grid of object type IDs. |
| `object_state_map` | `(H, W)` int array | Grid of per-cell state values. |
| `agent_inv` | `(n_agents, 1)` int array | All agents' inventories. |

### Extra-state fields

Arrays declared by components via `extra_state_schema` are accessible directly as attributes on `ctx`. The scope prefix is stripped automatically — if a component declares `goals_collected`, access it as `ctx.goals_collected`.

### Lookup tables

Built-in branches use guard tables to decide whether an action is allowed for a given object type and held item. These are available on the context:

- `ctx.CAN_PICKUP` — which object types can be picked up
- `ctx.PICKUP_FROM_GUARD` — which (object, held-item) pairs allow pickup-from
- `ctx.PLACE_ON_GUARD` — which (object, held-item) pairs allow place-on

## Built-in Branches

The autowire system assembles the branch list automatically from registered components. These are the built-in branches, in the order they run:

| Branch | Fires on | Description |
|--------|----------|-------------|
| `branch_pickup` | PickupDrop | Pick up a loose object from the faced cell into the agent's inventory. |
| `branch_pickup_from_container` | PickupDrop | Pick up the cooked result from a container (requires holding the right item, e.g. a plate). |
| `branch_pickup_from` | PickupDrop | Pick up from stacks or counters. Stacks dispense infinitely; counters yield their stored item. |
| `branch_drop_on_empty` | PickupDrop | Drop the held item onto an empty floor cell. |
| `branch_place_on_container` | PickupDrop | Place an ingredient into a container. Validates against recipe prefixes and container capacity. |
| `branch_place_on_consume` | PickupDrop | Place an item on a consume surface (e.g. delivery zone). The item is removed from play. |
| `branch_place_on` | PickupDrop | Place the held item on a surface like a counter. |

Container and consume branches are only included when the environment has container or consume objects. Autowire handles this automatically.

## Helper Functions

Import from `cogrid.core.pipeline.context`:

??? api "Helper functions"

    ::: cogrid.core.pipeline.context.clear_facing_cell
        options:
          heading_level: 4

    ::: cogrid.core.pipeline.context.set_facing_cell
        options:
          heading_level: 4

    ::: cogrid.core.pipeline.context.pickup_from_facing_cell
        options:
          heading_level: 4

    ::: cogrid.core.pipeline.context.place_in_facing_cell
        options:
          heading_level: 4

    ::: cogrid.core.pipeline.context.give_item
        options:
          heading_level: 4

    ::: cogrid.core.pipeline.context.empty_hands
        options:
          heading_level: 4

    ::: cogrid.core.pipeline.context.increment
        options:
          heading_level: 4

    ::: cogrid.core.pipeline.context.find_facing_instance
        options:
          heading_level: 4

## Writing a Custom Interaction

Define a function with the branch signature and pass it in the config. Here is an example that removes a goal when an agent picks it up:

```python
from cogrid.core.pipeline.context import clear_facing_cell, increment


def collect_goal(ctx):
    """Remove a goal when the agent picks it up."""
    goal_id = ctx.type_ids["goal"]
    is_pickup = ctx.action == ctx.action_id.pickup_drop
    should_apply = ctx.can_interact & is_pickup & (ctx.facing_type == goal_id)
    changes = {
        "object_type_map": clear_facing_cell(ctx),
        "goals_collected": increment(ctx.goals_collected, ctx.agent_index),
    }
    return should_apply, changes
```

Wire it into the environment config:

```python
goal_config = {
    ...
    "interactions": [collect_goal],
}
```

User-provided interactions run before auto-discovered ones, so they take priority for overlapping keys.

## Composition

The autowire system reads container and consume metadata from registered objects and calls `compose_interactions()` internally. This returns:

- An **extras function** that pre-computes container context (contents, timers, recipe matching) and attaches it to the interaction context.
- A **branch list** assembled conditionally — container branches are included only when containers exist, consume branches only when consume surfaces exist.

User-provided `"interactions"` from the config are prepended to the auto-discovered list, so they run first.
