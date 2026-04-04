# Objects

Every cell on the grid can hold a `GridObj` instance. Objects define their appearance, behavior, and interaction rules through class attributes and methods.

## GridObj Base Class

All grid objects inherit from `GridObj`:

```python
from cogrid.core.objects.base import GridObj

class MyObject(GridObj):
    object_id: str = None   # set by @register_object_type
    color: str | tuple       # render color (name or RGB tuple)
    char: str                # single-character layout symbol
```

**Core attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `object_id` | `str` | Set automatically by the registration decorator. |
| `color` | `str | tuple` | Render color. Use `cogrid.core.constants.Colors` or an RGB tuple. |
| `char` | `str` | Single character used in ASCII layouts. |
| `state` | `int` | Mutable integer state (e.g. door open/closed, pot cook progress). |
| `pos` | `tuple[int, int]` | Current `(row, col)` position on the grid. |
| `obj_placed_on` | `GridObj | None` | Object stacked on top of this one (e.g. item on a counter). |

**Methods:**

| Method | Default | Purpose |
|--------|---------|---------|
| `encode()` | type + state tuple | Converts to `(char_or_idx, extra, state)` for the state arrays. |
| `render(tile_img)` | Filled rectangle | Draws the object on a tile image. |
| `tick()` | No-op | Called each step for time-dependent behavior. |

## Registration

Objects are registered with the `@register_object_type` decorator:

```python
from cogrid.core.objects.registry import register_object_type

@register_object_type("my_object", scope="my_env")
class MyObject(GridObj):
    color = (200, 100, 50)
    char = "M"
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `name` | Unique string ID within the scope. |
| `scope` | Registry scope. `"global"` for built-in objects, or an environment name like `"overcooked"`. |

Objects in a named scope are only visible to environments that use that scope. Global objects are available everywhere.

## Built-in Objects (Global Scope)

| Object | Char | Color | Behavior |
|--------|------|-------|----------|
| `Wall` | `#` | Grey | Impassable, blocks visibility |
| `Floor` | ` ` | Pale Blue | Walkable empty tile |
| `Counter` | `C` | Light Brown | Surface that holds one object on top |
| `Key` | `K` | Yellow | Pickupable item |
| `Door` | `D` | Dark Grey | Three states: locked (0), closed (1), open (2) |

## Object Properties

Objects declare interaction rules through class-level descriptors imported from `cogrid.core.when`:

```python
from cogrid.core.objects.when import when

@register_object_type("counter")
class Counter(GridObj):
    can_place_on = when()    # agents can place held items on this
    can_pickup_from = when() # agents can pick up items from this
```

The `when()` descriptor (with no arguments) means the action is always allowed. The engine reads these descriptors at init time and compiles them into lookup tables.

## Containers

Objects that hold multiple items use the `Container` descriptor:

```python
from cogrid.core.objects.containers import Container
from cogrid.core.objects.base import GridObj
from cogrid.core.objects.registry import register_object_type

@register_object_type("pot", scope="overcooked")
class Pot(GridObj):
    container = Container(capacity=3, pickup_requires="plate")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `capacity` | `int` | Maximum items the container holds. |
| `pickup_requires` | `str | list[str] | None` | What the agent must hold to pick up from the container. `None` = empty inventory. |

The autowire system reads `container` at init time and generates:

- **Extra state arrays** — contents, timer, and positions per container instance.
- **Tick function** — decrements timers when the container is full.
- **Render sync** — syncs container state back to `GridObj` instances for rendering.
- **Interaction branches** — pickup/drop logic based on `capacity` and `pickup_requires`.

Containers can be paired with environment-specific recipe systems. The [Overcooked environment](../environments/overcooked.md#recipes) defines a `Recipe` dataclass that declares how ingredients combine into results inside a container.

## Tick Functions

A tick function runs once per step, before movement and interactions. It is the mechanism for time-dependent object behavior — cooking timers, object movement, state decay, anything that should happen every step regardless of agent actions.

### Pattern

Define a `build_tick_fn` classmethod on your `GridObj` subclass. It returns a closure:

```python
@register_object_type("pot", scope="overcooked")
class Pot(GridObj):
    container = Container(capacity=3, pickup_requires="plate")

    @classmethod
    def build_tick_fn(cls):
        def tick(state, scope_config):
            # Read container arrays from extra_state
            timer = state.extra_state[f"{scope_config['scope']}.pot_timer"]
            contents = state.extra_state[f"{scope_config['scope']}.pot_contents"]

            # Decrement timer for full containers (all slots nonzero)
            full = (contents > 0).all(axis=-1)
            cooking = full & (timer > 0)
            new_timer = xp.where(cooking, timer - 1, timer)

            new_extra = {**state.extra_state,
                         f"{scope_config['scope']}.pot_timer": new_timer}
            return dataclasses.replace(state, extra_state=new_extra)

        return tick
```

### Signature

The returned closure must match:

```
fn(state: EnvState, scope_config: dict) -> EnvState
```

- `state` — the full environment state (grid arrays, agent positions, extra state).
- `scope_config` — dict with `"type_ids"`, `"scope"`, and other init-time metadata.
- Returns a new `EnvState` with any modifications applied.

### Discovery and composition

The autowire system discovers all `build_tick_fn` classmethods on registered objects and composes them sequentially. Container objects get a tick function automatically (see [Containers](#containers) above). There is nothing to wire manually — registering the object is enough.

The step pipeline runs all tick functions before movement and interactions:

**tick → movement → interactions → observations → rewards**

## Other Lifecycle Hooks

Objects can define additional classmethods for initialization and rendering:

| Hook | Signature | Purpose |
|------|-----------|---------|
| `extra_state_schema()` | `cls -> dict` | Declare additional state arrays (shape, dtype). |
| `extra_state_builder()` | `cls -> callable` | Return a factory that initializes extra state arrays at reset time. |
| `build_render_sync_fn()` | `cls -> callable` | Return a callback that syncs array state back to `GridObj` instances for rendering. |

Example: `Counter.build_render_sync_fn()` returns a callback that reads `object_state_map` values and sets `obj_placed_on` on each Counter instance so items placed on counters are drawn correctly.
