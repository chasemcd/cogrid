# Grid & Layouts

A CoGrid environment is built on a 2D grid. The grid state is stored as three parallel integer arrays:

| Array | Contents |
|-------|----------|
| `wall_map` | Binary mask. `1` = impassable wall cell. |
| `object_type_map` | Integer type ID per cell (e.g. counter, pot, onion stack). |
| `object_state_map` | Per-cell state value (e.g. what a counter holds, pot cook progress). |

All three share the same `(H, W)` shape and are updated each step by the engine.

## ASCII Layouts

Layouts are defined as lists of equal-length strings. Each character maps to a registered object type via its `char` attribute.

```python
layout = [
    "CCUCC",
    "O   O",
    "C   C",
    "C=C@C",
]
```

### Built-in Characters

| Char | Object | Meaning |
|------|--------|---------|
| `#` | Wall | Impassable, blocks visibility |
| `+` | Spawn | Agent start position (replaced with floor at runtime) |
| ` ` | Empty | Walkable floor tile |

Environment-scoped objects add their own characters. For example, the Overcooked scope maps `O` to OnionStack, `U` to Pot, `=` to PlateStack, `@` to DeliveryZone, and `C` to Counter.

## Registering Layouts

Use `register_layout()` to store a named layout in the global registry:

```python
from cogrid.core.grid.layouts import register_layout

register_layout(
    "my_layout_v0",
    [
        "######",
        "#O  @#",
        "# ++ #",
        "#C  C#",
        "######",
    ],
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout_id` | `str` | Unique name for the layout. Raises `ValueError` if already taken. |
| `layout` | `list[str]` | Grid rows as strings. All rows must be the same length. |
| `state_encoding` | `list[list[int]]` or `ndarray` or `None` | Optional initial `object_state_map` values. Defaults to all zeros. |

The layout ID is referenced in the environment config:

```python
config = {
    "grid": {"layout": "my_layout_v0"},
    ...
}
```

## Retrieving Layouts

```python
from cogrid.core.grid.layouts import get_layout

layout, state_encoding = get_layout("my_layout_v0")
```

Returns a `(layout, state_encoding)` tuple.

## Layout Examples

### Overcooked Cramped Room

```
CCUCC
O   O
C   C
C=C@C
```

A compact 4x5 kitchen with one pot (`U`), two onion stacks (`O`), a plate stack (`=`), and a delivery zone (`@`). Counters (`C`) line the perimeter.

### Overcooked Coordination Ring

```
CCCUC
C   U
= C C
O   C
CO@CC
```

A ring-shaped layout that forces agents to navigate around a central counter. Two pots on the top and right edges.

### Overcooked Mixed Kitchen

```
CUCCCUC
O+ C +T
=  C  =
C  C  C
CCCCC@C
```

A dual-recipe kitchen with both onion (`O`) and tomato (`T`) stacks, two pots, and two plate stacks.
