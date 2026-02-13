"""GridObj defines an object in the CoGridEnv environment. It is largely derived from the Minigrid WorldObj:
https://github.com/Farama-Foundation/Minigrid/minigrid/core/world_object.py

"""

from __future__ import annotations
from copy import deepcopy
import math
import uuid

import numpy as np


from cogrid.constants import GridConstants
from cogrid.core.constants import ObjectColors, Colors, COLORS, COLOR_NAMES
from cogrid.core import constants
from cogrid.core.directions import Directions
from cogrid.visualization.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
    add_text_to_image,
)


# The OBJECT_REGISTRY holds all registered objects under a "scope" (e.g., "global", "search_rescue", "overcooked")
# which allows us to re-use IDs and character representations across environments (e.g., P is a purple target in
# search_rescue and a plate in overcooked).
OBJECT_REGISTRY: dict[str, dict[str, GridObj]] = {}

# Maps (scope, object_id) -> dict of static boolean properties for lookup table generation.
# Populated by the @register_object_type decorator.
_OBJECT_TYPE_PROPERTIES: dict[tuple[str, str], dict[str, bool]] = {}

# Known classmethod names to scan for during component registration.
_COMPONENT_METHODS = frozenset({
    "build_tick_fn",
    "build_interaction_fn",
    "extra_state_schema",
    "extra_state_builder",
    "build_static_tables",
})


def make_object(
    object_id: str | None, scope: str = "global", **kwargs
) -> GridObj:
    if object_id is None:
        return None

    if scope not in OBJECT_REGISTRY:
        raise ValueError(
            f"No objects registered with scope `{scope}`. Existing scopes are {list(OBJECT_REGISTRY.keys())}."
        )

    if object_id in OBJECT_REGISTRY["global"]:
        return OBJECT_REGISTRY["global"][object_id](**kwargs)
    elif object_id not in OBJECT_REGISTRY[scope]:
        raise ValueError(
            f"Object with object_id `{object_id}` not registered in scope `{scope}`. "
            f"Call register_object('{object_id}', <class>, scope='{scope}') to add it to the registry."
        )

    return OBJECT_REGISTRY[scope][object_id](**kwargs)


def get_object_class(object_id: str, scope: str = "global") -> GridObj:
    return OBJECT_REGISTRY[scope][object_id]


def register_object(
    object_id: str, obj_class: GridObj, scope: str = "global"
) -> None:
    global_scope_chars = [
        obj.char for obj in OBJECT_REGISTRY.get("global", {}).values()
    ]

    if obj_class.char in global_scope_chars:
        raise ValueError(
            f"Character `{obj_class.char}` is already in use in the global scope. "
            "Please choose a different character."
        )

    if object_id in OBJECT_REGISTRY.get("global", {}):
        raise ValueError(
            f"Object with object_id `{object_id}` already registered in the global scope. "
            "Please select a different ID."
        )

    if scope not in OBJECT_REGISTRY:
        OBJECT_REGISTRY[scope] = {}

    OBJECT_REGISTRY[scope][object_id] = obj_class


def register_object_type(
    object_id: str,
    scope: str = "global",
    can_pickup: bool = False,
    can_overlap: bool = False,
    can_place_on: bool = False,
    can_pickup_from: bool = False,
    is_wall: bool = False,
):
    """Decorator that registers a GridObj subclass with static property metadata.

    This replaces the manual ``register_object(id, cls, scope)`` pattern.
    The decorator calls ``register_object()`` internally for backward
    compatibility and additionally stores boolean properties in
    ``_OBJECT_TYPE_PROPERTIES`` for use by ``build_lookup_tables()``.

    Usage::

        @register_object_type("wall", is_wall=True)
        class Wall(GridObj):
            ...

    Args:
        object_id: Unique string identifier for this object type.
        scope: Registry scope (e.g. "global", "overcooked").
        can_pickup: Whether agents can pick up this object.
        can_overlap: Whether agents can walk over this object.
        can_place_on: Whether another object can be placed on this one.
        can_pickup_from: Whether agents can pick up an item from this object.
        is_wall: Whether this object acts as a wall (blocks movement and sight).
    """

    def decorator(cls):
        # Lazy import to avoid circular dependency
        from cogrid.core.component_registry import (
            get_all_components,
            register_component_metadata,
            _validate_classmethod_signature,
        )

        properties = {
            "can_pickup": can_pickup,
            "can_overlap": can_overlap,
            "can_place_on": can_place_on,
            "can_pickup_from": can_pickup_from,
            "is_wall": is_wall,
        }

        # Store static properties for lookup table generation
        _OBJECT_TYPE_PROPERTIES[(scope, object_id)] = properties

        # Duplicate char detection within the same scope
        for existing in get_all_components(scope):
            if existing.char == cls.char:
                raise ValueError(
                    f"Duplicate char '{cls.char}' in scope '{scope}': "
                    f"{existing.cls.__name__} and {cls.__name__}"
                )

        # Set object_id on the class
        cls.object_id = object_id

        # Delegate to existing register_object for backward compatibility
        register_object(object_id, cls, scope=scope)

        # Convention-based classmethod scan
        discovered: dict = {}
        for method_name in _COMPONENT_METHODS:
            method = getattr(cls, method_name, None)
            if method is not None and callable(method):
                _validate_classmethod_signature(cls, method_name, method)
                discovered[method_name] = method

        # Store component metadata in the registry
        register_component_metadata(
            scope=scope,
            object_id=object_id,
            cls=cls,
            properties=properties,
            methods=discovered,
        )

        return cls

    return decorator


def build_lookup_tables(scope: str = "global") -> dict[str, np.ndarray]:
    """Build integer-indexed property lookup tables from the type registry.

    Returns a dict of 1-D ``int32`` arrays, one per boolean property. Each
    array is indexed by the same integer encoding produced by
    ``object_to_idx()`` / ``get_object_names()``.

    The arrays are created with the backend array library (``xp``) so they
    work on both numpy and JAX backends.

    Args:
        scope: The registry scope to build tables for.

    Returns:
        Dict with keys ``"CAN_PICKUP"``, ``"CAN_OVERLAP"``, ``"CAN_PLACE_ON"``,
        ``"CAN_PICKUP_FROM"``, ``"IS_WALL"``, each mapping to an array of
        shape ``(n_types,)`` with dtype ``int32``.
    """
    from cogrid.backend import xp  # import at call time to avoid circular imports
    from cogrid.backend.array_ops import set_at

    type_names = get_object_names(scope=scope)
    n_types = len(type_names)

    property_keys = [
        "CAN_PICKUP",
        "CAN_OVERLAP",
        "CAN_PLACE_ON",
        "CAN_PICKUP_FROM",
        "IS_WALL",
    ]

    tables = {key: xp.zeros(n_types, dtype=xp.int32) for key in property_keys}

    for idx, name in enumerate(type_names):
        if name is None:
            # Index 0: empty cell -- overlappable
            tables["CAN_OVERLAP"] = set_at(tables["CAN_OVERLAP"], idx, 1)
            continue

        if name == "free_space":
            # Index 1: free_space -- overlappable (not in OBJECT_REGISTRY, hardcoded)
            tables["CAN_OVERLAP"] = set_at(tables["CAN_OVERLAP"], idx, 1)
            continue

        if name.startswith("agent_"):
            # Agent direction placeholders -- skip, leave all-zero
            continue

        # Look up properties: try (scope, name) first, then ("global", name)
        props = _OBJECT_TYPE_PROPERTIES.get((scope, name))
        if props is None:
            props = _OBJECT_TYPE_PROPERTIES.get(("global", name))

        if props is None:
            # Object registered via old register_object() without decorator.
            # Default to all-False properties.
            import warnings
            warnings.warn(
                f"Object '{name}' in scope '{scope}' has no static properties "
                f"(not registered via @register_object_type). "
                f"Defaulting to all-False in lookup tables.",
                stacklevel=2,
            )
            continue

        prop_map = {
            "can_pickup": "CAN_PICKUP",
            "can_overlap": "CAN_OVERLAP",
            "can_place_on": "CAN_PLACE_ON",
            "can_pickup_from": "CAN_PICKUP_FROM",
            "is_wall": "IS_WALL",
        }

        for prop_name, table_key in prop_map.items():
            if props.get(prop_name, False):
                tables[table_key] = set_at(tables[table_key], idx, 1)

    return tables


def get_registered_object_ids(scope: str = "global") -> list[str]:
    """Return a list of the object_ids of available objects in a given scope."""
    return list(OBJECT_REGISTRY[scope].keys())


def get_object_char(object_id: str, scope: str = "global") -> str:
    return get_object_class(object_id, scope=scope).char


def get_object_id_from_char(object_char: str, scope: str = "global") -> str:
    # First check global scope, no matter what scope was passed (default to global).
    for object_id, object_class in OBJECT_REGISTRY["global"].items():
        if object_class.char == object_char:
            return object_id

    if scope != "global":
        for object_id, object_class in OBJECT_REGISTRY[scope].items():
            if object_class.char == object_char:
                return object_id

    raise ValueError(
        f"There is no registered object with character representation `{object_char}` in scope `{scope}`."
    )


class GridObj:
    object_id: str = None
    color: str | tuple = None
    char: str = None

    def __init__(
        self,
        state: int = 0,
        toggle_value: float = 0,
        inventory_value: float = 0,
        overlap_value: float = 0,
        placed_on_value: float = 0,
        picked_up_from_value: float = 0,
    ):
        self.uuid: str = str(uuid.uuid4())

        self.state: int = state

        # If an object can be placed on top of this one, this will hold the object that's on top.
        self.obj_placed_on: GridObj | None = None

        # position info
        self.init_pos: tuple[int, int] | None = None
        self.pos: tuple[int, int] | None = None

        # defines rewards for holding/toggling/overlapping
        self.toggle_value: float | int = toggle_value
        self.inventory_value: float | int = inventory_value
        self.overlap_value: float | int = overlap_value
        self.placed_on_value: float | int = placed_on_value
        self.picked_up_from_value: float | int = picked_up_from_value

    def can_overlap(self, agent: GridAgent) -> bool:
        """Can an agent overlap with this object?"""
        return False

    def can_pickup(self, agent: GridAgent) -> bool:
        """Can an agent pick this object up and store in inventory?"""
        return False

    def can_place_on(self, agent: GridAgent, cell: GridObj) -> bool:
        """
        Can another object be placed on top of this object? e.g., a countertop that can't be walked through
        but can have an item on top of it.
        """
        return False

    def can_pickup_from(self, agent: GridAgent) -> bool:
        """Can the agent pick up an object from this one?"""
        return self.obj_placed_on is not None and self.obj_placed_on.can_pickup(
            agent=agent
        )

    def place_on(self, agent: GridAgent, cell: GridObj) -> None:
        self.obj_placed_on = cell
        self.state = hash(cell.__class__.__name__) % (2**31 - 1)

    def pick_up_from(self, agent: GridAgent) -> GridObj:
        assert (
            self.obj_placed_on is not None
        ), f"Picking up from but there's no object placed on {self.object_id}"
        cell = self.obj_placed_on
        self.obj_placed_on = None
        self.state = 0
        return cell

    def see_behind(self, agent: GridAgent) -> bool:
        """Can the agent see through this object?"""
        return True

    def visible(self) -> bool:
        return True

    def toggle(self, env, agent: GridAgent = None) -> bool:
        """
        Trigger/Toggle an action this object performs. Some toggles are conditioned on the environment
        and require specific conditions to be met, which can be checked with the end.
        """
        return False

    def encode(self, encode_char=True, scope: str = "global"):
        return (
            self.char if encode_char else object_to_idx(self, scope=scope),
            0,  # TODO(chase): Remove. This used to be color, but we're no longer using it.
            int(self.state),
        )

    def render(self, tile_img):
        """By default, everything will be rendered as a square with the specified color."""
        fill_coords(tile_img, point_in_rect(0, 1, 0, 1), color=self.color)

    @staticmethod
    def decode(char_or_idx: str | int, state: int, scope: str = "global"):

        if char_or_idx in [
            None,
            GridConstants.FreeSpace,
            GridConstants.Obscured,
        ]:
            return None

        # check if the name was passed instead of the character
        if _is_str(char_or_idx) and len(char_or_idx) > 1:
            object_id = char_or_idx
        elif _is_str(char_or_idx):
            object_id = get_object_id_from_char(char_or_idx, scope=scope)
        else:
            raise ValueError(f"Invalid identifier for decoding: {char_or_idx}")

        state = int(state)

        return make_object(object_id, state=state, scope=scope)

    def rotate_left(self):
        """Some objects (e.g., agents) have a rotation and must be rotated with the grid."""
        pass

    def tick(self):
        """
        Some objects have a time component (e.g., cooking soup), so we call the tick
        method on all objects for each env.step()
        """
        pass

    def _remove_from_grid(self, grid):
        cell = grid.get(*self.pos)
        assert self is cell
        grid.set(*self.pos, None)


def _is_str(chk):
    return isinstance(chk, str) or isinstance(chk, np.str)


def _is_int(chk):
    return isinstance(chk, int) or isinstance(chk, np.int)


def get_object_names(scope: str = "global") -> list[str]:
    """Get a list of all registered object IDs in a consistent order.

    The order is important for encoding/decoding and must remain stable.
    Returns a list starting with [None, "free_space"] followed by global objects,
    then scope-specific objects, and finally agent directions.

    Args:
        scope: The scope to include objects from, in addition to global scope
    """
    # Start with None and free_space which are special cases
    names = [None, "free_space"]

    # Add all registered global objects in sorted order (except free_space which we already added)
    global_objects = sorted(
        [
            obj_id
            for obj_id in OBJECT_REGISTRY.get("global", {}).keys()
            if obj_id != "free_space"
        ]
    )
    names.extend(global_objects)

    # Add scope-specific objects if a non-global scope is specified
    if scope != "global" and scope in OBJECT_REGISTRY:
        scope_objects = sorted(
            [
                obj_id
                for obj_id in OBJECT_REGISTRY[scope].keys()
                # Skip any objects that might overlap with global scope
                if obj_id not in OBJECT_REGISTRY.get("global", {})
            ]
        )
        names.extend(scope_objects)

    # Add agent directions last
    names.extend([f"agent_{direction}" for direction in "^>v<"])

    return names


def object_to_idx(object: GridObj | str | None, scope: str = "global") -> int:
    """Convert an object or object_id to its integer index."""
    if isinstance(object, GridObj):
        object_id = object.object_id
    else:
        object_id = object

    return get_object_names(scope=scope).index(object_id)


def idx_to_object(idx: int, scope: str = "global") -> str:
    """Convert an integer index back to its object_id."""
    names = get_object_names(scope=scope)
    if idx >= len(names):
        raise ValueError(
            f"Object index {idx} not in object registry (checked global and {scope} scopes)."
        )
    return names[idx]


class GridAgent(GridObj):
    def __init__(self, agent, num_agents: int, scope: str = "global"):
        """
        Grid agents are initialized slightly differently. State corresponds to the object they are holding
        and char/colors are unique for each agent.
        """

        self.char = {
            Directions.Up: "^",
            Directions.Down: "v",
            Directions.Left: "<",
            Directions.Right: ">",
        }[agent.dir]

        assert (
            len(agent.inventory) <= 1
        ), "Current implementation requires maximum inventory size of 1."

        self.object_id = f"agent_{self.char}"

        # TODO(chase): State must encapsulate carried objects and role
        # state = agent.role_idx
        state = (
            0
            if len(agent.inventory) == 0
            else object_to_idx(agent.inventory[0], scope=scope)
        )

        super().__init__(state=state)
        self.dir = agent.dir
        self.pos = agent.pos
        self.front_pos = agent.front_pos
        self.agent_id = agent.id
        self.inventory: list[GridObj] = deepcopy(agent.inventory)
        assert self.pos is not None

        # Generate high-contrast colors based on HSV color space
        # Hue values are evenly spaced around the color wheel
        hue = (agent.agent_number - 1) * (360 / num_agents)
        # Use high saturation (0.7-1.0) and value (0.8-1.0) for vibrant colors
        # This avoids whites (high V, low S), blacks (low V), and greys (low S)
        rgb_color = self._hsv_to_rgb(hue, 0.35, 0.99)
        self.color = rgb_color

    def rotate_left(self):
        self.char = {"^": "<", "<": "v", "v": ">", ">": "^"}[self.char]
        self.object_id = f"agent_{self.char}"
        self.dir -= 1
        if self.dir < 0:
            self.dir += 4

    def render(self, tile_img):
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the triangle based on agent direction
        assert self.dir is not None
        tri_fn = rotate_fn(
            tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir
        )
        fill_coords(tile_img, tri_fn, self.color)

        # add any item in the inventory to the corner
        inv_tile_rows, inv_tile_cols = (
            tile_img.shape[0] // 3,
            tile_img.shape[1] // 3,
        )
        assert (
            len(self.inventory) <= 3
        ), "We're rendering inventory items at 1/3 size, so can't do more than 3!"

        offset = 4  # offset so we still see grid lines
        for i, obj in enumerate(self.inventory):
            inventory_tile = np.zeros(shape=(inv_tile_rows, inv_tile_cols, 3))
            obj.render(inventory_tile)

            # Take the subset of the image that we'll fill, then only fill where the image is non-zero
            # this makes a transparent background, rather than adding in a black square
            tile_subset = tile_img[
                i * inv_tile_rows + offset : (i + 1) * inv_tile_rows + offset,
                offset : inv_tile_cols + offset,
                :,
            ]
            nonzero_entries = np.nonzero(inventory_tile)
            tile_subset[nonzero_entries] = inventory_tile[nonzero_entries]

    @staticmethod
    def decode(char_or_idx: str | int, state: int, scope: str = "global"):

        if char_or_idx in [
            None,
            GridConstants.FreeSpace,
            GridConstants.Obscured,
        ]:
            return None

        # check if the name was passed instead of the character
        if _is_str(char_or_idx) and len(char_or_idx) > 1:
            object_id = char_or_idx
        elif _is_str(char_or_idx):
            object_id = get_object_id_from_char(char_or_idx)
        elif _is_int(char_or_idx):
            object_id = get_object_names(scope=scope)[char_or_idx]
        else:
            raise ValueError(f"Invalid identifier for decoding: {char_or_idx}")

        state = int(state)

        return make_object(object_id, state=state)

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
        """Convert HSV color values to RGB tuple."""
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c

        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return ((r + m) * 255.0, (g + m) * 255.0, (b + m) * 255.0)


@register_object_type("wall", is_wall=True)
class Wall(GridObj):
    object_id = "wall"
    color = constants.Colors.Grey
    char = "#"

    def __init__(self, *args, **kwargs):
        super().__init__(state=0)

    def see_behind(self) -> bool:
        return False


@register_object_type("floor", can_overlap=True)
class Floor(GridObj):
    object_id = "floor"
    color = constants.Colors.PaleBlue
    char = GridConstants.FreeSpace

    def __init__(self, **kwargs):
        super().__init__(
            state=0,
        )

    def can_overlap(self) -> bool:
        return True


@register_object_type("counter", can_place_on=True)
class Counter(GridObj):
    object_id = "counter"
    color = constants.Colors.LightBrown
    char = "C"

    def __init__(self, state: int = 0, **kwargs):
        # TODO(chase): need to be able to initialize an object on top
        #   via the state. Take state and map it to an object.
        super().__init__(
            state=state,
        )

    def can_place_on(self, agent: GridAgent, cell: GridObj) -> bool:
        return self.obj_placed_on is None

    def render(self, tile_img):
        super().render(tile_img)

        if self.obj_placed_on is not None:
            self.obj_placed_on.render(tile_img)


@register_object_type("key", can_pickup=True)
class Key(GridObj):
    object_id = "key"
    color = constants.Colors.Yellow
    char = "K"

    def __init__(self, state=0):
        super().__init__(state=state)

    def can_pickup(self, agent: GridAgent):
        return True

    def render(self, tile_img):
        # Vertical quad
        fill_coords(tile_img, point_in_rect(0.50, 0.63, 0.31, 0.88), self.color)

        # Teeth
        fill_coords(tile_img, point_in_rect(0.38, 0.50, 0.59, 0.66), self.color)
        fill_coords(tile_img, point_in_rect(0.38, 0.50, 0.81, 0.88), self.color)

        # Ring
        fill_coords(
            tile_img, point_in_circle(cx=0.56, cy=0.28, r=0.190), self.color
        )
        fill_coords(
            tile_img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0)
        )


@register_object_type("door")
class Door(GridObj):
    object_id = "door"
    color = constants.Colors.DarkGrey
    char = "D"

    def __init__(self, state):
        super().__init__(state=state)
        self.is_open = state == 2
        self.is_locked = state == 0

    def can_overlap(self, agent: GridAgent) -> bool:
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self, agent: GridAgent) -> bool:
        return self.is_open

    def toggle(self, env, agent: GridAgent) -> bool:
        if self.is_locked:
            if any([isinstance(obj, Key) for obj in agent.inventory]):
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self, encode_char=False):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            self.state = 2
        elif self.is_locked:
            self.state = 0
        # if door is closed and unlocked
        elif not self.is_open:
            self.state = 1
        else:
            raise ValueError(
                f"There is no possible state encoding for the state:\n -Door Open: {self.is_open}\n -Door Closed: {not self.is_open}\n -Door Locked: {self.is_locked}"
            )

        return super().encode(encode_char=encode_char)

    def render(self, tile_img):

        if self.state == 2:
            fill_coords(
                tile_img, point_in_rect(0.88, 1.00, 0.00, 1.00), self.color
            )
            fill_coords(
                tile_img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0)
            )
            return

        # Door frame and door
        if self.state == 0:
            fill_coords(
                tile_img, point_in_rect(0.00, 1.00, 0.00, 1.00), self.color
            )
            fill_coords(
                tile_img,
                point_in_rect(0.06, 0.94, 0.06, 0.94),
                0.45 * np.array(self.color),
            )

            # Draw key slot
            fill_coords(
                tile_img, point_in_rect(0.52, 0.75, 0.50, 0.56), self.color
            )
        else:
            fill_coords(
                tile_img, point_in_rect(0.00, 1.00, 0.00, 1.00), self.color
            )
            fill_coords(
                tile_img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0)
            )
            fill_coords(
                tile_img, point_in_rect(0.08, 0.92, 0.08, 0.92), self.color
            )
            fill_coords(
                tile_img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0)
            )

            # Draw door handle
            fill_coords(
                tile_img, point_in_circle(cx=0.75, cy=0.50, r=0.08), self.color
            )
