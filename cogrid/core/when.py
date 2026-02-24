"""Declarative capability descriptor for grid objects.

Usage::

    from cogrid.core.when import when

    @register_object_type("onion", scope="overcooked")
    class Onion(GridObj):
        can_pickup = when()

    @register_object_type("pot", scope="overcooked")
    class Pot(GridObj):
        can_pickup_from = when(agent_holding="plate")
        can_place_on = when(agent_holding=["onion", "tomato"])
"""

SUPPORTED_CONDITIONS = frozenset({"agent_holding"})


class When:
    """Marker that declares an object capability as a class attribute.

    ``bool(When())`` is always ``True``, so lookup-table generation and
    truthiness checks work without change.

    Supports optional conditions that compile into guard tables at init
    time.  Currently supported conditions:

    - ``agent_holding``: ``str`` or ``list[str]`` — the agent must be
      holding one of the named object types for the interaction to proceed.
    """

    def __init__(self, **conditions):
        unsupported = set(conditions) - SUPPORTED_CONDITIONS
        if unsupported:
            raise ValueError(
                f"Unsupported when() conditions: {unsupported}. "
                f"Supported: {sorted(SUPPORTED_CONDITIONS)}"
            )
        # Normalize agent_holding to list[str]
        ah = conditions.get("agent_holding")
        if ah is not None:
            if isinstance(ah, str):
                conditions["agent_holding"] = [ah]
            elif not isinstance(ah, list) or not all(isinstance(s, str) for s in ah):
                raise TypeError(
                    "agent_holding must be a str or list[str], "
                    f"got {type(ah).__name__}"
                )
        self.conditions = conditions

    @property
    def has_conditions(self) -> bool:
        """True if any conditions were specified."""
        return bool(self.conditions)

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        if not self.conditions:
            return "when()"
        args = ", ".join(f"{k}={v!r}" for k, v in self.conditions.items())
        return f"when({args})"


def when(**conditions) -> When:
    """Create a capability descriptor."""
    return When(**conditions)
