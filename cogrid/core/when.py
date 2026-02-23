"""Declarative capability descriptor for grid objects.

Usage::

    from cogrid.core.when import when

    @register_object_type("onion", scope="overcooked")
    class Onion(GridObj):
        can_pickup = when()
"""


class When:
    """Marker that declares an object capability as a class attribute.

    ``bool(When())`` is always ``True``, so lookup-table generation and
    truthiness checks work without change.  Future-extensible for
    conditional capabilities (e.g. ``when(agent_holding="plate")``).
    """

    def __init__(self, **conditions):
        self.conditions = conditions

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
