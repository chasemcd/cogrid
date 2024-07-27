import numpy as np


LAYOUT_REGISTRY: dict[str, tuple[list[str], list[int]]] = {}


def make(layout_id: str, **kwargs) -> tuple[list[str], list[int]]:
    return LAYOUT_REGISTRY[layout_id]


def register(
    layout_id: str,
    layout: list[str],
    state_encoding: list[list[int]] | np.ndarray | None,
) -> None:
    if state_encoding is None:
        state_encoding = np.zeros((len(layout), len(layout[0])))

    if layout_id in LAYOUT_REGISTRY:
        raise ValueError(
            f"There is already a layout registered with ID {layout_id}. Please use a unique ID."
        )

    LAYOUT_REGISTRY[layout_id] = (layout, state_encoding)
