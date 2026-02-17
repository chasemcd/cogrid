"""Grid-level constants (free space, padding, obscured, spawn chars)."""

import dataclasses


@dataclasses.dataclass
class GridConstants:
    """Character constants for grid cell types."""

    FreeSpace = " "
    Padding = "0"
    Obscured = "."
    Spawn = "+"
