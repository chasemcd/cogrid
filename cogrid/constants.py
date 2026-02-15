import dataclasses

"""
Define all constants for CoGridEnv environments.
"""


@dataclasses.dataclass
class GridConstants:
    FreeSpace = " "
    Padding = "0"
    Obscured = "."
    Spawn = "+"
