"""Cardinal direction enumeration."""

# Enumeration of possible actions
from enum import IntEnum


class Directions(IntEnum):
    """Cardinal direction enum (Right=0, Down=1, Left=2, Up=3)."""

    Right = 0
    Down = 1
    Left = 2
    Up = 3
