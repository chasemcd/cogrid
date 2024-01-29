# Enumeration of possible actions
from __future__ import annotations

from enum import IntEnum


class Actions(IntEnum):
    Left = 0  # Turn Left
    Right = 1  # Turn Right
    Forward = 2  # Move forward
    Pickup = 3  # Pick up an object
    Drop = 4  # Drop an object
    Toggle = 5  # Interact with/toggle/activate an object
    Noop = 6
