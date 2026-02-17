"""Agent role definitions for role-based environments."""

import dataclasses


@dataclasses.dataclass
class Roles:
    """Named roles for role-based task assignment."""

    Medic = "medic"
    Engineer = "engineer"
