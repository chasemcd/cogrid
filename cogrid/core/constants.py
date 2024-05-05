import dataclasses

import numpy as np


@dataclasses.dataclass
class CoreConstants:
    TilePixels = 32


@dataclasses.dataclass
class ObjectColors:
    Wall = "grey"
    Door = "dark_grey"
    Floor = "pale_blue"
    Rubble = "brown"
    Key = "yellow"
    MedKit = "light_pink"
    Pickaxe = "light_brown"
    GreenVictim = "green"
    PurpleVictim = "purple"
    YellowVictim = "yellow"
    RedVictim = "red"
    Counter = "light_brown"

    AgentOne = "cyan"
    AgentTwo = "lavender"
    AgentThree = "yellow_grey"
    AgentFour = "puke_green"


@dataclasses.dataclass
class Colors:
    Red = "red"
    Green = "green"
    Blue = "blue"
    Purple = "purple"
    Yellow = "yellow"
    Grey = "grey"
    DarkGrey = "dark_grey"
    LightBrown = "light_brown"
    Brown = "brown"
    Cyan = "cyan"
    Lavender = "lavender"
    YellowGrey = "yellow_grey"
    PukeGreen = "puke_green"
    LightPink = "light_pink"
    Orange = "orange"
    PaleBlue = "pale_blue"
    White = "white"
    Black = "black"


# Map of color names to RGB values
COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
    "dark_grey": np.array([70, 70, 70]),
    "light_brown": np.array([181, 132, 0]),
    "brown": np.array([143, 106, 35]),
    "cyan": np.array([0, 234, 255]),
    "lavender": np.array([220, 185, 237]),
    "yellow_grey": np.array([231, 233, 185]),
    "puke_green": np.array([191, 155, 0]),
    "light_pink": np.array([237, 185, 185]),
    "orange": np.array([245, 130, 200]),
    "pale_blue": np.array([0, 0, 127]),
    "white": np.array([255, 255, 255]),
    "black": np.array([0, 0, 0]),
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS.keys())}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))
