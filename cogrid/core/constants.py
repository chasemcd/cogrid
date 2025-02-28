import dataclasses

import numpy as np


@dataclasses.dataclass
class CoreConstants:
    TilePixels = 32


@dataclasses.dataclass
class Colors:
    Red = [255, 0, 0]
    Green = [0, 255, 0]
    Blue = [0, 0, 255]
    Purple = [112, 39, 195]
    Yellow = [255, 255, 0]
    Grey = [100, 100, 100]
    DarkGrey = [70, 70, 70]
    LightBrown = [181, 132, 0]
    PukeGreen = [191, 155, 0]
    Brown = [143, 106, 35]
    Cyan = [0, 234, 255]
    Lavender = [220, 185, 237]
    YellowGrey = [231, 233, 185]
    LightPink = [237, 185, 185]
    Orange = [245, 130, 200]
    PaleBlue = [0, 0, 127]
    White = [255, 255, 255]
    Black = [0, 0, 0]


@dataclasses.dataclass
class ObjectColors:

    AgentOne = Colors.Cyan
    AgentTwo = Colors.Lavender
    AgentThree = Colors.YellowGrey
    AgentFour = Colors.PukeGreen
    AgentFive = Colors.LightPink
    AgentSix = Colors.Orange
    AgentSeven = Colors.PaleBlue
    AgentEight = Colors.Blue
    AgentNine = Colors.LightBrown


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
