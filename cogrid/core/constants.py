"""Core rendering and color constants."""

import dataclasses


@dataclasses.dataclass
class CoreConstants:
    """Core rendering constants."""

    TilePixels = 32


@dataclasses.dataclass
class Colors:
    """RGB color palette for grid objects."""

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
