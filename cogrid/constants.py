import dataclasses

import numpy as np

"""
Define all constants for CoGridEnv environments.
"""


@dataclasses.dataclass
class GridConstants:
    Wall = "#"
    Door = "D"
    Counter = "C"
    FreeSpace = " "
    Padding = "0"
    Obscured = "."
    Spawn = "S"
    Key = "K"
    Rubble = "X"
    Pickaxe = "*"
    MedKit = "M"
    GreenVictim = "G"
    YellowVictim = "Y"
    Pot = "P"
    Plate = "]"
    PlateStack = "["
    OnionSoup = "="
    OnionStack = "+"
    Onion = "&"
    DeliveryZone = "@"
    RedVictim = "R"
    AgentUp = "^"
    AgentRight = ">"
    AgentDown = "v"
    AgentLeft = "<"


FIXED_GRIDS = {
    "sa_overcooked": (
        [
            "#########",
            "#CCCPCCC#",
            "#C     C#",
            "#C     C#",
            "#+     [#",
            "#C     C#",
            "#C     C#",
            "#C     C#",
            "#CCCCC@C#",
            "#########",
        ],
        np.zeros((10, 9)),
    ),
    "overcooked-v0": (
        [
            "#########",
            "#CCCPCCC#",
            "#C     C#",
            "#C     C#",
            "#+     [#",
            "#C     C#",
            "#C     C#",
            "#C     C#",
            "#CCCCC@C#",
            "#########",
        ],
        np.zeros((10, 9)),
    ),
    "overcooked-crampedroom-v0": (
        [
            "#######",
            "#CCPCC#",
            "#+   +#",
            "#C   C#",
            "#C[C@C#",
            "#######",
        ],
        np.zeros((6, 7)),
    ),
    "m3minimap": (
        [
            "#############",
            "# S  #      #",
            "#  X #      #",
            "# XR #XYXG  #",
            "#    # X  XY#",
            "#   X#     X#",
            "#  XY#      #",
            "# S  #  RX  #",
            "##X###  X   #",
            "#    #     X#",
            "#    #G   XR#",
            "#  X #     X#",
            "# XYX#    # #",
            "#    ###X# ##",
            "#G   #    G #",
            "#    X     G#",
            "#    #X     #",
            "#    #YXX # #",
            "# X  #X RX ##",
            "# RX #  #####",
            "#   #########",
            "#############",
        ],
        np.zeros((22, 13)),
    ),
    "test_redvictim": (
        [
            "#######",
            "#######",
            "#######",
            "#MSRSM#",
            "#######",
            "#######",
            "#######",
        ],
        np.zeros((7, 7)),
    ),
    "test_key": (
        [
            "#######",
            "#SK####",
            "#D#####",
            "#GGDSK#",
            "#######",
            "#######",
            "#######",
        ],
        np.zeros((7, 7)),
    ),
    "item_map": (
        [
            "############",
            "#S S  #GMGP#",
            "#K    D   X#",
            "#XX   #XXXY#",
            "#RX   #  GX#",
            "#XYX  # G G#",
            "#######D####",
            "# X R    R #",
            "#XRX  Y  G #",
            "#RX G   X G#",
            "#XYX Y XYXG#",
            "############",
        ],
        np.zeros((12, 12)),
    ),
    "item_map_nored": (
        [
            "############",
            "#S S  #GMGP#",
            "#K    D   X#",
            "#XX   #XXXY#",
            "#YX   #  GX#",
            "#XYX  # G G#",
            "#######D####",
            "# X Y    Y #",
            "#XYX  Y  G #",
            "#YX G   X G#",
            "#XYX Y XYXG#",
            "############",
        ],
        np.zeros((12, 12)),
    ),
    "small_test_map": (
        [
            "##########",
            "#SS      #",
            "#        #",
            "#        #",
            "#       G#",
            "#        #",
            "#*     K #",
            "#XX M ##D#",
            "#GX Y #GG#",
            "##########",
        ],
        np.zeros((10, 10)),
    ),
    "test_counter": (
        [
            "##########",
            "#SS      #",
            "#   C    #",
            "#   C    #",
            "#   C   G#",
            "#        #",
            "#P     K #",
            "#XX M ##D#",
            "#GX Y #GG#",
            "##########",
        ],
        np.zeros((10, 10)),
    ),
}
