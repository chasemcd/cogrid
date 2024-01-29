"""
This is based off of the Minihack GlyphMapper:
https://github.com/facebookresearch/minihack/minihack/tiles/glyph_mapper.py
"""

import pickle
import pkg_resources
import os

import numpy as np

class ObjectRenderer:

    def __init__(self):
        self.name_to_rgb: dict[str, np.ndarray] = self._load_rgb_tiles()

    def _load_rgb_tiles(self):
        raise NotImplementedError
