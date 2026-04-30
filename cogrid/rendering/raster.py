"""Pygame rasterizer for ``mug.Surface`` wire packets.

``PygameRenderer`` consumes the wire-format dicts produced by
``Surface.commit().to_dict()`` and draws them onto a ``pygame.Surface``.
It maintains a uuid-keyed scene graph for persistent objects so that
incremental commits (which only retransmit new/changed persistents)
compose into the full frame; ephemerals are scoped to the most recent
``apply()`` call.

Shapes only in this version: rect, circle, line, polygon, text, arc,
ellipse. Sprite/image objects are skipped with a one-time warning per
``image_name``.
"""

from __future__ import annotations

import logging

import numpy as np

try:
    import pygame
    import pygame.freetype
except ImportError:
    pygame = None

logger = logging.getLogger(__name__)


class PygameRenderer:
    """Stateful pygame-backed rasterizer for wire-format render packets."""

    def __init__(
        self,
        width: int,
        height: int,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        """Create a renderer for a *width* × *height* pixel canvas."""
        if pygame is None:
            raise ImportError("PygameRenderer requires pygame. Install with `pip install pygame`.")
        self.width = width
        self.height = height
        self.background_color = background_color
        self._persistent: dict[str, dict] = {}
        self._frame_ephemerals: list[dict] = []
        self._missing_images: set[str] = set()
        self._font_cache: dict[tuple[str, int], pygame.freetype.Font] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, packet: dict) -> None:
        """Update internal scene from a ``RenderPacket.to_dict()`` payload.

        Persistent objects accumulate across calls; ephemerals from prior
        calls are discarded when a new packet arrives.
        """
        self._frame_ephemerals = []
        for obj in packet.get("game_state_objects", []):
            if obj.get("permanent"):
                self._persistent[obj["uuid"]] = obj
            else:
                self._frame_ephemerals.append(obj)
        for uuid in packet.get("removed", []):
            self._persistent.pop(uuid, None)

    def reset(self) -> None:
        """Drop all persistent and ephemeral state."""
        self._persistent.clear()
        self._frame_ephemerals.clear()

    def to_pygame_surface(self) -> pygame.Surface:
        """Render the current scene to a fresh ``pygame.Surface``."""
        surf = pygame.Surface((self.width, self.height))
        surf.fill(self.background_color)
        objects = list(self._persistent.values()) + self._frame_ephemerals
        objects.sort(key=lambda o: o.get("depth", 0))
        for obj in objects:
            self._draw(surf, obj)
        return surf

    def to_array(self) -> np.ndarray:
        """Render the current scene and return an HxWx3 RGB ``np.ndarray``."""
        surf = self.to_pygame_surface()
        # surfarray.array3d gives (W, H, 3); transpose to (H, W, 3).
        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _px(self, v: float) -> int:
        return int(round(v * self.width))

    def _py(self, v: float) -> int:
        return int(round(v * self.height))

    def _pdiag(self, v: float) -> int:
        return int(round(v * max(self.width, self.height)))

    def _color(self, c: str) -> pygame.Color:
        # Wire format normalizes colors to '#rrggbb'; pygame.Color parses that.
        return pygame.Color(c)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _draw(self, surf: pygame.Surface, obj: dict) -> None:
        kind = obj.get("object_type")
        handler = getattr(self, f"_draw_{kind}", None)
        if handler is None:
            logger.warning("PygameRenderer: unknown object_type %r, skipping", kind)
            return
        handler(surf, obj)

    def _draw_rect(self, surf: pygame.Surface, obj: dict) -> None:
        rect = pygame.Rect(
            self._px(obj["x"]),
            self._py(obj["y"]),
            self._px(obj["w"]),
            self._py(obj["h"]),
        )
        border_radius = obj.get("border_radius")
        kwargs = {}
        if border_radius is not None:
            kwargs["border_radius"] = int(round(border_radius))
        pygame.draw.rect(surf, self._color(obj["color"]), rect, 0, **kwargs)
        sw = obj.get("stroke_width")
        if sw is not None and obj.get("stroke_color") is not None:
            pygame.draw.rect(
                surf,
                self._color(obj["stroke_color"]),
                rect,
                int(round(sw)),
                **kwargs,
            )

    def _draw_circle(self, surf: pygame.Surface, obj: dict) -> None:
        center = (self._px(obj["x"]), self._py(obj["y"]))
        radius = self._pdiag(obj["radius"])
        pygame.draw.circle(surf, self._color(obj["color"]), center, radius)
        sw = obj.get("stroke_width")
        if sw is not None and obj.get("stroke_color") is not None:
            pygame.draw.circle(
                surf,
                self._color(obj["stroke_color"]),
                center,
                radius,
                width=int(round(sw)),
            )

    def _draw_line(self, surf: pygame.Surface, obj: dict) -> None:
        points = [(self._px(px), self._py(py)) for px, py in obj["points"]]
        if len(points) < 2:
            return
        width = int(round(obj.get("width", 1)))
        pygame.draw.lines(
            surf,
            self._color(obj["color"]),
            False,
            points,
            max(1, width),
        )

    def _draw_polygon(self, surf: pygame.Surface, obj: dict) -> None:
        points = [(self._px(px), self._py(py)) for px, py in obj["points"]]
        if len(points) < 3:
            return
        pygame.draw.polygon(surf, self._color(obj["color"]), points)
        sw = obj.get("stroke_width")
        if sw is not None and obj.get("stroke_color") is not None:
            pygame.draw.polygon(
                surf,
                self._color(obj["stroke_color"]),
                points,
                width=int(round(sw)),
            )

    def _draw_text(self, surf: pygame.Surface, obj: dict) -> None:
        font = self._get_font(obj.get("font", "Arial"), int(obj.get("size", 16)))
        text = str(obj.get("text", ""))
        color = self._color(obj.get("color", "#000000"))
        font.render_to(surf, (self._px(obj["x"]), self._py(obj["y"])), text, color)

    def _draw_arc(self, surf: pygame.Surface, obj: dict) -> None:
        cx = self._px(obj["x"])
        cy = self._py(obj["y"])
        r = self._pdiag(obj["radius"])
        rect = pygame.Rect(cx - r, cy - r, 2 * r, 2 * r)
        pygame.draw.arc(
            surf,
            self._color(obj.get("color", "#ffffff")),
            rect,
            float(obj["start_angle"]),
            float(obj["end_angle"]),
        )

    def _draw_ellipse(self, surf: pygame.Surface, obj: dict) -> None:
        cx = self._px(obj["x"])
        cy = self._py(obj["y"])
        rx = self._px(obj["rx"])
        ry = self._py(obj["ry"])
        rect = pygame.Rect(cx - rx, cy - ry, 2 * rx, 2 * ry)
        pygame.draw.ellipse(surf, self._color(obj.get("color", "#ffffff")), rect)

    def _draw_sprite(self, surf: pygame.Surface, obj: dict) -> None:
        # Sprite/image rendering deferred; track names so we warn once each.
        name = obj.get("image_name", "<unnamed>")
        if name not in self._missing_images:
            self._missing_images.add(name)
            logger.warning(
                "PygameRenderer: sprite/image rendering is not implemented; "
                "object with image_name=%r will not be drawn locally.",
                name,
            )

    # ------------------------------------------------------------------
    # Font cache
    # ------------------------------------------------------------------

    def _get_font(self, font_name: str, size: int) -> pygame.freetype.Font:
        if not pygame.freetype.was_init():
            pygame.freetype.init()
        key = (font_name, size)
        cached = self._font_cache.get(key)
        if cached is not None:
            return cached
        try:
            font = pygame.freetype.SysFont(font_name, size)
        except Exception:
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), size)
        self._font_cache[key] = font
        return font
