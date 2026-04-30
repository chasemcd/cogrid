"""EnvRenderer: PyGame window management for CoGridEnv rendering.

Encapsulates all PyGame-specific display logic (window creation, surface
blitting, font rendering, event pumping) so that cogrid_env.py remains
free of any PyGame dependency.
"""

from __future__ import annotations

import numpy as np

try:
    import pygame
    import pygame.freetype
except ImportError:
    pygame = None


# Overlay dicts passed to render_human are one of the following shapes.
# Coords are in screen-pixel space (post-scale; 0..screen_size on each axis).
#
#   {"type": "text", "text": str, "x": int, "y": int, "size": int = 22,
#    "color": tuple[int, int, int] = (255, 255, 255), "font": str | None = None,
#    "anchor": "center" | "topleft" | "midbottom" = "center"}
#
#   {"type": "rect", "x": int, "y": int, "w": int, "h": int,
#    "color": tuple[int, int, int], "border_width": int = 0}


class EnvRenderer:
    """Manages a PyGame window for human-mode rendering.

    Parameters
    ----------
    name : str
        Window title (passed to ``pygame.display.set_caption``).
    screen_size : int
        Width and height of the display window in pixels.
    render_fps : int
        Target frames-per-second for ``pygame.time.Clock.tick``.
    """

    def __init__(self, name: str, screen_size: int, render_fps: int) -> None:
        """Initialize the renderer with name, screen size, and FPS."""
        self.name = name
        self.screen_size = screen_size
        self.render_fps = render_fps

        self.window = None
        self.clock = None
        self.render_size: tuple[int, int] | None = None

    def render_human(
        self,
        img: np.ndarray,
        overlays: list[dict] | None = None,
    ) -> None:
        """Display a frame in the PyGame window.

        Lazily initialises PyGame, the display window, and the clock on the
        first call. Subsequent calls blit *img*, draw any post-scale
        *overlays*, and tick the clock.

        Parameters
        ----------
        img : np.ndarray
            RGB image array (H, W, 3) to display. The image is expected to
            already contain any in-canvas content (HUD strip, grid, agents);
            *overlays* are drawn on top of the scaled output, at native
            resolution so text stays crisp.
        overlays : list[dict] or None
            Arbitrary post-scale draw specs. See module docstring for the
            supported dict shapes.
        """
        if pygame is None:
            raise ImportError("Must install pygame to use interactive mode.")

        if self.render_size is None:
            self.render_size = img.shape[:2]
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption(self.name)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.surfarray.make_surface(img)

        # Correct orientation (pygame flips/rotates the array)
        surf = pygame.transform.flip(surf, False, True)
        surf = pygame.transform.rotate(surf, 270)

        # Pad horizontally and along the bottom so overlays have room.
        offset = surf.get_size()[0] * 0.1
        bg = pygame.Surface(
            (
                int(surf.get_size()[0] + offset),
                int(surf.get_size()[1] + offset),
            )
        )
        bg.convert()
        bg.fill((255, 255, 255))
        bg.blit(surf, (offset / 2, 0))

        bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

        if overlays:
            for ov in overlays:
                self._draw_overlay(bg, ov)

        self.window.blit(bg, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.render_fps)
        pygame.display.update()

    def _draw_overlay(self, target: pygame.Surface, overlay: dict) -> None:
        """Draw a single overlay spec onto *target*."""
        kind = overlay.get("type")
        if kind == "text":
            text = str(overlay["text"])
            size = int(overlay.get("size", 22))
            color = overlay.get("color", (255, 255, 255))
            font_name = overlay.get("font") or pygame.font.get_default_font()
            font = pygame.freetype.SysFont(font_name, size)
            rect = font.get_rect(text, size=size)
            anchor = overlay.get("anchor", "center")
            x = int(overlay.get("x", target.get_width() // 2))
            y = int(overlay.get("y", target.get_height() // 2))
            if anchor == "center":
                rect.center = (x, y)
            elif anchor == "topleft":
                rect.topleft = (x, y)
            elif anchor == "midbottom":
                rect.midbottom = (x, y)
            else:
                raise ValueError(f"Unknown text anchor: {anchor!r}")
            font.render_to(target, rect, text, color, size=size)
        elif kind == "rect":
            rect = pygame.Rect(
                int(overlay["x"]),
                int(overlay["y"]),
                int(overlay["w"]),
                int(overlay["h"]),
            )
            color = overlay.get("color", (0, 0, 0))
            border_width = int(overlay.get("border_width", 0))
            pygame.draw.rect(target, color, rect, border_width)
        else:
            raise ValueError(f"Unknown overlay type: {kind!r}")

    def close(self) -> None:
        """Shut down the PyGame display and quit PyGame."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.render_size = None
