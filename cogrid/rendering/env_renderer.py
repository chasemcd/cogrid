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
        self.name = name
        self.screen_size = screen_size
        self.render_fps = render_fps

        self.window = None
        self.clock = None
        self.render_size = None

    def render_human(
        self,
        img: np.ndarray,
        cumulative_score: float,
        render_message: str,
    ) -> None:
        """Display a frame in the PyGame window.

        Lazily initialises PyGame, the display window, and the clock on the
        first call.  Subsequent calls blit the provided image, overlay score
        and message text, and tick the clock.

        Parameters
        ----------
        img : np.ndarray
            RGB image array (H, W, 3) to display.
        cumulative_score : float
            Cumulative episode score shown in the overlay.
        render_message : str
            Additional message appended after the score text.
        """
        if pygame is None:
            raise ImportError(
                "Must install pygame to use interactive mode."
            )

        if self.render_size is None:
            self.render_size = img.shape[:2]
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.screen_size, self.screen_size)
            )
            pygame.display.set_caption(self.name)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.surfarray.make_surface(img)

        # Correct orientation (pygame flips/rotates the array)
        surf = pygame.transform.flip(surf, False, True)
        surf = pygame.transform.rotate(surf, 270)

        # Create background with score/message overlay
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

        bg = pygame.transform.smoothscale(
            bg, (self.screen_size, self.screen_size)
        )

        font_size = 22
        text = (
            f"Score: {np.round(cumulative_score, 2)}"
            + render_message
        )

        font = pygame.freetype.SysFont(
            pygame.font.get_default_font(), font_size
        )
        text_rect = font.get_rect(text, size=font_size)
        text_rect.center = bg.get_rect().center
        text_rect.y = bg.get_height() - font_size * 1.5
        font.render_to(bg, text_rect, text, size=font_size)

        self.window.blit(bg, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.render_fps)
        pygame.display.update()

    def close(self) -> None:
        """Shut down the PyGame display and quit PyGame."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.render_size = None
