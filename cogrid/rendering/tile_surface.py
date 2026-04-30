"""Per-cell drawing adapter over a frame-wide ``mug.Surface``.

CoGrid object renders work in 0-1 cell-local coordinates. ``TileSurface``
translates those into canvas pixel space and forwards to the parent
Surface, so a single env-level Surface accumulates draw commands from
every cell and ``commit()`` produces the wire packet for the whole frame.

Persistent ids are namespaced by an ``id_prefix`` (cell row/col by default)
so two cells that draw the same logical sub-id (e.g. an inventory item)
do not collide in the parent's persistent-object cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mug.rendering import Surface

Number = float | int
Color = tuple[int, int, int] | str
Point = tuple[Number, Number]


class TileSurface:
    """Sub-rectangle view over a ``mug.Surface`` with 0-1 local coords."""

    def __init__(
        self,
        parent: Surface,
        *,
        x_offset: float,
        y_offset: float,
        width: float,
        height: float,
        id_prefix: str = "",
    ) -> None:
        """Create a sub-region adapter over *parent*.

        :param parent: Frame-wide ``mug.Surface`` receiving the forwarded calls.
        :param x_offset: Canvas-pixel x of this region's top-left corner.
        :param y_offset: Canvas-pixel y of this region's top-left corner.
        :param width: Canvas-pixel width of the region.
        :param height: Canvas-pixel height of the region.
        :param id_prefix: Prepended to any persistent ``id`` to namespace it.
        """
        self.parent = parent
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.width = width
        self.height = height
        self.id_prefix = id_prefix

    @classmethod
    def for_cell(
        cls,
        parent: Surface,
        *,
        row: int,
        col: int,
        tile_size: int,
    ) -> TileSurface:
        """Build a ``TileSurface`` covering the (row, col) cell of a grid."""
        return cls(
            parent,
            x_offset=col * tile_size,
            y_offset=row * tile_size,
            width=tile_size,
            height=tile_size,
            id_prefix=f"{row}-{col}-",
        )

    def subregion(
        self,
        *,
        x: Number,
        y: Number,
        w: Number,
        h: Number,
        id_prefix: str = "",
    ) -> TileSurface:
        """Return a child ``TileSurface`` for a sub-rect of this region.

        Coordinates are 0-1 fractions of *this* surface (not the canvas).
        """
        return TileSurface(
            self.parent,
            x_offset=self.x_offset + x * self.width,
            y_offset=self.y_offset + y * self.height,
            width=w * self.width,
            height=h * self.height,
            id_prefix=self.id_prefix + id_prefix,
        )

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _scale_x(self, v: Number, *, relative: bool) -> float:
        return v * self.width if relative else v

    def _scale_y(self, v: Number, *, relative: bool) -> float:
        return v * self.height if relative else v

    def _scale_diag(self, v: Number, *, relative: bool) -> float:
        # Match mug.Surface circle/arc convention: radius is normalized by
        # max(width, height) on the wire side, so we do the same here.
        return v * max(self.width, self.height) if relative else v

    def _x(self, x: Number, *, relative: bool) -> float:
        return self.x_offset + self._scale_x(x, relative=relative)

    def _y(self, y: Number, *, relative: bool) -> float:
        return self.y_offset + self._scale_y(y, relative=relative)

    def _id(self, id: str | None) -> str | None:
        if id is None:
            return None
        return f"{self.id_prefix}{id}"

    # ------------------------------------------------------------------
    # Draw methods (signatures mirror mug.rendering.Surface)
    # ------------------------------------------------------------------

    def rect(
        self,
        *,
        x: Number,
        y: Number,
        w: Number,
        h: Number,
        color: Color = "white",
        border_radius: Number | None = None,
        stroke_color: Color | None = None,
        stroke_width: Number | None = None,
        id: str | None = None,
        persistent: bool = False,
        relative: bool = True,
        depth: int = 0,
        tween_duration: Number | None = None,
    ) -> None:
        """Draw a rectangle in cell-local coords."""
        kwargs: dict = {
            "x": self._x(x, relative=relative),
            "y": self._y(y, relative=relative),
            "w": self._scale_x(w, relative=relative),
            "h": self._scale_y(h, relative=relative),
            "color": color,
            "id": self._id(id),
            "persistent": persistent,
            "relative": False,
            "depth": depth,
            "tween_duration": tween_duration,
        }
        if border_radius is not None:
            kwargs["border_radius"] = border_radius
        if stroke_color is not None:
            kwargs["stroke_color"] = stroke_color
        if stroke_width is not None:
            kwargs["stroke_width"] = stroke_width
        self.parent.rect(**kwargs)

    def circle(
        self,
        *,
        x: Number,
        y: Number,
        radius: Number,
        color: Color = "white",
        stroke_color: Color | None = None,
        stroke_width: Number | None = None,
        id: str | None = None,
        persistent: bool = False,
        relative: bool = True,
        depth: int = 0,
        tween_duration: Number | None = None,
    ) -> None:
        """Draw a circle (center-origin) in cell-local coords."""
        kwargs: dict = {
            "x": self._x(x, relative=relative),
            "y": self._y(y, relative=relative),
            "radius": self._scale_diag(radius, relative=relative),
            "color": color,
            "id": self._id(id),
            "persistent": persistent,
            "relative": False,
            "depth": depth,
            "tween_duration": tween_duration,
        }
        if stroke_color is not None:
            kwargs["stroke_color"] = stroke_color
        if stroke_width is not None:
            kwargs["stroke_width"] = stroke_width
        self.parent.circle(**kwargs)

    def line(
        self,
        *,
        points: list[Point],
        color: Color = "white",
        width: int = 1,
        id: str | None = None,
        persistent: bool = False,
        relative: bool = True,
        depth: int = 0,
        tween_duration: Number | None = None,
    ) -> None:
        """Draw a multi-segment line in cell-local coords."""
        translated = [
            (self._x(px, relative=relative), self._y(py, relative=relative)) for px, py in points
        ]
        self.parent.line(
            points=translated,
            color=color,
            width=width,
            id=self._id(id),
            persistent=persistent,
            relative=False,
            depth=depth,
            tween_duration=tween_duration,
        )

    def polygon(
        self,
        *,
        points: list[Point],
        color: Color = "white",
        stroke_color: Color | None = None,
        stroke_width: Number | None = None,
        id: str | None = None,
        persistent: bool = False,
        relative: bool = True,
        depth: int = 0,
        tween_duration: Number | None = None,
    ) -> None:
        """Draw a filled polygon in cell-local coords."""
        translated = [
            (self._x(px, relative=relative), self._y(py, relative=relative)) for px, py in points
        ]
        kwargs: dict = {
            "points": translated,
            "color": color,
            "id": self._id(id),
            "persistent": persistent,
            "relative": False,
            "depth": depth,
            "tween_duration": tween_duration,
        }
        if stroke_color is not None:
            kwargs["stroke_color"] = stroke_color
        if stroke_width is not None:
            kwargs["stroke_width"] = stroke_width
        self.parent.polygon(**kwargs)

    def text(
        self,
        *,
        text: str,
        x: Number,
        y: Number,
        size: int = 16,
        color: Color = "black",
        font: str = "Arial",
        id: str | None = None,
        persistent: bool = False,
        relative: bool = True,
        depth: int = 0,
        tween_duration: Number | None = None,
    ) -> None:
        """Draw a text label.

        ``size`` is a font size in pixels and is *not* scaled by the cell.
        """
        self.parent.text(
            text=text,
            x=self._x(x, relative=relative),
            y=self._y(y, relative=relative),
            size=size,
            color=color,
            font=font,
            id=self._id(id),
            persistent=persistent,
            relative=False,
            depth=depth,
            tween_duration=tween_duration,
        )

    def image(
        self,
        *,
        image_name: str,
        x: Number,
        y: Number,
        w: Number,
        h: Number,
        frame: str | int | None = None,
        angle: Number | None = None,
        id: str | None = None,
        persistent: bool = False,
        relative: bool = True,
        depth: int = 0,
        tween_duration: Number | None = None,
    ) -> None:
        """Draw a sprite image in cell-local coords."""
        kwargs: dict = {
            "image_name": image_name,
            "x": self._x(x, relative=relative),
            "y": self._y(y, relative=relative),
            "w": self._scale_x(w, relative=relative),
            "h": self._scale_y(h, relative=relative),
            "id": self._id(id),
            "persistent": persistent,
            "relative": False,
            "depth": depth,
            "tween_duration": tween_duration,
        }
        if frame is not None:
            kwargs["frame"] = frame
        if angle is not None:
            kwargs["angle"] = angle
        self.parent.image(**kwargs)

    def arc(
        self,
        *,
        x: Number,
        y: Number,
        radius: Number,
        start_angle: float,
        end_angle: float,
        color: Color = "white",
        id: str | None = None,
        persistent: bool = False,
        relative: bool = True,
        depth: int = 0,
        tween_duration: Number | None = None,
    ) -> None:
        """Draw an arc in cell-local coords."""
        self.parent.arc(
            x=self._x(x, relative=relative),
            y=self._y(y, relative=relative),
            radius=self._scale_diag(radius, relative=relative),
            start_angle=start_angle,
            end_angle=end_angle,
            color=color,
            id=self._id(id),
            persistent=persistent,
            relative=False,
            depth=depth,
            tween_duration=tween_duration,
        )

    def ellipse(
        self,
        *,
        x: Number,
        y: Number,
        rx: Number,
        ry: Number,
        color: Color = "white",
        id: str | None = None,
        persistent: bool = False,
        relative: bool = True,
        depth: int = 0,
        tween_duration: Number | None = None,
    ) -> None:
        """Draw an ellipse in cell-local coords."""
        self.parent.ellipse(
            x=self._x(x, relative=relative),
            y=self._y(y, relative=relative),
            rx=self._scale_x(rx, relative=relative),
            ry=self._scale_y(ry, relative=relative),
            color=color,
            id=self._id(id),
            persistent=persistent,
            relative=False,
            depth=depth,
            tween_duration=tween_duration,
        )
