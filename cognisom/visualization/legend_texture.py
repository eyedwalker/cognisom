"""Bake a legend into a PNG so the render explains itself.

A frame full of coloured glyphs is a Rorschach test unless something
says what the glyphs mean. Core USD has no text primitive, so the only
way to get words into the *stage* -- not merely into a post-processed
image -- is to draw them into a texture and map it onto a quad. That is
what this module produces; :func:`interaction_usd.author_legend` maps it
onto a camera-parented quad so it holds a screen corner, and it renders
in Apple's usdrecord (Storm/Metal); verified.

matplotlib is the drawing surface because it is already a hard
dependency of this repo (requirements.txt) and the only other candidate,
Pillow, is not declared.

The glyph shapes mirror the 3-D marker shapes exactly -- circle/sphere,
square/cube, triangle/cone, bar/disc -- so a viewer matches the key to
the picture by outline, not only by colour. That matters for anyone who
cannot separate the red kill marker from the amber engagement marker.
"""

from __future__ import annotations

import logging
import textwrap
from typing import Any, Dict, List, Sequence, Tuple

log = logging.getLogger(__name__)

#: Marker shape -> matplotlib marker. Keep in step with
#: ``interaction_usd._define_prototype``.
_SHAPE_MARKERS = {
    "sphere": "o",
    "cube": "s",
    "cone": "^",
    "disc": "_",
}

_BACKGROUND = (0.043, 0.051, 0.078)
_FOREGROUND = (0.90, 0.92, 0.96)
_MUTED = (0.60, 0.64, 0.74)


def render_legend_png(
    entries: Sequence[Dict[str, Any]],
    output_path: str,
    title: str = "Cognisom — cell interactions",
    subtitle: str = "",
    footer: str = "",
    oxygen_ramp: bool = True,
    oxygen_caption: str = "Oxygen, per cell (disc under each cell)",
    hypoxia_threshold: float = 0.05,
    arterial: float = 0.21,
    width_px: int = 1100,
    dpi: int = 170,
    footer_wrap: int = 118,
) -> Tuple[str, float]:
    """Draw the key and return ``(path, aspect)``.

    ``aspect`` is width/height of the image, which the caller needs to
    size the quad: a quad with the wrong aspect stretches the text, and
    legibility was the entire point.

    Layout runs top-down on a cursor measured in inches, then converts
    to axes fractions once. Laying it out directly in axes fractions is
    what made the first version overlap its own footer as soon as the
    row count changed.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows = list(entries)
    footer_lines: List[str] = []
    if footer:
        footer_lines = textwrap.wrap(footer, width=footer_wrap) or [footer]

    # Vertical budget, in inches.
    pad_top, pad_bottom = 0.16, 0.14
    title_h = 0.22
    subtitle_h = 0.17 if subtitle else 0.0
    gap_after_head = 0.10
    row_h = 0.22
    ramp_block_h = 0.78 if oxygen_ramp else 0.0
    footer_h = 0.135 * len(footer_lines)

    width_in = width_px / float(dpi)
    height_in = (pad_top + title_h + subtitle_h + gap_after_head
                 + row_h * len(rows) + ramp_block_h
                 + (0.08 if footer_lines else 0.0) + footer_h + pad_bottom)
    height_in = max(0.75, height_in)

    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    fig.patch.set_facecolor(_BACKGROUND)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=_BACKGROUND, zorder=0))

    def y_of(inches_from_top: float) -> float:
        return 1.0 - inches_from_top / height_in

    cursor = pad_top
    ax.text(0.030, y_of(cursor + title_h * 0.45), title, color=_FOREGROUND,
            fontsize=10.5, fontweight="bold", va="center", ha="left", zorder=3)
    cursor += title_h
    if subtitle:
        ax.text(0.030, y_of(cursor + subtitle_h * 0.45), subtitle,
                color=_MUTED, fontsize=7.0, va="center", ha="left", zorder=3)
        cursor += subtitle_h
    cursor += gap_after_head

    for row in rows:
        marker = _SHAPE_MARKERS.get(row.get("shape", "sphere"), "o")
        color = tuple(float(c) for c in row.get("color", (1, 1, 1)))
        y = y_of(cursor + row_h * 0.5)
        if marker == "_":
            # An unfilled matplotlib marker takes its colour from the
            # line colour, not the face colour; passing edgecolors here
            # warns and is ignored.
            ax.scatter([0.050], [y], marker=marker, s=240,
                       c=[color], linewidths=3.0, zorder=3)
        else:
            ax.scatter([0.050], [y], marker=marker, s=135,
                       c=[color], edgecolors="none", zorder=3)
        ax.text(0.090, y, str(row.get("label", "")), color=_FOREGROUND,
                fontsize=8.0, va="center", ha="left", zorder=3)
        cursor += row_h

    if oxygen_ramp:
        from .interaction_usd import oxygen_color
        cursor += 0.06
        ax.text(0.030, y_of(cursor + 0.09), oxygen_caption,
                color=_FOREGROUND, fontsize=7.5, va="center", ha="left",
                zorder=3)
        cursor += 0.38
        bar_x0, bar_x1 = 0.050, 0.46
        bar_y = y_of(cursor + 0.085)
        bar_h = 0.085 / height_in
        n = 128
        for i in range(n):
            value = (i / float(n - 1)) * arterial
            x0 = bar_x0 + (bar_x1 - bar_x0) * i / n
            ax.add_patch(Rectangle(
                (x0, bar_y - bar_h / 2), (bar_x1 - bar_x0) / n * 1.05, bar_h,
                facecolor=oxygen_color(value, hypoxia_threshold, arterial),
                edgecolor="none", zorder=3))
        xt = bar_x0 + (bar_x1 - bar_x0) * (hypoxia_threshold / arterial)
        ax.plot([xt, xt], [bar_y - bar_h * 1.6, bar_y + bar_h * 1.6],
                color=_FOREGROUND, lw=1.0, zorder=4)
        # Above the bar, not across it: printed over the ramp the label
        # sat on the colours it was labelling.
        ax.text(xt, bar_y + bar_h * 2.0, f"hypoxic < {hypoxia_threshold:g}",
                color=_FOREGROUND, fontsize=6.0, ha="center", va="bottom",
                zorder=4)
        ax.text(bar_x1 + 0.012, bar_y, f"{arterial:g} arterial",
                color=_MUTED, fontsize=6.0, ha="left", va="center", zorder=4)
        cursor += 0.34

    if footer_lines:
        cursor += 0.08
        for line in footer_lines:
            ax.text(0.030, y_of(cursor + 0.0675), line, color=_MUTED,
                    fontsize=5.6, va="center", ha="left", zorder=3)
            cursor += 0.135

    fig.savefig(output_path, facecolor=_BACKGROUND, dpi=dpi)
    plt.close(fig)
    return output_path, float(width_in / height_in)
