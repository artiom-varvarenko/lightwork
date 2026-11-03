"""How the projector page draws C2 primitives, for code that must predict the projected pixels (B's overlay mask).

Mirrors ``web/js/projector.js`` and ``web/js/scene.js`` (``tests/vision_gate/test_overlay_mask.py`` keeps them equal).
Sizes are in projector px; one projector px is ``unit_css_px(page_w, page_h)`` page CSS px. Geometry is in camera px,
mapped point by point through ``H_cam_to_page`` (rings as ``RING_SEGMENTS``-gons of mapped circle points).
"""

from __future__ import annotations

PROJ_W, PROJ_H = 854, 480  # DLP native resolution
LINE_PROJ_PX = 5  # stroke width of ring/arrow/poly, round caps and joins
TEXT_PROJ_PX = 28  # bold sans; label `at` = left edge, vertical middle of the text
PULSE_HZ = 2  # pulsing rings: stroke width x (1 + PULSE_GAIN * (0.5 + 0.5 sin(2 pi PULSE_HZ t))); radius unchanged
PULSE_GAIN = 0.6
DOT_RADIUS_LINES = 1.5  # filled dot radius, in stroke widths
ARROW_HEAD_LINES = 4  # arrowhead side length, in stroke widths
ARROW_HEAD_RAD = 0.5  # half-angle between shaft and each head side
RING_SEGMENTS = 64
GLOW_WIDTH_LINES = 3.2  # glowing ring/poly: a halo stroke this many stroke widths wide under the stroke, same path
GLOW_ALPHA = 0.28  # halo opacity: saturated colour on black reads as a soft glow through the DLP


def unit_css_px(page_w: float, page_h: float) -> float:
    """Page CSS px per projector px."""
    return max(page_w / PROJ_W, page_h / PROJ_H)


def max_stroke_css_px(page_w: float, page_h: float) -> float:
    """Widest stroke ever drawn (the halo of a glowing pulsing ring at its peak), in page CSS px."""
    return LINE_PROJ_PX * (1 + PULSE_GAIN) * GLOW_WIDTH_LINES * unit_css_px(page_w, page_h)
