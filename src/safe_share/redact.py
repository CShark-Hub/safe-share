"""Opaque redaction of detected regions on a BGRA frame.

Covers the full OCR-line box of every line that has a detection (whole-line is
the reliable unit; sub-line interpolation could under-cover and leak). The
`[PRIVATE]` label is drawn only after the pixels are covered and only when it
fits inside the covered rectangle, so it never touches raw pixels.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detect import Detection
from .types import OcrLine

_LABEL = "[PRIVATE]"
_PAD_FRAC = 0.15
_PAD_MIN = 4
_MIN_LABEL_H = 14
_LABEL_MARGIN = 6
Box = tuple[int, int, int, int]


def redact(
    frame: np.ndarray,
    lines: list[OcrLine],
    detections: list[Detection],
    draw_label: bool = True,
) -> np.ndarray:
    """Cover every detected line's box on `frame` (mutated in place)."""
    h, w = frame.shape[:2]
    hit = {d.line_index for d in detections if 0 <= d.line_index < len(lines)}
    boxes = [_pad(lines[i].box, w, h) for i in hit]
    for x0, y0, x1, y1 in boxes:
        frame[y0:y1, x0:x1, 0:3] = 0
        frame[y0:y1, x0:x1, 3] = 255
    if draw_label:
        for box in boxes:
            _draw_label(frame, box)
    return frame


def _pad(box: Box, w: int, h: int) -> Box:
    x0, y0, x1, y1 = box
    pad = max(_PAD_MIN, round((y1 - y0) * _PAD_FRAC))
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(w, x1 + pad),
        min(h, y1 + pad),
    )


def _draw_label(frame: np.ndarray, box: Box) -> None:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bh < _MIN_LABEL_H:
        return
    font = ImageFont.load_default(size=min(bh * 0.7, 28.0))
    l, t, r, b = font.getbbox(_LABEL)
    tw, th = r - l, b - t
    if tw > bw - _LABEL_MARGIN:
        return  # omit rather than spill past the covered pixels
    sub = np.ascontiguousarray(frame[y0:y1, x0:x1])
    img = Image.fromarray(sub, "RGBA")
    draw = ImageDraw.Draw(img)
    draw.text(
        ((bw - tw) // 2 - l, (bh - th) // 2 - t),
        _LABEL,
        font=font,
        fill=(200, 200, 200, 255),
    )
    frame[y0:y1, x0:x1] = np.asarray(img)
