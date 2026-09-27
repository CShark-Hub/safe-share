"""Opaque redaction of detected regions on a BGRA frame.

Covers the full OCR-line box of every line that has a detection (whole-line is
the reliable unit; sub-line interpolation could under-cover and leak). Nearby
lines merge into one rounded pill; every padded line box is painted as well, so
corner rounding can never uncover text. Labels are opaque tiles placed only
after every pill is painted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .detect import CREDENTIAL_TYPES, Detection
from .labels import label_tile
from .types import OcrLine

Box = tuple[int, int, int, int]
RGB = tuple[int, int, int]

_PAD_FRAC = 0.15
_PAD_MIN = 4
_HPAD_FRAC = 0.35
_HPAD_MIN = 8
_MERGE_LEFT = 1.0  # left edges this close (x line height) count as aligned
_GRID = 4  # outward snap of the final rect
_RADIUS_FRAC = 0.3
_RADIUS_MIN, _RADIUS_MAX = 4, 12
_LABEL_FRAC = 0.75
_LABEL_STEP = 4
_LABEL_MAX = 32
_LABEL_SHRINK = 2 / 3  # smallest fitted size, relative to the frame's size
_LABEL_FLOOR = 8
_RING = 3
_DARK_MEDIAN, _DARK_IQR = 80, 40

# (fill, label ink); ON_DARK only on a dark, uniform background.
GRAPHITE: tuple[RGB, RGB] = ((44, 44, 46), (174, 174, 178))
ON_DARK: tuple[RGB, RGB] = ((108, 108, 112), (242, 242, 247))

_LABELS = {"person": "NAME", "email": "EMAIL", "term": "CONFIDENTIAL"}
_LABELS.update(dict.fromkeys(CREDENTIAL_TYPES, "SECRET"))
_GENERIC = "PRIVATE"
# Mixed pills: rule-detected types before model-detected ones.
_PRIORITY = ("SECRET", "EMAIL", "CONFIDENTIAL", "NAME", _GENERIC)


@dataclass(frozen=True)
class Pill:
    rect: Box  # rounded outer shape
    covers: tuple[Box, ...]  # padded line boxes, always painted in full
    first: Box  # raw box of the top line; anchors the label
    label: str
    radius: int
    lines: tuple[int, ...]  # indices of the OCR lines it covers


def redact(
    frame: np.ndarray,
    lines: list[OcrLine],
    detections: list[Detection],
    draw_label: bool = True,
) -> np.ndarray:
    """Cover every detected line on `frame` (mutated in place)."""
    h, w = frame.shape[:2]
    pills = plan(lines, detections, w, h)
    rects = [p.rect for p in pills]
    tones = [ON_DARK if _is_dark(frame, p.rect, rects) else GRAPHITE for p in pills]
    for pill, (fill, _) in zip(pills, tones, strict=True):
        _paint(frame, pill, fill)
    if draw_label and pills:
        px = label_px(lines)
        for pill, (fill, ink) in zip(pills, tones, strict=True):
            if (fit := fit_label(pill, px, fill, ink)) is not None:
                tile, x, y = fit
                frame[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return frame


def plan(
    lines: list[OcrLine], detections: list[Detection], w: int, h: int
) -> list[Pill]:
    """Pill geometry for the detected lines of a `w` x `h` frame; no pixels read.

    A merge never makes a pill contain the centre of an unflagged line.
    """
    types: dict[int, set[str]] = {}
    for d in detections:
        if 0 <= d.line_index < len(lines):
            types.setdefault(d.line_index, set()).add(d.entity_type)
    hit = sorted(types)
    others = [
        ((x0 + x1) / 2, (y0 + y1) / 2)
        for i, ln in enumerate(lines)
        if i not in types
        for x0, y0, x1, y1 in [ln.box]
    ]

    def build(group: list[int]) -> tuple[Pill, Box]:
        return _pill(group, lines, types, w, h)

    def allowed(group: list[int]) -> bool:
        x0, y0, x1, y1 = build(group)[0].rect
        return not any(x0 <= cx < x1 and y0 <= cy < y1 for cx, cy in others)

    # Closest pairs first; ties by line index.
    group_of = {i: [i] for i in hit}
    pairs = sorted(
        (_gap(lines[i].box, lines[j].box), i, j)
        for n, i in enumerate(hit)
        for j in hit[n + 1 :]
        if _adjacent(lines[i].box, lines[j].box)
    )
    for _, i, j in pairs:
        a, b = group_of[i], group_of[j]
        if a is not b and allowed(a + b):
            for k in (merged := a + b):
                group_of[k] = merged
    groups = list({id(g): g for g in group_of.values()}.values())

    # A grown pill can overlap a neighbour; fuse those under the same constraint.
    done: list[list[int]] = []
    for g in groups:
        while (k := _fusable(g, done, build, allowed)) is not None:
            g = g + done.pop(k)
        done.append(g)
    return [build(g)[0] for g in done]


def _fusable(
    group: list[int],
    done: list[list[int]],
    build: Callable[[list[int]], tuple[Pill, Box]],
    allowed: Callable[[list[int]], bool],
) -> int | None:
    loose = build(group)[1]
    return next(
        (
            k
            for k, other in enumerate(done)
            if _overlap(loose, build(other)[1]) and allowed(group + other)
        ),
        None,
    )


def _gap(a: Box, b: Box) -> int:
    """Vertical gap between two boxes; negative when they overlap vertically."""
    return max(a[1], b[1]) - min(a[3], b[3])


def _adjacent(a: Box, b: Box) -> bool:
    """Overlapping or left-aligned lines whose covers would leave a strip thinner
    than a corner radius."""
    ha, hb = a[3] - a[1], b[3] - b[1]
    if _gap(a, b) >= _padding(ha) + _padding(hb) + min(_radius(ha), _radius(hb)):
        return False
    aligned = abs(a[0] - b[0]) <= _MERGE_LEFT * min(ha, hb)
    return min(a[2], b[2]) > max(a[0], b[0]) or aligned


def label_for(entity_type: str) -> str:
    return _LABELS.get(entity_type, _GENERIC)


def _label(types: dict[int, set[str]], group: list[int]) -> str:
    found = {label_for(t) for i in group for t in types[i]}
    return next(label for label in _PRIORITY if label in found)


def _pill(
    group: list[int],
    lines: list[OcrLine],
    types: dict[int, set[str]],
    w: int,
    h: int,
) -> tuple[Pill, Box]:
    """The pill for `group` and its outer rect before the grid snap."""
    raw = [lines[i].box for i in group]
    first = min(raw, key=lambda b: (b[1], b[0]))
    lh = first[3] - first[1]
    radius = _radius(lh)
    covers = tuple(_pad(b, w, h) for b in raw)
    loose = _outer(covers, lh, radius)
    label = _label(types, group)
    return Pill(_snap(loose, w, h), covers, first, label, radius, tuple(group)), loose


def _radius(lh: int) -> int:
    return min(_RADIUS_MAX, max(_RADIUS_MIN, round(lh * _RADIUS_FRAC)))


def _padding(lh: int) -> int:
    return max(_PAD_MIN, round(lh * _PAD_FRAC))


def _pad(box: Box, w: int, h: int) -> Box:
    x0, y0, x1, y1 = box
    pad = _padding(y1 - y0)
    return (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))


def _hpad(lh: int) -> int:
    """Horizontal room beyond the cover box on each side, before the grid snap."""
    return max(_HPAD_MIN, round(lh * _HPAD_FRAC))


def _outer(covers: tuple[Box, ...], lh: int, radius: int) -> Box:
    # A side margin of at least the radius keeps corner arcs clear of the covers.
    dx = max(_hpad(lh), radius)
    return (
        min(c[0] for c in covers) - dx,
        min(c[1] for c in covers),
        max(c[2] for c in covers) + dx,
        max(c[3] for c in covers),
    )


def _snap(rect: Box, w: int, h: int) -> Box:
    x0, y0, x1, y1 = rect
    return (
        max(0, x0 // _GRID * _GRID),
        max(0, y0 // _GRID * _GRID),
        min(w, -(-x1 // _GRID) * _GRID),
        min(h, -(-y1 // _GRID) * _GRID),
    )


def _overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _is_dark(frame: np.ndarray, rect: Box, blocked: list[Box]) -> bool:
    """True if a thin ring outside `rect` is dark and uniform; pill interiors excluded."""
    h, w = frame.shape[:2]
    x0, y0 = max(0, rect[0] - _RING), max(0, rect[1] - _RING)
    x1, y1 = min(w, rect[2] + _RING), min(h, rect[3] + _RING)
    ring = np.ones((y1 - y0, x1 - x0), bool)
    for bx0, by0, bx1, by1 in blocked:
        ring[
            max(0, by0 - y0) : max(0, by1 - y0), max(0, bx0 - x0) : max(0, bx1 - x0)
        ] = False
    px = frame[y0:y1, x0:x1][ring]
    if len(px) < 16:
        return False
    lum = px[:, 2] * 0.2126 + px[:, 1] * 0.7152 + px[:, 0] * 0.0722
    q1, med, q3 = np.percentile(lum, (25, 50, 75))
    return med < _DARK_MEDIAN and q3 - q1 < _DARK_IQR


def _paint(frame: np.ndarray, pill: Pill, fill: RGB) -> None:
    x0, y0, x1, y1 = pill.rect
    mask = _rounded(y1 - y0, x1 - x0, pill.radius)
    for cx0, cy0, cx1, cy1 in pill.covers:
        mask[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = True
    frame[y0:y1, x0:x1][mask] = (fill[2], fill[1], fill[0], 255)


def _rounded(h: int, w: int, radius: int) -> np.ndarray:
    mask = np.ones((h, w), bool)
    r = min(radius, h // 2, w // 2)
    if r > 0:
        corner = _corner(r)
        mask[:r, :r] &= corner
        mask[:r, w - r :] &= corner[:, ::-1]
        mask[h - r :, :r] &= corner[::-1, :]
        mask[h - r :, w - r :] &= corner[::-1, ::-1]
    return mask


def _corner(r: int) -> np.ndarray:
    """Top-left quarter-disc stencil: True inside the arc."""
    yy, xx = np.mgrid[:r, :r] + 0.5
    return (r - xx) ** 2 + (r - yy) ** 2 <= r * r


def label_px(lines: list[OcrLine]) -> int:
    """One size per frame; per-line box heights vary with ascenders and descenders."""
    lh = float(np.median([ln.box[3] - ln.box[1] for ln in lines]))
    return min(_LABEL_MAX, int(lh * _LABEL_FRAC) // _LABEL_STEP * _LABEL_STEP)


def fit_label(
    pill: Pill, px: int, fill: RGB, ink: RGB
) -> tuple[np.ndarray, int, int] | None:
    """(tile, x, y) at the largest size up to `px` that fits the pill, else None."""
    _, fy0, _, fy1 = pill.first
    lh = fy1 - fy0
    x0, y0, x1, y1 = pill.rect
    x = x0 + _hpad(lh)
    floor = max(_LABEL_FLOOR, round(px * _LABEL_SHRINK))
    for size in range(px, floor - 1, -2):
        tile = label_tile(pill.label, size, fill, ink)
        th, tw = tile.shape[:2]
        y = fy0 + (lh - th) // 2
        if x + tw <= x1 - _hpad(lh) and y >= y0 and y + th <= y1:
            return tile, x, y
    return None  # omit rather than truncate
