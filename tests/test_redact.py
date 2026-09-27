"""Leak guards for redaction: covered lines keep no original pixel, nothing spills."""

from __future__ import annotations

import random

import numpy as np
import pytest

from safe_share.detect import Detection
from safe_share.redact import plan, redact
from safe_share.types import OcrLine

# Non-grey, so it cannot be mistaken for a pill fill, label ink or their blend.
DARK_MARKER = (3, 9, 1, 255)
LIGHT_MARKER = (250, 244, 252, 255)
SIZES = [(640, 400), (20, 10)]
TYPES = ["person", "email", "term", "aws_key"]


def _stacked(rng: random.Random, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Text-like rows, sometimes flush with the frame edges."""
    boxes, y = [], rng.choice([0, 3, 10])
    while y < h - 2:
        lh = rng.randint(1, 36)
        x0 = rng.choice([0, rng.randint(0, w // 2)])
        x1 = rng.choice([w, rng.randint(x0, w)])
        boxes.append((x0, y, x1, min(h, y + lh)))
        y += lh + rng.randint(0, 20)
    return boxes


def _scattered(rng: random.Random, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Boxes anywhere: overlapping, side by side, tiny or zero-size."""
    boxes = []
    for _ in range(rng.randint(1, 30)):
        x0, y0 = rng.randint(0, w), rng.randint(0, h)
        boxes.append((x0, y0, rng.randint(x0, w), rng.randint(y0, h)))
    return boxes


def _layout(seed: int, w: int, h: int) -> tuple[list[OcrLine], list[Detection]]:
    rng = random.Random(seed)
    make = _stacked if seed % 2 else _scattered
    lines = [OcrLine("x", box, 1.0) for box in make(rng, w, h)]
    detections = [
        Detection(i, 0, 1, rng.choice(TYPES))
        for i in range(len(lines))
        if rng.random() < 0.6
    ]
    return lines, detections


def _frame(w: int, h: int, marker) -> np.ndarray:
    frame = np.empty((h, w, 4), np.uint8)
    frame[:] = marker
    return frame


@pytest.mark.parametrize(("w", "h"), SIZES)
@pytest.mark.parametrize("marker", [DARK_MARKER, LIGHT_MARKER])
@pytest.mark.parametrize("seed", range(60))
def test_no_original_pixel_survives_in_a_detected_line(seed, marker, w, h):
    lines, detections = _layout(seed, w, h)
    frame = redact(_frame(w, h, marker), lines, detections)
    hit = {d.line_index for d in detections}
    covers = [c for p in plan(lines, detections, w, h) for c in p.covers]
    assert len(covers) == len(hit)  # fusing must not drop a line
    for x0, y0, x1, y1 in [lines[i].box for i in hit] + covers:
        region = frame[y0:y1, x0:x1]
        assert not np.all(region == marker, axis=-1).any()


@pytest.mark.parametrize(("w", "h"), SIZES)
@pytest.mark.parametrize("seed", range(60))
def test_paint_stays_in_pills_and_labels_on_paint(seed, w, h):
    lines, detections = _layout(seed, w, h)
    bare = redact(_frame(w, h, LIGHT_MARKER), lines, detections, draw_label=False)
    labelled = redact(_frame(w, h, LIGHT_MARKER), lines, detections)
    painted = ~np.all(bare == LIGHT_MARKER, axis=-1)
    changed = np.any(labelled != bare, axis=-1)
    assert not (changed & ~painted).any()
    inside = np.zeros((h, w), bool)
    for p in plan(lines, detections, w, h):
        x0, y0, x1, y1 = p.rect
        inside[y0:y1, x0:x1] = True
    assert not (painted & ~inside).any()
