"""Data types shared by the pipeline, OCR and redaction, free of PyObjC."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OcrLine:
    """One recognized text line. `box` is (x0, y0, x1, y1) full-res, top-left."""

    text: str
    box: tuple[int, int, int, int]
    confidence: float
