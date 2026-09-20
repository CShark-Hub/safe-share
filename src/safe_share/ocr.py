"""On-device OCR via Apple Vision (VNRecognizeTextRequest) over BGRA frames.

Returns line-level text with full-resolution, top-left-origin pixel boxes. OCR
runs on a downscaled copy for speed; boxes are mapped back to the full frame.
A failed recognition raises OcrError: the caller must treat that as a failure
and publish neutral output, never as "no sensitive text" (privacy rule 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import objc
import Quartz
import Vision
from PIL import Image


@dataclass(frozen=True)
class OcrLine:
    """One recognized text line. `box` is (x0, y0, x1, y1) full-res, top-left."""

    text: str
    box: tuple[int, int, int, int]
    confidence: float


class OcrError(Exception):
    """Recognition could not complete. Not equivalent to an empty result."""


def _downscale_bgra(frame: np.ndarray, factor: float) -> np.ndarray:
    """Lanczos-downscale a BGRA frame by `factor` (channel order is irrelevant
    to resizing and to text contrast, so the array is treated as RGBA)."""
    h, w = frame.shape[:2]
    nw, nh = max(1, round(w * factor)), max(1, round(h * factor))
    img = Image.fromarray(frame, mode="RGBA").resize((nw, nh), Image.LANCZOS)
    return np.asarray(img)


def _cgimage_from_bgra(arr: np.ndarray):
    """Build a CGImage over `arr` for Vision to read synchronously.

    Returns (cgimage, arr); keep `arr` referenced until performRequests returns,
    since the data provider does not copy.
    """
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    provider = Quartz.CGDataProviderCreateWithData(None, arr, arr.nbytes, None)
    cs = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB)
    info = Quartz.kCGBitmapByteOrder32Little | Quartz.kCGImageAlphaNoneSkipFirst
    cg = Quartz.CGImageCreate(
        w,
        h,
        8,
        32,
        arr.strides[0],
        cs,
        info,
        provider,
        None,
        False,
        Quartz.kCGRenderingIntentDefault,
    )
    return cg, arr


class TextRecognizer:
    """Reusable Vision text recognizer. One instance per worker thread.

    `downscale` is applied before OCR; boxes still come back in full-res
    coordinates because Vision reports normalized rects.
    """

    def __init__(
        self,
        downscale: float = 1.0,
        languages: tuple[str, ...] = ("en-US",),
        accurate: bool = True,
    ) -> None:
        self._downscale = downscale
        self._languages = list(languages)
        self._level = (
            Vision.VNRequestTextRecognitionLevelAccurate
            if accurate
            else Vision.VNRequestTextRecognitionLevelFast
        )

    def _make_request(self):
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(self._level)
        # Off: correction mangles emails, IDs and credential-like strings.
        req.setUsesLanguageCorrection_(False)
        req.setRecognitionLanguages_(self._languages)
        return req

    def recognize(self, frame: np.ndarray) -> list[OcrLine]:
        """OCR one full-res BGRA frame; an empty list means no text was found.

        Raises OcrError when Vision reports failure. Other exceptions may also
        propagate: the caller must treat any exception (and any timeout or
        cancellation) as a failed run and publish neutral output, never as
        "no sensitive text" (privacy rule 4).
        """
        full_h, full_w = frame.shape[:2]
        small = (
            _downscale_bgra(frame, self._downscale) if self._downscale != 1.0 else frame
        )
        # Convert observations to plain Python inside the pool; Vision results
        # are autoreleased and must not outlive it (memory grows otherwise).
        with objc.autorelease_pool():
            cg, _pin = _cgimage_from_bgra(small)
            handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
                cg, None
            )
            req = self._make_request()
            ok, err = handler.performRequests_error_([req], None)
            if not ok:
                # Message carries an NSError; callers log the exception type
                # only, never this string verbatim.
                raise OcrError(f"Vision performRequests failed: {err}")
            results = req.results()
            lines: list[OcrLine] = []
            for i in range(results.count() if results is not None else 0):
                obs = results.objectAtIndex_(i)
                cand = obs.topCandidates_(1)
                if not cand:
                    continue
                lines.append(
                    OcrLine(
                        text=cand[0].string(),
                        box=_box_to_pixels(obs.boundingBox(), full_w, full_h),
                        confidence=float(obs.confidence()),
                    )
                )
        # Vision does not guarantee reading order; sort top-to-bottom, then
        # left-to-right, so line_index matches visual order (the next-line
        # keyed-secret rule in detect.py depends on it).
        lines.sort(key=lambda ln: (ln.box[1], ln.box[0]))
        return lines


def _box_to_pixels(bb, w: int, h: int) -> tuple[int, int, int, int]:
    """Normalized bottom-left Vision rect -> full-res top-left pixel box,
    rounded outward so a cover never lands a pixel short."""
    import math

    x0 = math.floor(bb.origin.x * w)
    x1 = math.ceil((bb.origin.x + bb.size.width) * w)
    y0 = math.floor((1.0 - bb.origin.y - bb.size.height) * h)
    y1 = math.ceil((1.0 - bb.origin.y) * h)
    return (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
