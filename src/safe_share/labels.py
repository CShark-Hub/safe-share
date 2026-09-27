"""Redaction label tiles rendered with Core Text (thread-safe; runs on the worker)."""

from __future__ import annotations

import math
from functools import lru_cache

import CoreText
import numpy as np
import Quartz
from Foundation import NSAttributedString

RGB = tuple[int, int, int]

# Semibold on the Core Text weight scale (NSFontWeightSemibold).
_WEIGHT = 0.3


@lru_cache(maxsize=16)
def _font(px: int):
    base = CoreText.CTFontCreateUIFontForLanguage(
        CoreText.kCTFontUIFontSystem, px, None
    )
    desc = CoreText.CTFontDescriptorCreateWithAttributes(
        {CoreText.kCTFontTraitsAttribute: {CoreText.kCTFontWeightTrait: _WEIGHT}}
    )
    return CoreText.CTFontCreateCopyWithAttributes(base, px, None, desc)


@lru_cache(maxsize=128)
def label_tile(text: str, px: int, fill: RGB, ink: RGB) -> np.ndarray:
    """Opaque BGRA tile of `text` on `fill`, cropped to cap height (uppercase only)."""
    if not text:
        raise ValueError("empty label")
    ink_cg = Quartz.CGColorCreateSRGB(*(c / 255 for c in ink), 1.0)
    attrs = {
        CoreText.kCTFontAttributeName: _font(px),
        CoreText.kCTForegroundColorAttributeName: ink_cg,
    }
    line = CoreText.CTLineCreateWithAttributedString(
        NSAttributedString.alloc().initWithString_attributes_(text, attrs)
    )
    width, ascent, descent, _ = CoreText.CTLineGetTypographicBounds(
        line, None, None, None
    )
    w, h = math.ceil(width), math.ceil(ascent + descent)
    ctx = Quartz.CGBitmapContextCreate(
        None,
        w,
        h,
        8,
        w * 4,
        Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB),
        Quartz.kCGBitmapByteOrder32Little | Quartz.kCGImageAlphaPremultipliedFirst,
    )
    Quartz.CGContextSetFillColorWithColor(
        ctx, Quartz.CGColorCreateSRGB(*(c / 255 for c in fill), 1.0)
    )
    Quartz.CGContextFillRect(ctx, Quartz.CGRectMake(0, 0, w, h))
    Quartz.CGContextSetTextPosition(ctx, 0, descent)
    CoreText.CTLineDraw(line, ctx)
    img = Quartz.CGBitmapContextCreateImage(ctx)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
    stride = Quartz.CGImageGetBytesPerRow(img)
    rows = np.frombuffer(bytes(data), np.uint8).reshape(h, stride)
    top = max(0, math.floor(ascent - CoreText.CTFontGetCapHeight(_font(px))) - 1)
    bottom = min(h, math.ceil(ascent) + 1)
    tile = rows[top:bottom, : w * 4].reshape(bottom - top, w, 4).copy()
    tile[..., 3] = 255
    tile.flags.writeable = False  # shared by the cache
    return tile
