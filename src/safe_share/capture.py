"""ScreenCaptureKit display capture: produces raw BGRA frames from one display.

Excludes this process's own windows so the output window is not recursively
captured. Masking, scheduling and the neutral screen live in `spike.py`.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass

import CoreMedia
import libdispatch
import numpy as np
import objc
import Quartz
import ScreenCaptureKit as SCK
from Foundation import NSObject

# BGRA, shape (h, w, 4); a private copy that outlives its CVPixelBuffer.
Frame = np.ndarray

FrameHandler = Callable[[Frame], None]  # called per Complete frame, capture queue
StopHandler = Callable[[str], None]  # called on unexpected stop, SCK queue


@dataclass
class CaptureError(Exception):
    """Raised when capture cannot be set up (no display, filter, or start)."""

    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def _fetch_shareable_content(timeout: float = 5.0):
    """Fetch SCShareableContent synchronously.

    Call before NSApplication.run() or from a worker thread, never from the main
    thread once the run loop is live: the handler lands on an XPC queue and we
    would deadlock.
    """
    box: dict = {}
    done = threading.Event()

    def handler(content, error):
        box["content"] = content
        box["error"] = error
        done.set()

    SCK.SCShareableContent.getShareableContentWithCompletionHandler_(handler)
    if not done.wait(timeout):
        raise CaptureError("timed out fetching shareable content")
    if box.get("error") is not None:
        raise CaptureError(f"shareable content error: {box['error']}")
    return box["content"]


def _build_self_excluding_filter(content, display):
    """Filter capturing `display` minus this process's windows.

    Matches this process by PID, not bundle id (framework Python shares the
    org.python.python bundle). Returns (filter, note) where note names which
    exclusion path was taken.
    """
    pid = os.getpid()

    own_apps = [a for a in content.applications() if a.processID() == pid]
    if own_apps:
        flt = SCK.SCContentFilter.alloc().initWithDisplay_excludingApplications_exceptingWindows_(
            display, own_apps, []
        )
        return flt, f"excluding application pid={pid}"

    own_windows = [
        w
        for w in content.windows()
        if w.owningApplication() is not None
        and w.owningApplication().processID() == pid
    ]
    if own_windows:
        flt = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
            display, own_windows
        )
        return flt, f"excluding {len(own_windows)} own window(s) (app not listed)"

    flt = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])
    return flt, "WARNING: could not self-exclude; expect mirror recursion"


def _frame_from_sample_buffer(sbuf) -> Frame | None:
    """Return a private BGRA copy of a Complete frame, or None to skip it."""
    attaches = CoreMedia.CMSampleBufferGetSampleAttachmentsArray(sbuf, False)
    if not attaches or attaches.count() == 0:
        return None
    info = attaches.objectAtIndex_(0)
    if info.get(SCK.SCStreamFrameInfoStatus) != SCK.SCFrameStatusComplete:
        return None

    pb = CoreMedia.CMSampleBufferGetImageBuffer(sbuf)
    if pb is None:
        return None

    Quartz.CVPixelBufferLockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)
    try:
        w = Quartz.CVPixelBufferGetWidth(pb)
        h = Quartz.CVPixelBufferGetHeight(pb)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
        base = Quartz.CVPixelBufferGetBaseAddress(pb)
        if base is None:
            return None
        buf = base.as_buffer(bpr * h)
        # bpr is padded (> w*4); reshape by bpr//4 then slice. .copy() is
        # required: the bytes are valid only until unlock.
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpr // 4, 4)
        return arr[:, :w, :].copy()
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)


class _StreamOutput(
    NSObject,
    protocols=[
        objc.protocolNamed("SCStreamOutput"),
        objc.protocolNamed("SCStreamDelegate"),
    ],
):
    """Receives sample buffers and stream-stopped errors from SCStream."""

    def initWithHandlers_stop_(self, frame_handler, stop_handler):
        self = objc.super(_StreamOutput, self).init()
        if self is None:
            return None
        self._frame_handler = frame_handler
        self._stop_handler = stop_handler
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream, sbuf, otype):
        if otype != SCK.SCStreamOutputTypeScreen:
            return
        # Off-main callback: drain temporaries here or memory climbs.
        with objc.autorelease_pool():
            frame = _frame_from_sample_buffer(sbuf)
            if frame is not None:
                self._frame_handler(frame)

    def stream_didStopWithError_(self, stream, error):
        code = error.code() if error is not None else 0
        self._stop_handler(f"stream stopped (code {code})")


class DisplayCapture:
    """Captures one display, calling `frame_handler` per Complete frame.

    Call build() before start()/stop(). Frames and the stop callback arrive on
    ScreenCaptureKit queues, not the calling thread.
    """

    def __init__(
        self,
        frame_handler: FrameHandler,
        stop_handler: StopHandler,
        display_index: int = 0,
        fps: int = 15,
        show_cursor: bool = False,
    ) -> None:
        self._frame_handler = frame_handler
        self._stop_handler = stop_handler
        self._display_index = display_index
        self._fps = fps
        self._show_cursor = show_cursor

        self._stream = None
        self._output = None
        self._queue = libdispatch.dispatch_queue_create(b"safeshare.capture", None)

        # Set by build().
        self.pixel_width = 0
        self.pixel_height = 0
        self.point_scale = 1.0
        self.exclusion_note = ""

    def build(self) -> None:
        content = _fetch_shareable_content()
        displays = content.displays()
        if not displays or self._display_index >= displays.count():
            raise CaptureError(
                f"no display at index {self._display_index} "
                f"({displays.count() if displays else 0} available)"
            )
        display = displays.objectAtIndex_(self._display_index)

        flt, note = _build_self_excluding_filter(content, display)
        self.exclusion_note = note

        scale = flt.pointPixelScale() or 1.0
        rect = flt.contentRect()
        self.point_scale = float(scale)
        self.pixel_width = int(rect.size.width * scale)
        self.pixel_height = int(rect.size.height * scale)

        cfg = SCK.SCStreamConfiguration.alloc().init()
        cfg.setWidth_(self.pixel_width)
        cfg.setHeight_(self.pixel_height)
        cfg.setPixelFormat_(Quartz.kCVPixelFormatType_32BGRA)
        cfg.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, self._fps))
        cfg.setQueueDepth_(5)
        cfg.setShowsCursor_(self._show_cursor)
        cfg.setColorSpaceName_(Quartz.kCGColorSpaceSRGB)

        self._output = _StreamOutput.alloc().initWithHandlers_stop_(
            self._frame_handler, self._stop_handler
        )
        self._stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(
            flt, cfg, self._output
        )
        ok, err = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self._output, SCK.SCStreamOutputTypeScreen, self._queue, None
        )
        if not ok:
            raise CaptureError(f"addStreamOutput failed: {err}")

    def start(self) -> None:
        if self._stream is None:
            raise CaptureError("start() before build()")

        def on_started(error):
            if error is not None:
                self._stop_handler(f"failed to start capture: {error}")

        self._stream.startCaptureWithCompletionHandler_(on_started)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stopCaptureWithCompletionHandler_(lambda error: None)
