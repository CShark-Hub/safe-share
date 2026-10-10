"""ScreenCaptureKit display capture: produces raw BGRA frames from one display.

Excludes this process's own windows so the output window is not recursively
captured. Scheduling is in `pipeline.py`, masking in `redact.py`, windows in `app.py`.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import CoreMedia
import libdispatch
import numpy as np
import objc
import Quartz
import ScreenCaptureKit as SCK
from Foundation import NSObject

from .capture_errors import StopDisposition, classify_stop

# BGRA, shape (h, w, 4); a private copy that outlives its CVPixelBuffer.
Frame = np.ndarray

FrameHandler = Callable[[Frame], None]  # called per Complete frame, capture queue
IdleHandler = Callable[[], None]  # called per Idle frame (screen unchanged)
LostHandler = Callable[[], None]  # called when a Complete frame has no readable image
# Called when the stream stops, on an SCK queue: (disposition, human-readable).
StopHandler = Callable[[StopDisposition, str], None]

# SCK stops calling back after ~35 s of unchanged screen; log when it resumes.
_SILENCE_NOTE = 5.0


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


def _frame_status(sbuf) -> int | None:
    attaches = CoreMedia.CMSampleBufferGetSampleAttachmentsArray(sbuf, False)
    if not attaches or attaches.count() == 0:
        return None
    status = attaches.objectAtIndex_(0).get(SCK.SCStreamFrameInfoStatus)
    return None if status is None else int(status)


def _frame_from_sample_buffer(sbuf) -> Frame | None:
    """Return a private BGRA copy of a Complete frame, or None to skip it."""
    if _frame_status(sbuf) != SCK.SCFrameStatusComplete:
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
        # required: the bytes are valid only until unlock, and the published
        # CGImage reads this array without copying it.
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpr // 4, 4)
        return arr[:, :w, :].copy()
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)


class _StreamOutput(
    NSObject,
    # PyObjC consumes this class keyword; it is not an __init_subclass__ arg.
    protocols=[  # pyright: ignore[call-arg]
        objc.protocolNamed("SCStreamOutput"),
        objc.protocolNamed("SCStreamDelegate"),
    ],
):
    """Receives sample buffers and stream-stopped errors from SCStream."""

    def initWithHandlers_stop_idle_lost_(
        self, frame_handler, stop_handler, idle_handler, lost_handler
    ):
        self = objc.super(_StreamOutput, self).init()
        if self is None:
            return None
        self._frame_handler = frame_handler
        self._stop_handler = stop_handler
        self._idle_handler = idle_handler
        self._lost_handler = lost_handler
        self._last_callback: float | None = None
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream, sbuf, otype):
        if otype != SCK.SCStreamOutputTypeScreen:
            return
        # PyObjC swallows exceptions raised in a callback, so a broken frame path
        # would silently kill the pipeline. Route any failure to a neutral stop.
        try:
            # Off-main callback: drain temporaries here or memory climbs.
            with objc.autorelease_pool():
                status = _frame_status(sbuf)
                self._note_callback(status)
                # A static screen yields only Idle frames, which carry no image.
                if status == SCK.SCFrameStatusIdle:
                    if self._idle_handler is not None:
                        self._idle_handler()
                    return
                frame = _frame_from_sample_buffer(sbuf)
                if frame is not None:
                    self._frame_handler(frame)
                elif status == SCK.SCFrameStatusComplete and self._lost_handler:
                    self._lost_handler()
        except Exception as exc:  # noqa: BLE001
            self._stop_handler(
                StopDisposition.TERMINAL, f"frame handler error: {type(exc).__name__}"
            )

    @objc.python_method
    def _note_callback(self, status: int | None) -> None:
        now = time.monotonic()
        if (
            self._last_callback is not None
            and now - self._last_callback > _SILENCE_NOTE
        ):
            print(
                f"[capture] callbacks resumed after {now - self._last_callback:.0f}s "
                f"silence (status {status})",
                file=sys.stderr,
            )
        self._last_callback = now

    def stream_didStopWithError_(self, stream, error):
        with objc.autorelease_pool():  # foreign queue; drain the NSError
            if error is None:
                self._stop_handler(StopDisposition.CLEAN, "stream stopped")
                return
            code = error.code()
            disposition = classify_stop(code)
            # Log domain+code only, never localizedDescription: keep any possible
            # app/window name out of the log (rule 6).
            self._stop_handler(disposition, f"{error.domain()} code {code}")

    # Active/inactive fire on the SCK queue (macOS 15.2+). Observational for now:
    # inactive behaviour on a display filter is unverified on this machine, so we
    # log rather than assume it is terminal.
    def streamDidBecomeInactive_(self, stream):
        print("[capture] stream inactive", file=sys.stderr)

    def streamDidBecomeActive_(self, stream):
        print("[capture] stream active", file=sys.stderr)


class DisplayCapture:
    """Captures one display, calling `frame_handler` per Complete frame,
    `idle_handler` per Idle one and `lost_handler` for an unreadable Complete one.

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
        idle_handler: IdleHandler | None = None,
        lost_handler: LostHandler | None = None,
    ) -> None:
        self._frame_handler = frame_handler
        self._stop_handler = stop_handler
        self._idle_handler = idle_handler
        self._lost_handler = lost_handler
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

        self._output = _StreamOutput.alloc().initWithHandlers_stop_idle_lost_(
            self._frame_handler,
            self._stop_handler,
            self._idle_handler,
            self._lost_handler,
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
                disposition = classify_stop(error.code())
                self._stop_handler(
                    disposition, f"failed to start capture: {error.code()}"
                )

        self._stream.startCaptureWithCompletionHandler_(on_started)

    def stop(self) -> None:
        # Idempotent: a double Stop (or Stop after an error stop) must not raise
        # -3808 or re-stop a cleared stream.
        stream, self._stream = self._stream, None
        if stream is None:
            return

        def on_stopped(error):
            if error is not None:
                print(f"[capture] stop error: {error.code()}", file=sys.stderr)

        stream.stopCaptureWithCompletionHandler_(on_stopped)
