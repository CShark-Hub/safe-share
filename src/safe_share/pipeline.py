"""Frame scheduling: latest-frame slot, OCR/detect/redact, epoch invalidation.

No AppKit or ScreenCaptureKit dependency; the AppKit side implements `Publisher`.

Invariants:
  - only a fully processed frame is published, and a mask is only ever applied
    to the frame it was computed on
  - pause, resume and stop bump the epoch; a frame stamped with an older epoch
    is dropped, and the skip cache resets when the epoch changes
  - any OCR/detect/render failure publishes neutral, never a raw frame
  - if output stalls while live (a hang or silent capture stall), a watchdog
    thread stops the session and publishes neutral
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np
from PIL import Image

from .detect import RuleDetector
from .redact import redact
from .types import OcrLine

# Unchanged-frame skip: re-OCR'ing identical pixels wastes the frame budget.
_SIG_SIZE = 128
_SIG_DIFF_THRESH = 4.0
# Force a full re-OCR at least this often so a comparator miss cannot freeze the
# output indefinitely while the real screen moves on.
_MAX_SKIP_FRAMES = 15

# Watchdog: while live, if nothing has reached the screen for this long (an OCR
# hang, or capture silently stalling), fall to neutral. Placeholder values; set
# them from a measured process() p95 on a real capture run.
_WATCHDOG_TIMEOUT = 2.0
_WATCHDOG_POLL = 0.25
# The first frame after start/resume gets a wider budget: a cold Vision request
# (model warm-up) can take several seconds without anything being wrong.
_WATCHDOG_FIRST_FRAME = 10.0


class Recognizer(Protocol):
    def recognize(self, frame: np.ndarray) -> list[OcrLine]: ...


class Publisher(Protocol):
    """Sink for finished output. Implementations must be thread-safe."""

    def publish(self, epoch: int, seq: int, frame: np.ndarray) -> None: ...

    def publish_neutral(self, epoch: int) -> None: ...


class SessionState:
    """Epoch, live and stopped under one lock: the publication gate."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._epoch = 0
        self._live = False
        self._stopped = False

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    @property
    def live(self) -> bool:
        with self._lock:
            return self._live and not self._stopped

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def go_live(self) -> bool:
        """Start or resume; False if the session is already stopped."""
        with self._lock:
            if self._stopped:
                return False  # Stop is terminal; Resume cannot revive a session.
            self._epoch += 1
            self._live = True
            return True

    def pause(self) -> None:
        with self._lock:
            self._epoch += 1
            self._live = False

    def stop(self) -> None:
        with self._lock:
            self._epoch += 1
            self._live = False
            self._stopped = True

    def accepts(self, epoch: int) -> bool:
        """True only for work stamped with the current epoch of a live session."""
        with self._lock:
            return self._live and not self._stopped and epoch == self._epoch


def frame_signature(frame: np.ndarray) -> np.ndarray:
    """Box-pooled luma thumbnail for change detection.

    Area pooling, not subsampling: a change confined to a few pixels still moves
    the cell covering it, so a single new text line cannot fall between samples.
    Channel order is irrelevant to change detection, so BGRA is read as RGBA.
    """
    img = Image.fromarray(frame, mode="RGBA").convert("L")
    img = img.resize((_SIG_SIZE, _SIG_SIZE), Image.BOX)
    return np.asarray(img, dtype=np.float32)


def unchanged(a: np.ndarray | None, b: np.ndarray) -> bool:
    """True only if no cell moved by more than the threshold.

    Max per cell, not mean: a text line is a small fraction of frame area and a
    mean would average it away.
    """
    return (
        a is not None
        and a.shape == b.shape
        and float(np.abs(a - b).max()) < _SIG_DIFF_THRESH
    )


class FramePipeline:
    """Consumes raw frames from a single-slot queue and publishes masked ones.

    `submit` runs on the capture queue; `run` and the publisher calls run on the
    worker thread.
    """

    def __init__(
        self,
        recognizer: Recognizer,
        detector: RuleDetector,
        publisher: Publisher,
        state: SessionState,
        on_error: Callable[[str], None] | None = None,
        watchdog_timeout: float = _WATCHDOG_TIMEOUT,
        watchdog_poll: float = _WATCHDOG_POLL,
        watchdog_first_frame: float = _WATCHDOG_FIRST_FRAME,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._recognizer = recognizer
        self._detector = detector
        self._publisher = publisher
        self._state = state
        self._on_error = on_error
        self._watchdog_timeout = watchdog_timeout
        self._watchdog_poll = watchdog_poll
        self._watchdog_first_frame = watchdog_first_frame
        self._monotonic = monotonic

        self._slot_lock = threading.Lock()
        self._pending: tuple[np.ndarray, int, int] | None = None
        self._seq = 0
        self._ready = threading.Event()
        self._shutdown = threading.Event()

        # Worker-thread only; reset whenever the epoch changes.
        self._last_sig: np.ndarray | None = None
        self._last_masked: np.ndarray | None = None
        self._last_epoch: int | None = None
        self._skips = 0  # consecutive republishes, capped by _MAX_SKIP_FRAMES

        # Time of the last output that reached the screen; the watchdog measures
        # the gap since. `_first_frame` widens the budget until the first publish
        # after a live edge. Both guarded by _progress_lock (worker + watchdog).
        self._progress_lock = threading.Lock()
        self._last_progress = self._monotonic()
        self._first_frame = True

        # Counts only: no frame content and no recognized text.
        self.dropped_stale = 0
        self.dropped_not_live = 0
        self.dropped_epoch = 0
        self.published = 0
        self.republished = 0
        self.failed = 0
        self.watchdog_fired = 0

    def submit(self, frame: np.ndarray) -> int:
        """Newest frame wins; an unprocessed older one is dropped.

        The epoch is stamped here, not at processing time: what decides whether
        a frame may be shown is the state it was *captured* under.
        """
        epoch = self._state.epoch
        with self._slot_lock:
            if self._pending is not None:
                self.dropped_stale += 1
            self._seq += 1
            self._pending = (frame, self._seq, epoch)
        self._ready.set()
        return self._seq

    def invalidate(self) -> None:
        """Discard the pending frame. The epoch stamp is what gates publication;
        this only avoids working on a frame that can no longer be shown."""
        with self._slot_lock:
            self._pending = None
            self._ready.clear()

    def shutdown(self) -> None:
        self._shutdown.set()
        self._ready.set()  # wake the worker so it observes the flag

    @property
    def pending_seq(self) -> int | None:
        with self._slot_lock:
            return None if self._pending is None else self._pending[1]

    def _take(self) -> tuple[np.ndarray, int, int] | None:
        with self._slot_lock:
            pending = self._pending
            self._pending = None
            return pending

    def _mark_progress(self, now: float | None = None) -> None:
        """Record that output reached the screen; a real publish also ends the
        first-frame grace."""
        with self._progress_lock:
            self._last_progress = self._monotonic() if now is None else now
            self._first_frame = False

    def _watchdog_tick(self, was_live: bool) -> bool:
        """One watchdog check. Returns whether the session is live afterwards.

        Extracted from the loop so it runs deterministically under a fake clock.
        """
        now = self._monotonic()
        live = self._state.live
        if live and not was_live:
            # Live edge (start/resume): restart the clock and re-arm the wider
            # first-frame budget so a cold OCR or a pause's duration cannot fire.
            with self._progress_lock:
                self._last_progress = now
                self._first_frame = True
            return True
        with self._progress_lock:
            gap = now - self._last_progress
            budget = (
                self._watchdog_first_frame
                if self._first_frame
                else self._watchdog_timeout
            )
        if live and gap > budget:
            self._trigger_watchdog()
            return False
        return live

    def _trigger_watchdog(self) -> None:
        # A hung worker cannot be cancelled from Python: make the output safe and
        # abandon the thread. stop() is terminal, so the thread's eventual late
        # publish is dropped at the epoch gate.
        self.watchdog_fired += 1
        self._state.stop()
        self._shutdown.set()
        self._ready.set()  # release a merely-stalled (not hung) worker
        try:
            if self._on_error is not None:
                self._on_error("watchdog")
            self._publisher.publish_neutral(self._state.epoch)
        except Exception:  # noqa: BLE001, S110
            pass

    def _watchdog_loop(self) -> None:
        was_live = False
        while not self._shutdown.wait(self._watchdog_poll):
            was_live = self._watchdog_tick(was_live)

    def run(self) -> None:
        watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        watchdog.start()
        try:
            while not self._shutdown.is_set():
                if not self._ready.wait(0.5):
                    continue
                self._ready.clear()
                pending = self._take()
                if pending is None:
                    continue
                self.process(*pending)
        except BaseException as exc:  # noqa: BLE001 - a dead worker must go neutral
            # The worker dying silently would leave the last sanitized frame on
            # screen while the real screen moves on. Stop first so the neutral
            # publish cannot be overtaken by in-flight work. Not re-raised: the
            # default thread excepthook would print a message that can carry
            # recognized text.
            self.failed += 1
            self._state.stop()
            # The publisher may be what died, so its failure must not escape
            # here: the default excepthook would print a message that can carry
            # recognized text.
            try:
                if self._on_error is not None:
                    self._on_error(type(exc).__name__)
                self._publisher.publish_neutral(self._state.epoch)
            except Exception:  # noqa: BLE001, S110
                pass
        finally:
            self._shutdown.set()
            watchdog.join(timeout=1.0)

    def process(self, frame: np.ndarray, seq: int, epoch: int) -> None:
        """Run one captured frame end to end. `epoch` is its capture-time stamp."""
        if not self._state.accepts(epoch):
            # Captured under a state that no longer permits publication, so it
            # is not OCR'd either: never read a screen the user paused.
            if self._state.live:
                self.dropped_epoch += 1
            else:
                self.dropped_not_live += 1
            return
        if epoch != self._last_epoch:
            self._last_sig = None
            self._last_masked = None
            self._last_epoch = epoch
            self._skips = 0

        sig = frame_signature(frame)
        last_masked = self._last_masked
        if (
            last_masked is not None
            and self._skips < _MAX_SKIP_FRAMES
            and unchanged(self._last_sig, sig)
        ):
            # Republish the last SANITIZED frame: never re-apply old geometry to
            # new pixels, never publish this frame raw.
            if self._emit(epoch, seq, last_masked):
                self.republished += 1
                self._skips += 1
            return

        try:
            lines = self._recognizer.recognize(frame)
            detections = self._detector.detect([ln.text for ln in lines])
            masked = redact(frame, lines, detections)
        except Exception as exc:  # noqa: BLE001 - any failure must go neutral
            # A failed run is not "no sensitive text". The exception message can
            # carry recognized text, so only its type is reported.
            self.failed += 1
            self._last_sig = None
            self._last_masked = None
            if self._on_error is not None:
                self._on_error(type(exc).__name__)
            if self._state.accepts(epoch):
                self._publisher.publish_neutral(epoch)
                self._mark_progress()
            else:
                self.dropped_epoch += 1
            return

        if self._emit(epoch, seq, masked):
            self.published += 1
            self._last_sig = sig
            self._last_masked = masked
            self._skips = 0

    def _emit(self, epoch: int, seq: int, masked: np.ndarray) -> bool:
        """Publish unless the epoch was invalidated while this frame was worked on."""
        if not self._state.accepts(epoch):
            self.dropped_epoch += 1
            return False
        self._publisher.publish(epoch, seq, masked)
        self._mark_progress()
        return True
