"""Step 1 spike: live display capture with one hardcoded mask, shown in an
AppKit window Teams can share, plus a separate Pause/Resume/Stop window.

Invariants: only a masked frame is ever shown; before the first frame and on
pause/stop/error the window shows a neutral screen; an epoch drops in-flight
frames so a late one cannot overwrite neutral.

Run:  python -m safe_share.spike   (needs Screen Recording permission)
"""

from __future__ import annotations

import signal
import sys
import threading

import numpy as np
import objc
import Quartz
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyRegular,
    NSBackingStoreBuffered,
    NSButton,
    NSMakeRect,
    NSStackView,
    NSUserInterfaceLayoutOrientationHorizontal,
    NSView,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSObject
from PyObjCTools import AppHelper
from Quartz import CATransaction

from .capture import CaptureError, DisplayCapture
from .config import load_rule_config
from .detect import RuleConfig, RuleDetector
from .ocr import TextRecognizer
from .redact import redact

_NEUTRAL_RGBA = (0.10, 0.10, 0.10, 1.0)


def _cgimage_from_bgra(arr: np.ndarray):
    """Build a CGImage from a BGRA array; returns (cgimage, arr).

    The provider does not copy, so `arr` is returned to be pinned alive for as
    long as the CGImage is referenced.
    """
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    stride = arr.strides[0]
    provider = Quartz.CGDataProviderCreateWithData(None, arr, arr.nbytes, None)
    cs = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB)  # match capture
    info = Quartz.kCGBitmapByteOrder32Little | Quartz.kCGImageAlphaNoneSkipFirst
    cg = Quartz.CGImageCreate(
        w,
        h,
        8,
        32,
        stride,
        cs,
        info,
        provider,
        None,
        False,
        Quartz.kCGRenderingIntentDefault,
    )
    return cg, arr


class Presenter(NSObject):
    """Publishes frames or the neutral screen to the layer, main thread only.

    Only frames stamped with the current epoch are published; pause/stop bump
    the epoch to invalidate in-flight frames.
    """

    def initWithLayer_(self, layer):
        self = objc.super(Presenter, self).init()
        if self is None:
            return None
        self._layer = layer
        self._epoch = 0
        self._live = False
        self._stopped = False
        # Guards _epoch/_live/_stopped, read on the worker thread and mutated
        # on the main thread.
        self._state_lock = threading.Lock()
        self._pinned = None  # keeps the current frame's array alive
        self._show_neutral()
        return self

    def goLive(self):
        with self._state_lock:
            if self._stopped:
                return  # Stop is terminal; Resume cannot revive the session.
            self._live = True

    def pause(self):
        with self._state_lock:
            self._epoch += 1
            self._live = False
        self._show_neutral()

    def stop(self):
        with self._state_lock:
            self._epoch += 1
            self._live = False
            self._stopped = True
        self._show_neutral()

    def currentEpoch(self) -> int:
        with self._state_lock:
            return self._epoch

    def publishPayload_(self, payload):
        """payload = (epoch, cgimage, pinned_array), main thread."""
        epoch, cg, pinned = payload
        with self._state_lock:
            drop = self._stopped or not self._live or epoch != self._epoch
        if drop:
            return  # stale or paused/stopped: never overwrite neutral
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        self._layer.setContents_(cg)
        CATransaction.commit()
        self._pinned = pinned

    def publishNeutralForEpoch_(self, epoch):
        """Neutral for a failed frame; ignored if stale or paused/stopped."""
        with self._state_lock:
            drop = self._stopped or not self._live or epoch != self._epoch
        if drop:
            return
        self._show_neutral()

    def _show_neutral(self):
        r, g, b, a = _NEUTRAL_RGBA
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        self._layer.setContents_(None)
        self._layer.setBackgroundColor_(Quartz.CGColorCreateGenericRGB(r, g, b, a))
        CATransaction.commit()
        self._pinned = None


class Controls(NSObject):
    def initWithApp_(self, app):
        self = objc.super(Controls, self).init()
        if self is None:
            return None
        self._app = app
        return self

    def pauseClicked_(self, sender):
        self._app.request_pause()

    def resumeClicked_(self, sender):
        self._app.request_resume()

    def stopClicked_(self, sender):
        self._app.request_stop()

    def quitClicked_(self, sender):
        self._app.request_quit()


class SafeShareSpike:
    """Wires capture, worker, presenter and controls into one running app."""

    def __init__(self) -> None:
        self._app = None
        self._presenter = None
        self._controls_target = None
        self._output_window = None
        self._controls_window = None
        self._capture = None
        self._rule_config = RuleConfig()

        # Latest-frame slot: newest raw frame wins, older pending ones drop.
        self._slot_lock = threading.Lock()
        self._latest_frame = None
        self._frame_ready = threading.Event()
        self._worker_stop = threading.Event()
        self._worker = None

    def _on_frame(self, frame: np.ndarray) -> None:
        with self._slot_lock:
            self._latest_frame = frame
        self._frame_ready.set()

    def _on_stop(self, message: str) -> None:
        print(f"[capture] {message}", file=sys.stderr)
        self._worker_stop.set()
        AppHelper.callAfter(self._presenter.stop)

    def _worker_loop(self) -> None:
        # One recognizer per worker thread (Vision request built per call).
        recognizer = TextRecognizer()
        detector = RuleDetector(self._rule_config)
        while not self._worker_stop.is_set():
            if not self._frame_ready.wait(0.5):
                continue
            self._frame_ready.clear()
            with self._slot_lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                continue
            # Stamp epoch now; a later pause/stop bumps it and drops the publish.
            epoch = self._presenter.currentEpoch()
            try:
                lines = recognizer.recognize(frame)
                detections = detector.detect([ln.text for ln in lines])
                masked = redact(frame, lines, detections)
            except Exception as exc:  # noqa: BLE001 - any failure must go neutral
                # Any failure (OCR/detect/render) is not "no sensitive text":
                # publish neutral, never the raw or partially processed frame.
                print(f"[pipeline] frame failed: {type(exc).__name__}", file=sys.stderr)
                AppHelper.callAfter(self._presenter.publishNeutralForEpoch_, epoch)
                continue
            cg, pinned = _cgimage_from_bgra(masked)
            # callAfter passes the Python tuple to the main thread without ObjC
            # container coercion, so `pinned` stays alive to back the CGImage.
            AppHelper.callAfter(self._presenter.publishPayload_, (epoch, cg, pinned))

    def request_pause(self) -> None:
        print("[controls] pause")
        self._presenter.pause()

    def request_resume(self) -> None:
        print("[controls] resume")
        self._presenter.goLive()

    def request_stop(self) -> None:
        print("[controls] stop")
        self._worker_stop.set()
        self._presenter.stop()
        if self._capture is not None:
            self._capture.stop()

    def request_quit(self) -> None:
        print("[controls] quit")
        self._worker_stop.set()
        if self._capture is not None:
            self._capture.stop()
        AppHelper.stopEventLoop()

    def _build_windows(self) -> None:
        out_style = NSWindowStyleMaskTitled | NSWindowStyleMaskMiniaturizable
        self._output_window = (
            NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(120, 120, 960, 600), out_style, NSBackingStoreBuffered, False
            )
        )
        self._output_window.setTitle_("Safe Share")
        self._output_window.setReleasedWhenClosed_(False)

        view = NSView.alloc().initWithFrame_(self._output_window.contentView().bounds())
        layer = Quartz.CALayer.layer()
        layer.setContentsGravity_(Quartz.kCAGravityResizeAspect)
        view.setLayer_(layer)  # before setWantsLayer_ = layer-hosting
        view.setWantsLayer_(True)
        scale = self._output_window.backingScaleFactor() or 1.0
        layer.setContentsScale_(scale)
        self._output_window.setContentView_(view)
        self._presenter = Presenter.alloc().initWithLayer_(layer)

        # Separate window: buttons must not land in the shared surface.
        ctl_style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
        )
        self._controls_window = (
            NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(120, 40, 420, 70), ctl_style, NSBackingStoreBuffered, False
            )
        )
        self._controls_window.setTitle_("Safe Share Controls")
        self._controls_window.setReleasedWhenClosed_(False)

        self._controls_target = Controls.alloc().initWithApp_(self)
        pause = NSButton.buttonWithTitle_target_action_(
            "Pause", self._controls_target, b"pauseClicked:"
        )
        resume = NSButton.buttonWithTitle_target_action_(
            "Resume", self._controls_target, b"resumeClicked:"
        )
        stop = NSButton.buttonWithTitle_target_action_(
            "Stop", self._controls_target, b"stopClicked:"
        )
        quit_ = NSButton.buttonWithTitle_target_action_(
            "Quit", self._controls_target, b"quitClicked:"
        )
        stack = NSStackView.stackViewWithViews_([pause, resume, stop, quit_])
        stack.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        stack.setSpacing_(8)
        stack.setFrame_(self._controls_window.contentView().bounds())
        self._controls_window.contentView().addSubview_(stack)

        self._output_window.makeKeyAndOrderFront_(None)
        self._controls_window.orderFront_(None)

    def run(self) -> int:
        self._app = NSApplication.sharedApplication()
        # Regular policy before capture so this process is self-excludable.
        self._app.setActivationPolicy_(NSApplicationActivationPolicyRegular)

        if not Quartz.CGPreflightScreenCaptureAccess():
            Quartz.CGRequestScreenCaptureAccess()  # prompts once; grant needs relaunch
            print(
                "Screen Recording permission is required.\n"
                "Grant it to this Python in System Settings > Privacy & Security >\n"
                "Screen & System Audio Recording, then quit and run again.",
                file=sys.stderr,
            )
            return 1

        self._build_windows()
        self._app.activate()

        self._rule_config, cfg_source = load_rule_config()
        print(
            f"[config] source={cfg_source} emails={self._rule_config.emails} "
            f"credentials={self._rule_config.credentials} "
            f"terms={len(self._rule_config.terms)}"
        )

        self._capture = DisplayCapture(self._on_frame, self._on_stop)
        try:
            self._capture.build()
        except CaptureError as exc:
            print(f"[capture] setup failed: {exc}", file=sys.stderr)
            return 1

        print(
            f"[capture] display {self._capture.pixel_width}x{self._capture.pixel_height} "
            f"@scale {self._capture.point_scale} | {self._capture.exclusion_note}"
        )

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self._presenter.goLive()
        self._capture.start()

        # installInterrupt handles SIGINT (Ctrl-C); add SIGTERM so `kill` also
        # stops the stream before exit rather than dropping to default terminate.
        signal.signal(signal.SIGTERM, lambda *_: self.request_quit())

        AppHelper.runEventLoop(installInterrupt=True)
        return 0


def main() -> int:
    return SafeShareSpike().run()


if __name__ == "__main__":
    raise SystemExit(main())
