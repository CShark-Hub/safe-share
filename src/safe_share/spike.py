"""Live display capture, redacted and shown in an AppKit window Teams can
share, plus a separate Pause/Resume/Stop window.

Scheduling and the privacy invariants live in `pipeline.py`; this module is the
AppKit publisher and the window/control wiring.

Run:  python -m safe_share.spike   (needs Screen Recording permission)
"""

from __future__ import annotations

import signal
import sys
import threading
import time

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
    NSTextField,
    NSTimer,
    NSUserInterfaceLayoutOrientationHorizontal,
    NSUserInterfaceLayoutOrientationVertical,
    NSView,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSData, NSObject
from PyObjCTools import AppHelper, MachSignals
from Quartz import CATransaction

from .capture import CaptureError, DisplayCapture
from .capture_errors import StopDisposition
from .config import load_config
from .control_state import button_enablement, control_state, status_text
from .detect import Detector, RuleConfig, RuleDetector, UnionDetector
from .ocr import TextRecognizer
from .pipeline import FramePipeline, SessionState

_NEUTRAL_RGBA = (0.10, 0.10, 0.10, 1.0)


def should_deliver(accepted: bool, seq: int, last_seq: int) -> bool:
    """Gate for a masked frame on the main thread: current epoch, and newer than
    whatever is already on screen."""
    return accepted and seq > last_seq


def _cgimage_from_bgra(arr: np.ndarray):
    """Build a CGImage owning a copy of a BGRA array.

    The NSData copy makes the pixel lifetime the CGImage's own; a no-copy
    provider would leave Core Animation reading an array Python may free.
    """
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    stride = arr.strides[0]
    provider = Quartz.CGDataProviderCreateWithCFData(
        NSData.dataWithBytes_length_(arr, arr.nbytes)
    )
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
    return cg


class LayerPresenter:
    """Publishes frames or the neutral screen to a CALayer.

    `publish`/`publish_neutral` are called from the worker thread and hop to the
    main thread; the layer is never touched off it.
    """

    def __init__(self, layer, state: SessionState) -> None:
        self._layer = layer
        self._state = state
        self._last_seq = 0
        self._cache_frame = None
        self._cache_cg = None
        self.show_neutral()

    def publish(self, epoch: int, seq: int, frame: np.ndarray) -> None:
        if not self._state.accepts(epoch):
            return
        AppHelper.callAfter(self._deliver, epoch, seq, frame)

    def publish_neutral(self, epoch: int) -> None:
        AppHelper.callAfter(self._deliver_neutral, epoch)

    def _deliver(self, epoch: int, seq: int, frame: np.ndarray) -> None:
        if not should_deliver(self._state.accepts(epoch), seq, self._last_seq):
            return  # stale, paused or stopped: never overwrite neutral
        cg = self._cgimage_for(frame)
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        self._layer.setContents_(cg)
        CATransaction.commit()
        self._last_seq = seq

    def _deliver_neutral(self, epoch: int) -> None:
        # Neutral is always safe to show; only a stale failure from an earlier
        # epoch is ignored, so it cannot blank a good frame after a resume.
        if self._state.live and epoch != self._state.epoch:
            return
        self.show_neutral()

    def _cgimage_for(self, frame: np.ndarray):
        if frame is self._cache_frame:  # republished unchanged frame
            return self._cache_cg
        cg = _cgimage_from_bgra(frame)
        self._cache_frame, self._cache_cg = frame, cg
        return cg

    def show_neutral(self) -> None:
        r, g, b, a = _NEUTRAL_RGBA
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        self._layer.setContents_(None)
        self._layer.setBackgroundColor_(Quartz.CGColorCreateGenericRGB(r, g, b, a))
        CATransaction.commit()
        self._cache_frame = None
        self._cache_cg = None


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

    def refreshTick_(self, timer):
        self._app._refresh_controls()


class SafeShareSpike:
    """Wires capture, pipeline, presenter and controls into one running app."""

    def __init__(self) -> None:
        self._app = None
        self._presenter: LayerPresenter | None = None
        self._controls_target = None
        self._output_window = None
        self._controls_window = None
        self._buttons: dict[str, object] = {}
        self._status_label = None
        self._refresh_timer = None
        self._capture: DisplayCapture | None = None
        self._rule_config = RuleConfig()
        self._detector_note = "rules only"
        self._state = SessionState()
        self._pipeline: FramePipeline | None = None
        self._worker: threading.Thread | None = None

    def _on_frame(self, frame: np.ndarray) -> None:
        if self._pipeline is not None:
            self._pipeline.submit(frame)

    def _on_stop(self, disposition: StopDisposition, message: str) -> None:
        # Every disposition ends the session with neutral output; the class only
        # informs the log (a future rebuild path can branch on RECOVERABLE).
        print(f"[capture] stop ({disposition.value}): {message}", file=sys.stderr)
        self._state.stop()  # before shutdown: no in-flight frame may publish
        if self._pipeline is not None:
            self._pipeline.shutdown()
        if self._presenter is not None:
            AppHelper.callAfter(self._presenter.show_neutral)
        # A frame-handler error stops via here while the stream is still live;
        # tear it down on the main thread (idempotent if already stopped).
        AppHelper.callAfter(self._teardown_capture)

    def _on_pipeline_error(self, reason: str) -> None:
        if reason == "watchdog":
            print(
                "[pipeline] watchdog fired: output stalled → neutral", file=sys.stderr
            )
        else:
            print(f"[pipeline] frame failed: {reason}", file=sys.stderr)

    def _on_session_stopped(self) -> None:
        # Pipeline ended the session (watchdog/worker death) on a background
        # thread; tear down capture on the main thread. stop() is idempotent.
        AppHelper.callAfter(self._teardown_capture)

    def _teardown_capture(self) -> None:
        if self._capture is not None:
            self._capture.stop()
        self._refresh_controls()

    def request_pause(self) -> None:
        print("[controls] pause")
        self._state.pause()
        if self._pipeline is not None:
            self._pipeline.invalidate()
        if self._presenter is not None:
            self._presenter.show_neutral()
        self._refresh_controls()

    def request_resume(self) -> None:
        if not self._state.go_live():
            print("[controls] resume ignored: session stopped")
            self._refresh_controls()
            return
        print("[controls] resume")
        if self._pipeline is not None:
            self._pipeline.invalidate()
        self._refresh_controls()

    def request_stop(self) -> None:
        print("[controls] stop")
        self._state.stop()
        if self._pipeline is not None:
            self._pipeline.invalidate()
            self._pipeline.shutdown()
        if self._presenter is not None:
            self._presenter.show_neutral()
        if self._capture is not None:
            self._capture.stop()
        self._refresh_controls()

    def request_quit(self) -> None:
        print("[controls] quit")
        self._state.stop()
        if self._pipeline is not None:
            self._pipeline.shutdown()
        if self._capture is not None:
            self._capture.stop()
        if self._pipeline is not None:
            # Here, not atexit: terminate_ calls C exit(), which skips Python's atexit.
            print(self._pipeline.timing_summary(), flush=True)
        AppHelper.stopEventLoop()

    def _build_windows(self) -> LayerPresenter:
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
        presenter = LayerPresenter(layer, self._state)
        self._presenter = presenter

        # Separate window: buttons must not land in the shared surface.
        ctl_style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
        )
        self._controls_window = (
            NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(120, 40, 420, 104), ctl_style, NSBackingStoreBuffered, False
            )
        )
        self._controls_window.setTitle_("Safe Share Controls")
        self._controls_window.setReleasedWhenClosed_(False)

        self._controls_target = Controls.alloc().initWithApp_(self)
        self._buttons = {
            "pause": NSButton.buttonWithTitle_target_action_(
                "Pause", self._controls_target, b"pauseClicked:"
            ),
            "resume": NSButton.buttonWithTitle_target_action_(
                "Resume", self._controls_target, b"resumeClicked:"
            ),
            "stop": NSButton.buttonWithTitle_target_action_(
                "Stop", self._controls_target, b"stopClicked:"
            ),
        }
        quit_ = NSButton.buttonWithTitle_target_action_(
            "Quit", self._controls_target, b"quitClicked:"
        )
        button_row = NSStackView.stackViewWithViews_(
            [
                self._buttons["pause"],
                self._buttons["resume"],
                self._buttons["stop"],
                quit_,
            ]
        )
        button_row.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        button_row.setSpacing_(8)

        self._status_label = NSTextField.labelWithString_("")
        column = NSStackView.stackViewWithViews_([self._status_label, button_row])
        column.setOrientation_(NSUserInterfaceLayoutOrientationVertical)
        column.setSpacing_(8)
        column.setFrame_(self._controls_window.contentView().bounds())
        self._controls_window.contentView().addSubview_(column)

        self._output_window.makeKeyAndOrderFront_(None)
        self._controls_window.orderFront_(None)
        self._refresh_controls()
        return presenter

    def _refresh_controls(self) -> None:
        """Reflect the session state in the buttons and status label. Main thread
        only; a repeating timer also runs it so a background stop (watchdog,
        capture error) updates the UI without a per-path hop."""
        live, stopped = self._state.snapshot()
        state = control_state(live, stopped)
        enabled = button_enablement(state)
        for name, button in self._buttons.items():
            button.setEnabled_(enabled[name])
        if self._status_label is not None:
            self._status_label.setStringValue_(
                f"{status_text(state)} · {self._detector_note}"
            )

    def _build_detector(self, use_gliner: bool) -> Detector:
        """Rules, plus GLiNER when enabled and loadable; never downloads."""
        rules = RuleDetector(self._rule_config)
        if not use_gliner:
            return rules
        try:
            from .gliner_detect import GlinerDetector, ModelUnavailable, load_best

            t = time.monotonic()
            model, where = load_best()
        except ImportError:
            print("[detect] GLiNER not installed (uv sync --extra ml); rules only")
            return rules
        except ModelUnavailable as exc:
            print(f"[detect] GLiNER unavailable ({exc}); rules only")
            return rules
        print(
            f"[detect] rules + GLiNER on {where}, loaded in {time.monotonic() - t:.1f}s"
        )
        self._detector_note = "rules + GLiNER"
        return UnionDetector(rules, GlinerDetector(model))

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

        presenter = self._build_windows()
        self._app.activate()

        config = load_config()
        self._rule_config = config.rules
        print(
            f"[config] source={config.source} emails={config.rules.emails} "
            f"credentials={config.rules.credentials} "
            f"terms={len(config.rules.terms)} gliner={config.gliner}"
        )
        # Before capture starts, so the load stays outside the watchdog budget.
        detector = self._build_detector(config.gliner)

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

        pipeline = FramePipeline(
            TextRecognizer(),
            detector,
            presenter,
            self._state,
            on_error=self._on_pipeline_error,
            on_session_stopped=self._on_session_stopped,
        )
        self._pipeline = pipeline
        self._worker = threading.Thread(target=pipeline.run, daemon=True)
        self._worker.start()

        self._state.go_live()
        self._capture.start()
        self._refresh_controls()

        # Reflect a background stop (watchdog, capture error) in the controls.
        self._refresh_timer = (
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.3, self._controls_target, b"refreshTick:", None, True
            )
        )

        # Mach signals run on the event loop; a plain signal.signal handler would
        # wait for Python code to run. Ctrl-C and `kill` take the Quit path.
        for sig in (signal.SIGINT, signal.SIGTERM):
            MachSignals.signal(sig, lambda _: self.request_quit())

        AppHelper.runEventLoop(installInterrupt=False)
        return 0


def main() -> int:
    return SafeShareSpike().run()


if __name__ == "__main__":
    raise SystemExit(main())
