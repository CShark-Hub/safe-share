"""Live display capture, redacted and shown in an AppKit window Teams can
share, plus a separate Pause/Resume/Stop window.

Scheduling and the privacy invariants live in `pipeline.py`; this module is the
AppKit publisher and the window/control wiring.

Run:  safe-share   (needs Screen Recording permission; downloads the model once)
"""

from __future__ import annotations

import argparse
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
    NSAttributedString,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSButton,
    NSColor,
    NSEdgeInsets,
    NSEventModifierFlagCommand,
    NSFont,
    NSFontAttributeName,
    NSFontWeightMedium,
    NSFontWeightRegular,
    NSFontWeightSemibold,
    NSForegroundColorAttributeName,
    NSGraphicsContext,
    NSImage,
    NSImageSymbolConfiguration,
    NSLayoutAttributeLeading,
    NSMakePoint,
    NSMakeRect,
    NSMaxX,
    NSMaxY,
    NSMutableAttributedString,
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
from .control_state import control_state, control_view
from .detect import Detector, RuleConfig, RuleDetector, UnionDetector
from .name_cache import NameCacheDetector
from .ocr import TextRecognizer
from .pipeline import FramePipeline, SessionState

_NEUTRAL_RGBA = (0.10, 0.10, 0.10, 1.0)
_CARD_SIZE = (1920, 1200)
_STATUS_MIN_WIDTH = 300
_DOT_COLORS = {
    "green": NSColor.systemGreenColor,
    "orange": NSColor.systemOrangeColor,
    "red": NSColor.systemRedColor,
    "gray": NSColor.systemGrayColor,
}


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


def _neutral_card():
    """The static screen shown whenever output is withheld; identical in every state.

    Drawn with AppKit into a bitmap context; main thread only.
    """
    w, h = _CARD_SIZE
    cs = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB)
    ctx = Quartz.CGBitmapContextCreate(
        None,
        w,
        h,
        8,
        0,
        cs,
        Quartz.kCGBitmapByteOrder32Little | Quartz.kCGImageAlphaPremultipliedFirst,
    )
    NSGraphicsContext.saveGraphicsState()
    try:
        NSGraphicsContext.setCurrentContext_(
            NSGraphicsContext.graphicsContextWithCGContext_flipped_(ctx, False)
        )
        NSColor.colorWithSRGBRed_green_blue_alpha_(*_NEUTRAL_RGBA).setFill()
        NSBezierPath.fillRect_(NSMakeRect(0, 0, w, h))
        cx, cy = w / 2, h / 2  # origin bottom-left

        config = NSImageSymbolConfiguration.configurationWithPointSize_weight_(
            96, NSFontWeightMedium
        ).configurationByApplyingConfiguration_(
            NSImageSymbolConfiguration.configurationWithHierarchicalColor_(
                NSColor.systemGrayColor()
            )
        )
        lock = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            "lock.fill", None
        )
        if lock is not None:
            lock = lock.imageWithSymbolConfiguration_(config)
            lw, lh = lock.size().width, lock.size().height
            lock.drawInRect_(NSMakeRect(cx - lw / 2, cy + 70, lw, lh))

        _draw_centered("Safe Share", NSFontWeightSemibold, 64, 0.90, cx, cy - 10)
        _draw_centered("Screen hidden", NSFontWeightRegular, 36, 0.56, cx, cy - 80)
    finally:
        NSGraphicsContext.restoreGraphicsState()
    return Quartz.CGBitmapContextCreateImage(ctx)


def _draw_centered(text, weight, size, white, cx, baseline):
    attrs = {
        NSFontAttributeName: NSFont.systemFontOfSize_weight_(size, weight),
        NSForegroundColorAttributeName: NSColor.colorWithWhite_alpha_(white, 1.0),
    }
    s = NSAttributedString.alloc().initWithString_attributes_(text, attrs)
    s.drawAtPoint_(NSMakePoint(cx - s.size().width / 2, baseline))


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
        try:
            self._card = _neutral_card()
        except Exception as exc:  # noqa: BLE001 - cosmetic; plain colour is still neutral
            print(f"[ui] neutral card failed: {exc}", file=sys.stderr)
            self._card = None
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
        self._layer.setContents_(self._card)
        # Same colour as the card, so aspect-fit letterboxing is invisible.
        self._layer.setBackgroundColor_(Quartz.CGColorCreateSRGB(r, g, b, a))
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

    def toggleClicked_(self, sender):
        self._app.request_toggle()

    def stopClicked_(self, sender):
        self._app.request_stop()

    def quitClicked_(self, sender):
        self._app.request_quit()

    def refreshTick_(self, timer):
        self._app._refresh_controls()


class SafeShareApp:
    """Wires capture, pipeline, presenter and controls into one running app."""

    def __init__(self) -> None:
        self._app = None
        self._presenter: LayerPresenter | None = None
        self._controls_target = None
        self._output_window = None
        self._controls_window = None
        self._buttons: dict[str, object] = {}
        self._status_label = None
        self._shown_view = None
        self._refresh_timer = None
        self._capture: DisplayCapture | None = None
        self._rule_config = RuleConfig()
        self._detector_note = "rules only"
        self._names_off = False
        self._name_cache: NameCacheDetector | None = None
        self._state = SessionState()
        self._pipeline: FramePipeline | None = None
        self._worker: threading.Thread | None = None

    def _on_frame(self, frame: np.ndarray) -> None:
        if self._pipeline is not None:
            self._pipeline.submit(frame)

    def _on_idle(self) -> None:
        if self._pipeline is not None:
            self._pipeline.heartbeat()

    def _on_lost(self) -> None:
        if self._pipeline is not None:
            self._pipeline.lost()

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
            note = self._pipeline.stall_report() if self._pipeline is not None else ""
            print(
                f"[pipeline] watchdog fired: output stalled → neutral ({note})",
                file=sys.stderr,
            )
        elif reason == "lost":
            print("[pipeline] capture lost a frame → neutral", file=sys.stderr)
        else:
            print(f"[pipeline] frame failed: {reason}", file=sys.stderr)

    def _on_session_stopped(self) -> None:
        # Pipeline ended the session (watchdog/worker death) on a background
        # thread; tear down capture on the main thread. stop() is idempotent.
        AppHelper.callAfter(self._teardown_capture)

    def _teardown_capture(self) -> None:
        if self._capture is not None:
            self._capture.stop()
        self._clear_names()
        self._refresh_controls()

    def _clear_names(self) -> None:
        if self._name_cache is not None:
            # Counts only: the names themselves are never logged.
            n, refused = len(self._name_cache), self._name_cache.refused
            self._name_cache.clear()
            self._name_cache = None
            print(f"[detect] name cache cleared ({n} entries, {refused} refused)")

    def request_toggle(self) -> None:
        live, stopped = self._state.snapshot()
        if stopped:
            self._refresh_controls()
        elif live:
            self.request_pause()
        else:
            self.request_resume()

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
        self._clear_names()
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
        self._clear_names()
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
        toggle = NSButton.buttonWithTitle_target_action_(
            "Resume", self._controls_target, b"toggleClicked:"
        )
        # Fixed at the wider title so Stop/Quit do not shift on each toggle.
        toggle.widthAnchor().constraintEqualToConstant_(
            toggle.fittingSize().width
        ).setActive_(True)
        toggle.setKeyEquivalent_("p")
        toggle.setKeyEquivalentModifierMask_(NSEventModifierFlagCommand)
        self._buttons = {
            "toggle": toggle,
            "stop": NSButton.buttonWithTitle_target_action_(
                "Stop", self._controls_target, b"stopClicked:"
            ),
        }
        quit_ = NSButton.buttonWithTitle_target_action_(
            "Quit", self._controls_target, b"quitClicked:"
        )
        button_row = NSStackView.stackViewWithViews_(
            [self._buttons["toggle"], self._buttons["stop"], quit_]
        )
        button_row.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        button_row.setSpacing_(8)

        self._status_label = NSTextField.labelWithString_("")
        self._status_label.widthAnchor().constraintGreaterThanOrEqualToConstant_(
            _STATUS_MIN_WIDTH
        ).setActive_(True)
        column = NSStackView.stackViewWithViews_([self._status_label, button_row])
        column.setOrientation_(NSUserInterfaceLayoutOrientationVertical)
        column.setAlignment_(NSLayoutAttributeLeading)
        column.setSpacing_(12)
        column.setEdgeInsets_(NSEdgeInsets(20, 20, 20, 20))
        self._controls_window.setContentSize_(column.fittingSize())
        column.setFrame_(self._controls_window.contentView().bounds())
        self._controls_window.contentView().addSubview_(column)
        out = self._output_window.frame()
        x = NSMaxX(out) + 16
        screen = self._output_window.screen()
        if screen is not None:
            x = min(x, NSMaxX(screen.visibleFrame()) - column.fittingSize().width)
        self._controls_window.setFrameTopLeftPoint_(NSMakePoint(x, NSMaxY(out)))

        self._output_window.makeKeyAndOrderFront_(None)
        self._controls_window.orderFront_(None)
        self._refresh_controls()
        return presenter

    def _refresh_controls(self) -> None:
        """Reflect the session state in the buttons and status label. Main thread
        only; a repeating timer also runs it so a background stop (watchdog,
        capture error) updates the UI without a per-path hop."""
        live, stopped = self._state.snapshot()
        started = self._pipeline is not None and self._pipeline.published > 0
        view = control_view(control_state(live, stopped, started), self._names_off)
        if view == self._shown_view or self._status_label is None:
            return
        self._shown_view = view
        self._buttons["toggle"].setTitle_(view.toggle_title)
        self._buttons["toggle"].setEnabled_(view.toggle_enabled)
        self._buttons["stop"].setEnabled_(view.stop_enabled)
        font = NSFont.systemFontOfSize_weight_(13, NSFontWeightSemibold)
        text = NSMutableAttributedString.alloc().initWithString_attributes_(
            f"● {view.status}",
            {
                NSFontAttributeName: font,
                NSForegroundColorAttributeName: NSColor.labelColor(),
            },
        )
        text.addAttribute_value_range_(
            NSForegroundColorAttributeName, _DOT_COLORS[view.dot](), (0, 1)
        )
        self._status_label.setAttributedStringValue_(text)
        self._status_label.setToolTip_(f"Detection: {self._detector_note}")

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
            self._names_off = True
            return rules
        except ModelUnavailable as exc:
            print(f"[detect] GLiNER unavailable ({exc}); rules only")
            self._names_off = True
            return rules
        print(
            f"[detect] rules + GLiNER on {where}, loaded in {time.monotonic() - t:.1f}s"
        )
        self._detector_note = "rules + GLiNER"
        self._name_cache = NameCacheDetector(
            UnionDetector(rules, GlinerDetector(model))
        )
        return self._name_cache

    def _fetch_model_if_missing(self) -> None:
        """One-time model download, before any window or capture exists. A failure
        leaves the model missing, so `_build_detector` falls back to rules only."""
        try:
            from .gliner_detect import fetch_in_subprocess, model_cached

            if model_cached():
                return
        except ImportError:
            return  # reported by _build_detector
        print("[detect] GLiNER model not cached: downloading once, before capture")
        if not fetch_in_subprocess():
            print("[detect] model download failed")

    def run(self) -> int:
        config = load_config()
        self._rule_config = config.rules
        print(
            f"[config] source={config.source} emails={config.rules.emails} "
            f"credentials={config.rules.credentials} "
            f"terms={len(config.rules.terms)} gliner={config.gliner}"
        )
        if config.gliner:
            self._fetch_model_if_missing()

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

        # Before capture starts, so the load stays outside the watchdog budget.
        detector = self._build_detector(config.gliner)
        self._shown_view = None  # the tooltip carries the detector note

        self._capture = DisplayCapture(
            self._on_frame,
            self._on_stop,
            idle_handler=self._on_idle,
            lost_handler=self._on_lost,
        )
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="safe-share", description="Redact a live screen for screen sharing."
    )
    ap.add_argument(
        "--fetch-model",
        action="store_true",
        help="download the GLiNER model (uses network) and exit",
    )
    args = ap.parse_args(argv)
    if args.fetch_model:
        try:
            from .gliner_detect import fetch_in_subprocess, model_cached

            ok = model_cached() or fetch_in_subprocess()
        except ImportError:
            print("GLiNER not installed: uv sync --extra ml", file=sys.stderr)
            return 1
        print("model ready" if ok else "model download failed")
        return 0 if ok else 1
    return SafeShareApp().run()


if __name__ == "__main__":
    raise SystemExit(main())
