"""Offline tests for DisplayCapture teardown. No screen capture, no TCC."""

from __future__ import annotations

from safe_share.capture import DisplayCapture


class FakeStream:
    def __init__(self) -> None:
        self.stops = 0

    def stopCaptureWithCompletionHandler_(self, handler) -> None:
        self.stops += 1
        handler(None)


def test_stop_is_idempotent():
    cap = DisplayCapture(lambda *_: None, lambda *_: None)
    fake = FakeStream()
    cap._stream = fake

    cap.stop()
    cap.stop()  # second stop must be a no-op, not a double stopCapture

    assert fake.stops == 1
    assert cap._stream is None


def _output(monkeypatch, statuses, frames):
    """A real _StreamOutput fed a scripted sequence of (status, frame) callbacks."""
    import ScreenCaptureKit as SCK

    from safe_share import capture

    seq = iter(zip(statuses, frames, strict=True))
    current: list = [None, None]
    monkeypatch.setattr(capture, "_frame_status", lambda _: current[0])
    monkeypatch.setattr(capture, "_frame_from_sample_buffer", lambda _: current[1])
    calls: list[str] = []
    stops: list = []
    out = capture._StreamOutput.alloc().initWithHandlers_stop_idle_lost_(
        lambda _f: calls.append("frame"),
        lambda d, _m: stops.append(d),
        lambda: calls.append("idle"),
        lambda: calls.append("lost"),
    )

    def feed():
        for current[0], current[1] in seq:
            out.stream_didOutputSampleBuffer_ofType_(
                None, None, SCK.SCStreamOutputTypeScreen
            )

    return feed, calls, stops


def test_idle_is_a_heartbeat_never_a_frame(monkeypatch):
    import ScreenCaptureKit as SCK

    c, i = SCK.SCFrameStatusComplete, SCK.SCFrameStatusIdle
    feed, calls, stops = _output(monkeypatch, [c, i, i], ["px", None, None])
    feed()
    assert calls == ["frame", "idle", "idle"]
    assert stops == []


def test_unreadable_complete_frame_is_reported_at_once(monkeypatch):
    import ScreenCaptureKit as SCK

    c = SCK.SCFrameStatusComplete
    feed, calls, _ = _output(monkeypatch, [c, c], ["px", None])
    feed()
    assert calls == ["frame", "lost"]  # no Idle needed to report it


def test_a_failing_idle_handler_stops_the_session(monkeypatch):
    import ScreenCaptureKit as SCK

    from safe_share import capture
    from safe_share.capture_errors import StopDisposition

    monkeypatch.setattr(capture, "_frame_status", lambda _: SCK.SCFrameStatusIdle)
    stops: list = []

    def boom():
        raise RuntimeError("idle failed")

    out = capture._StreamOutput.alloc().initWithHandlers_stop_idle_lost_(
        lambda _f: None, lambda d, _m: stops.append(d), boom, None
    )
    out.stream_didOutputSampleBuffer_ofType_(None, None, SCK.SCStreamOutputTypeScreen)
    assert stops == [StopDisposition.TERMINAL]
