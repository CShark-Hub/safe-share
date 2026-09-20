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
