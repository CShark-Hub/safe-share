"""Offline tests for the publication invariants. No screen, no Vision."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from safe_share.detect import RuleConfig, RuleDetector
from safe_share.ocr import OcrError
from safe_share.pipeline import (
    _MAX_SKIP_FRAMES,
    FramePipeline,
    SessionState,
    frame_signature,
    unchanged,
)
from safe_share.types import OcrLine

_EMAIL_BOX = (4, 4, 60, 20)


def _frame(fill: int = 200) -> np.ndarray:
    frame = np.full((64, 96, 4), fill, dtype=np.uint8)
    frame[..., 3] = 255
    return frame


class FakeRecognizer:
    """Returns one line of text regardless of the pixels."""

    def __init__(self, text: str = "mail me at ana@example.invalid") -> None:
        self._text = text
        self.calls = 0

    def recognize(self, frame: np.ndarray) -> list[OcrLine]:
        self.calls += 1
        return [OcrLine(text=self._text, box=_EMAIL_BOX, confidence=1.0)]


class BlockingRecognizer(FakeRecognizer):
    """Blocks inside recognize() so a state change can race the OCR call."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def recognize(self, frame: np.ndarray) -> list[OcrLine]:
        self.entered.set()
        assert self.release.wait(5), "test did not release the recognizer"
        return super().recognize(frame)


class FailingRecognizer:
    def recognize(self, frame: np.ndarray) -> list[OcrLine]:
        raise OcrError("Vision performRequests failed")


class RecordingPublisher:
    def __init__(self) -> None:
        self.frames: list[tuple[int, int, np.ndarray]] = []
        self.neutrals: list[int] = []

    def publish(self, epoch: int, seq: int, frame: np.ndarray) -> None:
        self.frames.append((epoch, seq, frame))

    def publish_neutral(self, epoch: int) -> None:
        self.neutrals.append(epoch)


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _pipeline(recognizer, terms: list[str] | None = None, **kwargs):
    state = SessionState()
    publisher = RecordingPublisher()
    pipeline = FramePipeline(
        recognizer,
        RuleDetector(RuleConfig(terms=terms or [])),
        publisher,
        state,
        **kwargs,
    )
    return pipeline, publisher, state


def _run_in_thread(pipeline: FramePipeline) -> threading.Thread:
    worker = threading.Thread(target=pipeline.run, daemon=True)
    worker.start()
    return worker


def test_pause_during_ocr_publishes_nothing():
    recognizer = BlockingRecognizer()
    pipeline, publisher, state = _pipeline(recognizer)
    state.go_live()
    worker = _run_in_thread(pipeline)

    pipeline.submit(_frame())
    assert recognizer.entered.wait(5)
    state.pause()
    pipeline.invalidate()
    recognizer.release.set()

    pipeline.shutdown()
    worker.join(5)
    assert publisher.frames == []
    assert pipeline.dropped_epoch == 1


def test_frame_captured_while_paused_is_never_published_after_resume():
    """The reported leak: the worker dequeues a paused-period frame, and the
    resume lands before it is processed."""
    recognizer = FakeRecognizer()
    pipeline, publisher, state = _pipeline(recognizer)
    state.go_live()
    state.pause()

    pipeline.submit(_frame())  # captured while the output was neutral
    taken = pipeline._take()  # worker dequeues it before the resume
    state.go_live()
    pipeline.invalidate()  # cannot reach a frame already dequeued
    pipeline.process(*taken)

    assert recognizer.calls == 0
    assert publisher.frames == []


def test_pause_and_resume_between_capture_and_processing_drops_the_frame():
    """A full pause/resume cycle while a frame waits must discard it, even
    though the session is live again by the time it is processed."""
    recognizer = FakeRecognizer()
    pipeline, publisher, state = _pipeline(recognizer)
    state.go_live()

    pipeline.submit(_frame())
    taken = pipeline._take()
    assert taken is not None
    state.pause()
    state.go_live()

    pipeline.process(*taken)

    assert recognizer.calls == 0
    assert publisher.frames == []


def test_resume_bumps_the_epoch():
    """Without the bump, a frame stamped during the pause becomes publishable
    again on resume."""
    state = SessionState()
    state.go_live()
    paused_epoch = (state.pause(), state.epoch)[1]
    state.go_live()

    assert state.accepts(paused_epoch) is False


def test_frames_submitted_while_paused_are_not_ocrd():
    recognizer = FakeRecognizer()
    pipeline, publisher, state = _pipeline(recognizer)
    state.go_live()
    epoch = state.epoch
    state.pause()

    pipeline.process(_frame(), 1, epoch)

    assert recognizer.calls == 0
    assert publisher.frames == []
    assert pipeline.dropped_not_live == 1


def test_ocr_failure_publishes_neutral_and_reruns_ocr_next_frame():
    recognizer = FailingRecognizer()
    pipeline, publisher, state = _pipeline(recognizer)
    state.go_live()

    pipeline.process(_frame(), 1, state.epoch)

    assert publisher.frames == []
    assert publisher.neutrals == [state.epoch]
    assert pipeline.failed == 1

    # The skip cache must not survive a failure: the next identical frame is
    # processed again rather than republished.
    pipeline._recognizer = FakeRecognizer()
    pipeline.process(_frame(), 2, state.epoch)
    assert pipeline.republished == 0
    assert pipeline.published == 1


def test_failure_while_paused_publishes_no_neutral_for_a_stale_epoch():
    pipeline, publisher, state = _pipeline(FailingRecognizer())
    state.go_live()
    epoch = state.epoch
    state.pause()
    state.go_live()

    pipeline.process(_frame(), 1, epoch)

    assert publisher.neutrals == []
    assert publisher.frames == []


def test_worker_death_stops_the_session_and_publishes_neutral():
    """A failure outside the per-frame try (here, in the publisher) must not
    leave the last sanitized frame on screen."""

    class ExplodingPublisher(RecordingPublisher):
        def publish(self, epoch: int, seq: int, frame: np.ndarray) -> None:
            raise RuntimeError("publish failed")

    state = SessionState()
    publisher = ExplodingPublisher()
    stopped_calls = []
    pipeline = FramePipeline(
        FakeRecognizer(),
        RuleDetector(RuleConfig()),
        publisher,
        state,
        on_session_stopped=lambda: stopped_calls.append(True),
    )
    state.go_live()
    pipeline.submit(_frame())
    worker = _run_in_thread(pipeline)
    worker.join(5)

    assert worker.is_alive() is False
    assert state.stopped is True
    assert publisher.frames == []
    assert publisher.neutrals != []
    assert stopped_calls == [True]  # worker death tears down capture too


def test_worker_death_survives_a_publisher_that_also_fails_on_neutral():
    class DeadPublisher(RecordingPublisher):
        def publish(self, epoch: int, seq: int, frame: np.ndarray) -> None:
            raise RuntimeError("publish failed")

        def publish_neutral(self, epoch: int) -> None:
            raise RuntimeError("runloop gone")

    state = SessionState()
    pipeline = FramePipeline(
        FakeRecognizer(), RuleDetector(RuleConfig()), DeadPublisher(), state
    )
    state.go_live()
    pipeline.submit(_frame())

    escaped: list[BaseException] = []
    previous = threading.excepthook
    threading.excepthook = lambda args: escaped.append(args.exc_value)
    try:
        worker = _run_in_thread(pipeline)
        worker.join(5)
    finally:
        threading.excepthook = previous

    # An escaping exception would be printed with its message, which can carry
    # recognized text.
    assert escaped == []
    assert state.stopped is True


def test_watchdog_fires_neutral_and_stops_when_output_stalls():
    clock = FakeClock()
    pipeline, publisher, state = _pipeline(
        BlockingRecognizer(),
        watchdog_timeout=2.0,
        watchdog_first_frame=2.0,
        monotonic=clock,
    )
    state.go_live()
    pipeline.submit(_frame())  # waits, never processed

    live = pipeline._watchdog_tick(False)  # live edge: baseline at t=0
    assert live is True
    clock.t = 1.0
    assert pipeline._watchdog_tick(True) is True  # within timeout
    clock.t = 3.0
    assert pipeline._watchdog_tick(True) is False  # gap exceeded

    assert pipeline.watchdog_fired == 1
    assert publisher.neutrals != []
    assert state.stopped is True


def test_watchdog_does_not_fire_while_paused():
    clock = FakeClock()
    pipeline, _, state = _pipeline(
        FakeRecognizer(),
        watchdog_timeout=2.0,
        watchdog_first_frame=2.0,
        monotonic=clock,
    )
    state.go_live()
    pipeline.process(_frame(), 1, state.epoch)  # first publish ends the grace

    was_live = pipeline._watchdog_tick(False)  # live, baseline at t=0
    clock.t = 1.0
    was_live = pipeline._watchdog_tick(was_live)  # live, near firing, no fire
    assert was_live is True
    state.pause()
    for t in (3.0, 6.0):  # well past the timeout, but paused
        clock.t = t
        was_live = pipeline._watchdog_tick(was_live)

    assert pipeline.watchdog_fired == 0
    assert state.stopped is False


def test_watchdog_gives_the_first_frame_a_wider_budget():
    clock = FakeClock()
    pipeline, publisher, state = _pipeline(
        BlockingRecognizer(),
        watchdog_timeout=2.0,
        watchdog_first_frame=10.0,
        monotonic=clock,
    )
    state.go_live()
    pipeline.submit(_frame())  # waits, never processed

    pipeline._watchdog_tick(False)  # baseline, first-frame budget armed
    clock.t = 5.0  # past the 2s steady timeout, under the 10s first-frame grace
    assert pipeline._watchdog_tick(True) is True
    assert pipeline.watchdog_fired == 0
    clock.t = 11.0  # now past the first-frame grace
    assert pipeline._watchdog_tick(True) is False
    assert publisher.neutrals != []


def test_watchdog_fire_notifies_session_stopped():
    clock = FakeClock()
    stopped_calls = []
    pipeline, _, state = _pipeline(
        BlockingRecognizer(),
        watchdog_timeout=2.0,
        watchdog_first_frame=2.0,
        monotonic=clock,
        on_session_stopped=lambda: stopped_calls.append(True),
    )
    state.go_live()
    pipeline.submit(_frame())  # waits, never processed
    pipeline._watchdog_tick(False)
    clock.t = 3.0
    pipeline._watchdog_tick(True)

    assert stopped_calls == [True]


def test_per_frame_failure_does_not_notify_session_stopped():
    stopped_calls = []
    pipeline, _, state = _pipeline(
        FailingRecognizer(),
        on_session_stopped=lambda: stopped_calls.append(True),
    )
    state.go_live()
    pipeline.process(_frame(), 1, state.epoch)

    assert stopped_calls == []  # a failed frame goes neutral but keeps running


def test_watchdog_narrows_to_steady_timeout_after_first_publish():
    clock = FakeClock()
    pipeline, publisher, state = _pipeline(
        FakeRecognizer(),
        watchdog_timeout=2.0,
        watchdog_first_frame=10.0,
        monotonic=clock,
    )
    state.go_live()
    pipeline._watchdog_tick(False)  # live edge: first-frame budget armed
    pipeline.process(_frame(), 1, state.epoch)  # publishes, ends the grace
    pipeline.submit(_frame())  # waits, never processed

    clock.t = 3.0  # past the 2s steady timeout, under the 10s first-frame grace
    assert pipeline._watchdog_tick(True) is False  # steady budget now applies
    assert pipeline.watchdog_fired == 1
    assert publisher.neutrals != []


@pytest.mark.parametrize("stuck", ["pending", "in_progress"])
def test_stuck_frame_fires_without_any_input(stuck):
    clock = FakeClock()
    pipeline, publisher, state = _pipeline(
        FakeRecognizer(), watchdog_timeout=2.0, monotonic=clock
    )
    state.go_live()
    pipeline._watchdog_tick(False)
    pipeline.process(_frame(), 1, state.epoch)
    pipeline.submit(_frame())
    if stuck == "in_progress":
        assert pipeline._take() is not None  # taken, never finished
    clock.t = 3.0
    assert pipeline._watchdog_tick(True) is False
    assert pipeline.watchdog_fired == 1
    assert publisher.neutrals != []


def test_watchdog_loop_fires_through_the_real_thread():
    """Exercises _watchdog_loop, run()'s start/join, and was_live threading with
    a real worker and real (short) timings — no fake clock."""
    recognizer = BlockingRecognizer()
    pipeline, publisher, state = _pipeline(
        recognizer, watchdog_timeout=0.3, watchdog_first_frame=0.3, watchdog_poll=0.02
    )
    state.go_live()
    pipeline.submit(_frame())
    worker = _run_in_thread(pipeline)
    assert recognizer.entered.wait(5)

    deadline = time.monotonic() + 5
    while not publisher.neutrals and time.monotonic() < deadline:
        time.sleep(0.02)

    assert publisher.neutrals != []
    assert state.stopped is True
    recognizer.release.set()
    worker.join(5)
    assert publisher.frames == []


def test_progress_resets_on_the_live_edge_after_resume():
    clock = FakeClock()
    pipeline, _, state = _pipeline(
        FakeRecognizer(), watchdog_timeout=2.0, monotonic=clock
    )
    state.go_live()
    clock.t = 10.0  # a long pause elapses on the clock
    state.pause()
    state.go_live()

    assert pipeline._watchdog_tick(False) is True  # resume edge rebaselines
    clock.t = 11.0
    assert pipeline._watchdog_tick(True) is True  # not the 10s pause gap
    assert pipeline.watchdog_fired == 0


def test_republish_resets_the_watchdog_gap():
    clock = FakeClock()
    pipeline, _, state = _pipeline(
        FakeRecognizer(), watchdog_timeout=2.0, monotonic=clock
    )
    state.go_live()
    pipeline.process(_frame(), 1, state.epoch)  # first OCR, marks progress
    clock.t = 1.5
    pipeline.process(_frame(), 2, state.epoch)  # republish, marks progress again
    clock.t = 3.0  # 1.5s since the republish, under the 2s timeout

    assert pipeline._watchdog_tick(True) is True
    assert pipeline.watchdog_fired == 0


def test_hung_worker_publishes_nothing_after_the_watchdog_fires():
    """The un-hang guarantee: a frame that finishes OCR after the watchdog has
    stopped the session is dropped, not published over neutral."""
    clock = FakeClock()
    recognizer = BlockingRecognizer()
    pipeline, publisher, state = _pipeline(
        recognizer,
        watchdog_timeout=2.0,
        watchdog_first_frame=2.0,
        monotonic=clock,
    )
    state.go_live()
    pipeline.submit(_frame())
    worker = _run_in_thread(pipeline)
    assert recognizer.entered.wait(5)  # worker is inside recognize()

    pipeline._watchdog_tick(False)
    clock.t = 3.0
    pipeline._watchdog_tick(True)  # fires: stop + neutral
    recognizer.release.set()  # worker un-hangs and tries to publish
    worker.join(5)

    assert publisher.frames == []
    assert pipeline.dropped_epoch == 1


def test_published_frame_is_masked():
    pipeline, publisher, state = _pipeline(FakeRecognizer())
    state.go_live()

    pipeline.process(_frame(), 1, state.epoch)

    assert len(publisher.frames) == 1
    _, _, out = publisher.frames[0]
    x0, y0, x1, y1 = _EMAIL_BOX
    assert not (out[y0:y1, x0:x1, 0:3] == 200).all(axis=-1).any()


def _page(fill: int = 210) -> np.ndarray:
    """A larger blank frame that the 128x128 signature can pool meaningfully."""
    frame = np.full((720, 1280, 4), fill, dtype=np.uint8)
    frame[..., 3] = 255
    return frame


def _with_text_line(frame: np.ndarray) -> np.ndarray:
    """Add one dark 12px-tall line, ~0.3% of frame area, like a line of text."""
    out = frame.copy()
    out[300:312, 100:520, 0:3] = 20
    return out


def test_identical_frames_are_unchanged():
    a = frame_signature(_page())
    b = frame_signature(_page())
    assert unchanged(a, b) is True


def test_new_text_line_is_detected_as_changed():
    """The reported failure: a single line appearing scored ~0 under the old
    mean-of-subsample comparator and was skipped."""
    before = frame_signature(_page())
    after = frame_signature(_with_text_line(_page()))
    assert unchanged(before, after) is False


def test_thin_line_between_subsample_rows_is_detected():
    """At Retina size a 16px line can fall entirely between the rows a strided
    subsample would keep; area pooling must still see it."""
    base = np.full((1964, 3024, 4), 210, dtype=np.uint8)
    base[..., 3] = 255
    changed = base.copy()
    # Row 1005 sits between multiples of 1964//64 == 30 (990, 1020).
    changed[1005:1021, 400:1200, 0:3] = 20
    assert unchanged(frame_signature(base), frame_signature(changed)) is False


def test_max_skip_age_forces_reocr_on_a_static_screen():
    recognizer = FakeRecognizer()
    pipeline, _, state = _pipeline(recognizer)
    state.go_live()

    for seq in range(_MAX_SKIP_FRAMES + 2):
        pipeline.process(_page(), seq + 1, state.epoch)

    # One OCR to establish the frame, one forced after _MAX_SKIP_FRAMES skips.
    assert recognizer.calls == 2
    assert pipeline.republished == _MAX_SKIP_FRAMES


def test_unchanged_frame_republishes_the_sanitized_frame():
    recognizer = FakeRecognizer()
    pipeline, publisher, state = _pipeline(recognizer)
    state.go_live()

    pipeline.process(_frame(), 1, state.epoch)
    second = _frame()
    pipeline.process(second, 2, state.epoch)

    assert recognizer.calls == 1
    assert pipeline.republished == 1
    assert publisher.frames[1][2] is not second


def test_stop_is_terminal():
    pipeline, publisher, state = _pipeline(FakeRecognizer())
    state.go_live()
    epoch = state.epoch
    state.stop()

    assert state.go_live() is False
    pipeline.process(_frame(), 1, epoch)
    assert publisher.frames == []


def test_older_slot_frames_are_dropped():
    pipeline, _, state = _pipeline(FakeRecognizer())
    state.go_live()

    pipeline.submit(_frame())
    second_seq = pipeline.submit(_frame())

    assert pipeline.dropped_stale == 1
    assert pipeline.pending_seq == second_seq


@pytest.mark.parametrize(
    ("accepted", "seq", "last_seq", "expected"),
    [
        (True, 2, 1, True),
        (False, 2, 1, False),  # paused, stopped or stale epoch
        (True, 1, 1, False),  # already on screen
        (True, 1, 2, False),  # out of order
    ],
)
def test_should_deliver(accepted, seq, last_seq, expected):
    from safe_share.app import should_deliver

    assert should_deliver(accepted, seq, last_seq) is expected


def test_lost_frame_goes_neutral_and_clears_the_cache():
    recognizer = FakeRecognizer()
    pipeline, publisher, state = _pipeline(recognizer)
    state.go_live()
    pipeline.process(_frame(), 1, state.epoch)
    pipeline.process(None, 2, state.epoch)
    assert publisher.neutrals == [state.epoch]
    pipeline.process(_frame(), 3, state.epoch)  # same pixels, but re-OCR'd
    assert recognizer.calls == 2
    assert pipeline.republished == 0


def test_lost_frame_while_paused_publishes_nothing():
    pipeline, publisher, state = _pipeline(FakeRecognizer())
    state.go_live()
    epoch = state.epoch
    state.pause()
    pipeline.process(None, 1, epoch)
    assert publisher.neutrals == [] and publisher.frames == []


def test_a_newer_frame_replaces_a_lost_marker():
    pipeline, _, state = _pipeline(FakeRecognizer())
    state.go_live()
    pipeline.lost()
    seq = pipeline.submit(_frame())
    taken = pipeline._take()
    assert taken is not None and taken[0] is not None and taken[1] == seq
