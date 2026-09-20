"""Offline tests for ScreenCaptureKit stop classification. No macOS frameworks."""

from __future__ import annotations

import pytest

from safe_share.capture_errors import StopDisposition, classify_stop


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (None, StopDisposition.CLEAN),
        (-3817, StopDisposition.CLEAN),  # userStopped
        (-3805, StopDisposition.RECOVERABLE),  # connection interrupted
        (-3813, StopDisposition.RECOVERABLE),  # noWindowList
        (-3814, StopDisposition.RECOVERABLE),  # noDisplayList
        (-3815, StopDisposition.RECOVERABLE),  # noCaptureSource
        (-3801, StopDisposition.TERMINAL),  # userDeclined (permission)
        (-3803, StopDisposition.TERMINAL),  # missingEntitlements
        (-3812, StopDisposition.TERMINAL),  # invalidParameter (our bug)
        (0, StopDisposition.TERMINAL),  # unknown code
    ],
)
def test_classify_stop(code, expected):
    assert classify_stop(code) is expected


def test_user_stopped_is_not_an_error():
    assert classify_stop(-3817) is StopDisposition.CLEAN
