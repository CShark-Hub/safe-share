"""Classify a ScreenCaptureKit stop reason, free of PyObjC.

Numeric codes are SCStreamError.Code values (SCError.h), kept as literals so
this module and its tests need no macOS frameworks.
"""

from __future__ import annotations

from enum import Enum

SC_ERROR_DOMAIN = "SCStreamErrorDomain"

# The user or system ended the stream intentionally; documented as not an error.
_USER_STOPPED = -3817

# Transient conditions where a fresh stream could be built (connection dropped,
# display list changed). Recovery itself is not implemented yet.
_RECOVERABLE = frozenset(
    {
        -3804,  # failedApplicationConnectionInvalid
        -3805,  # failedApplicationConnectionInterrupted
        -3806,  # failedNoMatchingApplicationContext
        -3811,  # internalError
        -3813,  # noWindowList
        -3814,  # noDisplayList
        -3815,  # noCaptureSource
    }
)


class StopDisposition(Enum):
    CLEAN = "clean"  # intentional stop; not a failure
    RECOVERABLE = "recoverable"  # transient; a rebuilt stream might continue
    TERMINAL = "terminal"  # cannot continue (permission, entitlements, our bug)


def classify_stop(code: int | None) -> StopDisposition:
    """Disposition for a stream stop. A missing code is treated as a clean stop."""
    if code is None or code == _USER_STOPPED:
        return StopDisposition.CLEAN
    if code in _RECOVERABLE:
        return StopDisposition.RECOVERABLE
    return StopDisposition.TERMINAL
