"""Offline tests for control-window state. No AppKit."""

from __future__ import annotations

import pytest

from safe_share.control_state import (
    ControlState,
    button_enablement,
    control_state,
    status_text,
)


@pytest.mark.parametrize(
    ("live", "stopped", "expected"),
    [
        (True, False, ControlState.RUNNING),
        (False, False, ControlState.PAUSED),
        (False, True, ControlState.STOPPED),
        (True, True, ControlState.STOPPED),  # stopped wins over a stale live flag
    ],
)
def test_control_state(live, stopped, expected):
    assert control_state(live, stopped) is expected


def test_running_allows_pause_and_stop_only():
    assert button_enablement(ControlState.RUNNING) == {
        "pause": True,
        "resume": False,
        "stop": True,
    }


def test_paused_allows_resume_and_stop_only():
    assert button_enablement(ControlState.PAUSED) == {
        "pause": False,
        "resume": True,
        "stop": True,
    }


def test_stopped_disables_every_transition():
    assert button_enablement(ControlState.STOPPED) == {
        "pause": False,
        "resume": False,
        "stop": False,
    }


def test_stopped_status_names_the_dead_session():
    assert "quit" in status_text(ControlState.STOPPED).lower()
