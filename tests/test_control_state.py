"""Offline tests for control-window state. No AppKit."""

from __future__ import annotations

import pytest

from safe_share.control_state import ControlState, control_state, control_view


@pytest.mark.parametrize(
    ("live", "stopped", "started", "expected"),
    [
        (True, False, True, ControlState.RUNNING),
        (True, False, False, ControlState.STARTING),
        (False, False, True, ControlState.PAUSED),
        (False, True, True, ControlState.STOPPED),
        (True, True, True, ControlState.STOPPED),  # stopped wins over a stale live flag
    ],
)
def test_control_state(live, stopped, started, expected):
    assert control_state(live, stopped, started) is expected


def test_stopped_disables_every_transition():
    view = control_view(ControlState.STOPPED)
    assert not view.toggle_enabled
    assert not view.stop_enabled


def test_stopped_status_names_the_dead_session():
    assert "relaunch" in control_view(ControlState.STOPPED).status.lower()


def test_paused_toggle_resumes():
    assert control_view(ControlState.PAUSED).toggle_title == "Resume"
