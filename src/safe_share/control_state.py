"""Control-window state derived from the session, free of AppKit.

Maps the session's (live, stopped) flags to which controls are legal, so the
buttons cannot invoke a transition the session will refuse (a stopped session
never resumes) and a dead session is visible rather than silent.
"""

from __future__ import annotations

from enum import Enum


class ControlState(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"  # terminal: only restarting the process continues


def control_state(live: bool, stopped: bool) -> ControlState:
    if stopped:
        return ControlState.STOPPED
    return ControlState.RUNNING if live else ControlState.PAUSED


_ENABLED: dict[ControlState, dict[str, bool]] = {
    ControlState.RUNNING: {"pause": True, "resume": False, "stop": True},
    ControlState.PAUSED: {"pause": False, "resume": True, "stop": True},
    ControlState.STOPPED: {"pause": False, "resume": False, "stop": False},
}

_STATUS: dict[ControlState, str] = {
    ControlState.RUNNING: "Protecting",
    ControlState.PAUSED: "Paused — output hidden",
    ControlState.STOPPED: "Stopped — quit and relaunch to protect again",
}


def button_enablement(state: ControlState) -> dict[str, bool]:
    """Enabled flag per control button (Quit is always enabled, so omitted)."""
    return _ENABLED[state]


def status_text(state: ControlState) -> str:
    return _STATUS[state]
