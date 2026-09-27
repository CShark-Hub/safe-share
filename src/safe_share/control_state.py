"""Control-window state derived from the session, free of AppKit.

Maps the session's (live, stopped) flags to which controls are legal, so the
buttons cannot invoke a transition the session will refuse (a stopped session
never resumes) and a dead session is visible rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ControlState(Enum):
    STARTING = "starting"  # live, no frame published yet
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"  # terminal: only restarting the process continues


def control_state(live: bool, stopped: bool, started: bool = True) -> ControlState:
    if stopped:
        return ControlState.STOPPED
    if not live:
        return ControlState.PAUSED
    return ControlState.RUNNING if started else ControlState.STARTING


@dataclass(frozen=True)
class ControlView:
    status: str
    dot: str  # "green" | "orange" | "red" | "gray"
    toggle_title: str
    toggle_enabled: bool
    stop_enabled: bool


def control_view(state: ControlState, names_off: bool = False) -> ControlView:
    """Status line and button states for the controls window (Quit is always on)."""
    if state is ControlState.STARTING:
        return ControlView("Starting protection…", "gray", "Pause", True, True)
    if state is ControlState.RUNNING:
        status = "Protecting your screen" + (" (names off)" if names_off else "")
        return ControlView(status, "green", "Pause", True, True)
    if state is ControlState.PAUSED:
        return ControlView(
            "Paused · viewers see a hidden screen", "orange", "Resume", True, True
        )
    return ControlView(
        "Stopped · relaunch to share again", "red", "Resume", False, False
    )
