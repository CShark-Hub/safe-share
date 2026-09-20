"""Offline tests for the rule detector's handling of private terms."""

from __future__ import annotations

from safe_share.detect import RuleConfig, RuleDetector

_TERM = "Halibut"


def test_repr_does_not_expose_terms() -> None:
    cfg = RuleConfig(terms=[_TERM])
    assert _TERM not in repr(cfg)
    assert _TERM not in str(cfg)


def test_repr_does_not_expose_terms_through_the_detector() -> None:
    detector = RuleDetector(RuleConfig(terms=[_TERM]))
    assert _TERM not in repr(detector)


def test_terms_still_match_after_being_hidden_from_repr() -> None:
    detector = RuleDetector(RuleConfig(emails=False, credentials=False, terms=[_TERM]))
    assert detector.detect([f"project {_TERM.lower()} notes"])
