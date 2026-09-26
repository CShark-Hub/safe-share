"""Offline tests for the fixture evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from safe_share.detect import Detection
from safe_share.eval_detect import run


class WordDetector:
    """Flags every occurrence of one word as a person."""

    def __init__(self, word: str) -> None:
        self._word = word

    def detect(self, texts: list[str]) -> list[Detection]:
        return [
            Detection(
                i, t.index(self._word), t.index(self._word) + len(self._word), "person"
            )
            for i, t in enumerate(texts)
            if self._word in t
        ]


@pytest.fixture
def fixtures(tmp_path: Path) -> Path:
    path = tmp_path / "names.json"
    path.write_text(
        json.dumps(
            {
                "terms": [],
                "cases": [
                    {
                        "id": "chat",
                        "tags": ["en"],
                        "lines": ["Ada 10:42", "hello", "Bo 10:43"],
                        "expect": [
                            {"line": 0, "text": "Ada", "type": "person"},
                            {"line": 2, "text": "Bo", "type": "person"},
                        ],
                    },
                    {
                        "id": "neg",
                        "tags": ["ui"],
                        "lines": ["Ada Lovelace Bridge"],
                        "expect": [],
                    },
                ],
            }
        )
    )
    return path


def test_runs_a_supplied_detector_and_reports_line_coverage(
    fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(fixtures, detector=WordDetector("Ada"))
    out = capsys.readouterr().out

    assert code == 1  # one leak (Bo) and one false positive
    rows = {line.split()[0]: line for line in out.splitlines() if line.strip()}
    assert "1/2 (50%)" in rows["type:person"]
    assert "1/1 (100%)" in rows["ui"]


def test_clean_detector_exits_zero(
    fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class Perfect:
        def detect(self, texts: list[str]) -> list[Detection]:
            return [
                Detection(i, 0, len(t.split()[0]), "person")
                for i, t in enumerate(texts)
                if t.split()[0] in {"Ada", "Bo"} and t.endswith(("42", "43"))
            ]

    assert run(fixtures, detector=Perfect()) == 0
    assert "exposures (leaks): 0" in capsys.readouterr().out


def test_missing_fixture_file_returns_2(tmp_path: Path) -> None:
    assert run(tmp_path / "absent.json") == 2
