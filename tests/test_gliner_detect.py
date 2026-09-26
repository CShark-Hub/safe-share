"""GLiNER detector tests with a fake model; the real-model test needs the cache."""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path
from typing import Any

import pytest

from safe_share.detect import Detection, RuleConfig, RuleDetector, UnionDetector
from safe_share.gliner_detect import GlinerDetector


class FakeModel:
    """Tags every occurrence of the given names and records its inputs."""

    def __init__(self, names: dict[str, str], max_words: int | None = None) -> None:
        self._names = names
        self._max_words = max_words
        self.calls: list[list[str]] = []
        self.threshold: float | None = None

    def inference(
        self,
        texts: list[str],
        labels: list[str],
        threshold: float = 0.5,
        batch_size: int = 8,
    ) -> list[list[dict[str, Any]]]:
        self.calls.append(texts)
        self.threshold = threshold
        out = []
        for text in texts:
            if self._max_words is not None:
                assert len(text.split()) <= self._max_words, "chunk over budget"
            ents = []
            for name, label in self._names.items():
                start = text.find(name)
                while start != -1:
                    ents.append(
                        {
                            "start": start,
                            "end": start + len(name),
                            "text": name,
                            "label": label,
                            "score": 0.9,
                        }
                    )
                    start = text.find(name, start + 1)
            out.append(ents)
        return out


def _spans(dets: list[Detection], texts: list[str]) -> set[tuple[int, str, str]]:
    return {
        (d.line_index, texts[d.line_index][d.start : d.end], d.entity_type)
        for d in dets
    }


def test_chunk_offsets_map_back_to_the_right_lines() -> None:
    texts = ["Lena Hartwig 10:42", "did the invoice go out?", "ask Fabrikam today"]
    model = FakeModel({"Lena Hartwig": "person", "Fabrikam": "organization"})
    dets = GlinerDetector(model).detect(texts)

    assert len(model.calls) == 1 and len(model.calls[0]) == 1  # one chunk
    assert _spans(dets, texts) == {
        (0, "Lena Hartwig", "person"),
        (2, "Fabrikam", "organization"),
    }
    assert all(d.source == "gliner" and d.score == 0.9 for d in dets)


def test_name_split_across_lines_marks_both_lines() -> None:
    texts = ["reviewed by Rafael", "Quintero yesterday"]
    dets = GlinerDetector(FakeModel({"Rafael\nQuintero": "person"})).detect(texts)
    assert _spans(dets, texts) == {(0, "Rafael", "person"), (1, "Quintero", "person")}


def test_chunks_never_exceed_the_word_budget() -> None:
    texts = [f"line {i} mentions Ada Kowalczyk here" for i in range(40)]  # 6 words each
    model = FakeModel({"Ada Kowalczyk": "person"}, max_words=20)
    dets = GlinerDetector(model, max_chunk_words=20).detect(texts)

    assert sum(len(c) for c in model.calls) == 14  # 3 lines per chunk
    assert {d.line_index for d in dets} == set(range(40))


def test_line_over_budget_raises_instead_of_truncating() -> None:
    with pytest.raises(ValueError):
        GlinerDetector(FakeModel({}), max_chunk_words=5).detect(
            ["one two three four five six"]
        )


def test_result_count_mismatch_is_a_failure() -> None:
    class Short(FakeModel):
        def inference(self, texts, labels, threshold=0.5, batch_size=8):
            return []

    with pytest.raises(RuntimeError):
        GlinerDetector(Short({})).detect(["Ada here"])


def test_union_keeps_hits_from_every_detector() -> None:
    texts = ["mail ada@example.com", "ask Priya Venkat"]
    union = UnionDetector(
        RuleDetector(RuleConfig(terms=[])),
        GlinerDetector(FakeModel({"Priya Venkat": "person"})),
    )
    assert _spans(union.detect(texts), texts) == {
        (0, "ada@example.com", "email"),
        (1, "Priya Venkat", "person"),
    }


def _model_cached() -> bool:
    if importlib.util.find_spec("gliner") is None:
        return False
    from huggingface_hub.constants import HF_HUB_CACHE

    from safe_share.gliner_detect import MODEL_ID, MODEL_REVISION

    repo = f"models--{MODEL_ID.replace('/', '--')}"
    return (Path(HF_HUB_CACHE) / repo / "snapshots" / MODEL_REVISION).is_dir()


@pytest.mark.skipif(not _model_cached(), reason="ml extra or cached model missing")
def test_real_model_loads_and_runs_with_the_network_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)

    from safe_share.gliner_detect import load_model

    texts = ["# TODO ask Priya Venkat about the retry logic", "File  Edit  View  Help"]
    dets = GlinerDetector(load_model()).detect(texts)
    assert (0, "Priya Venkat", "person") in _spans(dets, texts)
    assert all(d.line_index == 0 for d in dets)
