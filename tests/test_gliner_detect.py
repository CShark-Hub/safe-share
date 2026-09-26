"""GLiNER detector tests with a fake model; the real-model tests need the cache."""

from __future__ import annotations

import importlib.util
import os
import socket
from pathlib import Path
from typing import Any

import pytest

from safe_share.detect import Detection, RuleConfig, RuleDetector, UnionDetector
from safe_share.gliner_detect import MODEL_ID, MODEL_REVISION, GlinerDetector


class FakeModel:
    """Tags every occurrence of the given names."""

    def __init__(self, names: dict[str, str]) -> None:
        self._names = names

    def inference(
        self,
        texts: list[str],
        labels: list[str],
        threshold: float = 0.5,
        batch_size: int = 8,
    ) -> list[list[dict[str, Any]]]:
        out = []
        for text in texts:
            ents = []
            for name, label in self._names.items():
                start = text.find(name)
                while start != -1:
                    ents.append(
                        {
                            "start": start,
                            "end": start + len(name),
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
    texts = [
        "",
        "Lena Hartwig 10:42",
        "   ",
        "did the invoice go out?",
        "ask Fabrikam today",
    ]
    model = FakeModel({"Lena Hartwig": "person", "Fabrikam": "organization"})
    assert _spans(GlinerDetector(model).detect(texts), texts) == {
        (1, "Lena Hartwig", "person"),
        (4, "Fabrikam", "organization"),
    }


def test_name_split_across_lines_marks_both_lines() -> None:
    texts = ["reviewed by Rafael", "Quintero yesterday"]
    dets = GlinerDetector(FakeModel({"Rafael\nQuintero": "person"})).detect(texts)
    assert _spans(dets, texts) == {(0, "Rafael", "person"), (1, "Quintero", "person")}


def test_punctuation_counts_toward_the_word_budget() -> None:
    # 3 whitespace words, 13 words as gliner splits them.
    with pytest.raises(ValueError):
        GlinerDetector(FakeModel({}), max_chunk_words=10).detect(["a.b.c.d(e,f) = g"])


def test_chunks_respect_the_subword_budget() -> None:
    class Tokenizer:
        """One subword per character, so long words cost more."""

        def __call__(self, lines, is_split_into_words, add_special_tokens):
            return {"input_ids": [[0] * sum(len(w) for w in words) for words in lines]}

    class WithTokenizer(FakeModel):
        def __init__(self, names: dict[str, str]) -> None:
            super().__init__(names)
            self.data_processor = type(
                "P", (), {"transformer_tokenizer": Tokenizer()}
            )()
            self.chunks: list[str] = []

        def inference(self, texts, labels, threshold=0.5, batch_size=8):
            self.chunks += texts
            return super().inference(texts, labels, threshold, batch_size)

    model = WithTokenizer({"Ada": "person"})
    texts = ["Ada " + "x" * 200 for _ in range(10)]  # ~200 subwords per line
    dets = GlinerDetector(model).detect(texts)
    assert all(len(c.replace("\n", "").replace(" ", "")) <= 480 for c in model.chunks)
    assert {d.line_index for d in dets} == set(range(10))

    with pytest.raises(ValueError):  # a single line over the cap fails, never truncates
        GlinerDetector(model).detect(["x" * 600])


def test_result_count_mismatch_is_a_failure() -> None:
    class Short(FakeModel):
        def inference(self, texts, labels, threshold=0.5, batch_size=8):
            return []

    with pytest.raises(RuntimeError):
        GlinerDetector(Short({})).detect(["Ada here"])


def test_union_keeps_hits_from_every_detector_and_propagates_failures() -> None:
    texts = ["mail ada@example.com", "ask Priya Venkat"]
    union = UnionDetector(
        RuleDetector(RuleConfig(terms=[])),
        GlinerDetector(FakeModel({"Priya Venkat": "person"})),
    )
    assert _spans(union.detect(texts), texts) == {
        (0, "ada@example.com", "email"),
        (1, "Priya Venkat", "person"),
    }

    class Broken:
        def detect(self, texts: list[str]) -> list[Detection]:
            raise RuntimeError("model failed")

    with pytest.raises(RuntimeError):
        UnionDetector(RuleDetector(RuleConfig(terms=[])), Broken()).detect(texts)


def _model_cached() -> bool:
    # Must not import huggingface_hub: it reads HF_HUB_OFFLINE at import time.
    if importlib.util.find_spec("gliner") is None:
        return False
    hub = (
        os.environ.get("HF_HUB_CACHE")
        or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    )
    repo = f"models--{MODEL_ID.replace('/', '--')}"
    return (Path(hub) / repo / "snapshots" / MODEL_REVISION).is_dir()


needs_model = pytest.mark.skipif(
    not _model_cached(), reason="ml extra or cached model missing"
)


@needs_model
def test_warm_up_that_finds_nothing_falls_back_to_the_next_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from safe_share import gliner_detect

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    models = {"mps": FakeModel({}), "cpu": FakeModel({"Priya Venkat": "person"})}
    monkeypatch.setattr(
        gliner_detect, "load_model", lambda device, dtype: models[device]
    )
    assert gliner_detect.load_best()[1] == "cpu/float32"


@needs_model
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
