"""Leak guards for the session name cache."""

from __future__ import annotations

from safe_share.detect import Detection
from safe_share.name_cache import NameCacheDetector

_NAME = "Amara Osei"  # synthetic


class FlakyModel:
    """Finds `names` only while `on` is set, like GLiNER flipping between frames."""

    def __init__(self, *names: str) -> None:
        self.names = names or (_NAME,)
        self.on = True

    def detect(self, texts: list[str]) -> list[Detection]:
        if not self.on:
            return []
        return [
            Detection(i, t.index(n), t.index(n) + len(n), "person", "gliner")
            for i, t in enumerate(texts)
            for n in self.names
            if n in t
        ]


def _covered(dets: list[Detection], text: str, word: str) -> bool:
    a = text.index(word)
    return any(d.start <= a and a + len(word) <= d.end for d in dets)


def test_name_seen_once_stays_masked_when_the_model_misses_it() -> None:
    model = FlakyModel()
    cache = NameCacheDetector(model)
    cache.detect([f"call with {_NAME} at noon"])
    model.on = False
    for text in (f"Re: {_NAME}", "AMARA OSEI joined", "thanks Amara!"):
        assert _covered(cache.detect(["header", text]), text, text.split()[1][:4])


def test_single_word_needs_a_capital_in_the_line() -> None:
    model = FlakyModel()
    cache = NameCacheDetector(model)
    cache.detect([_NAME])
    model.on = False
    assert cache.detect(["the amara variable"]) == []


def test_single_word_is_not_matched_through_ocr_folding() -> None:
    model = FlakyModel("Ali Khan")
    cache = NameCacheDetector(model)
    cache.detect(["Ali Khan"])
    model.on = False
    assert cache.detect(["All set"]) == []  # folds to "all", like "Ali"


def test_clear_forgets_and_admits_nothing_after() -> None:
    model = FlakyModel()
    cache = NameCacheDetector(model)
    cache.detect([_NAME])
    cache.clear()
    cache.detect([_NAME])  # a late frame after Stop
    model.on = False
    assert len(cache) == 0
    assert cache.detect([_NAME]) == []


def test_full_cache_refuses_instead_of_evicting() -> None:
    model = FlakyModel(_NAME, "Tomasz Lindqvist")
    cache = NameCacheDetector(model, max_names=3)
    cache.detect([_NAME])  # phrase + 2 words fills it
    cache.detect(["Tomasz Lindqvist"])
    model.on = False
    assert cache.refused == 1
    assert cache.detect([f"hi {_NAME}"])
