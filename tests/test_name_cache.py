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


def test_common_word_is_remembered_only_inside_its_full_name() -> None:
    model = FlakyModel("Grace Whitfield", "After", "GLiNER")
    cache = NameCacheDetector(model, frozenset({"grace", "after"}))
    cache.detect(["Grace Whitfield", "After", "+ GLiNER"])
    model.on = False
    assert _covered(
        cache.detect(["re: grace whitfield"]), "re: grace whitfield", "grace"
    )
    for text in ("Grace", "After the call", "+ GLiNER"):
        assert cache.detect([text]) == []


def test_rare_name_is_still_remembered_alone() -> None:
    model = FlakyModel("Kasia", "PriyaVenkat")
    cache = NameCacheDetector(model, frozenset({"after", "rules"}))
    cache.detect(["Kasia", "Meeting with PriyaVenkat"])
    model.on = False
    for text, word in (("Kasia", "Kasia"), ("Re: PriyaVenkat", "PriyaVenkat")):
        assert _covered(cache.detect([text]), text, word)


def test_common_words_compares_lowercase_and_capitalised_scores() -> None:
    import json
    from types import SimpleNamespace

    from safe_share.gliner_detect import common_words

    vocab = [["▁after", -5.0], ["▁After", -8.0], ["▁kasia", -12.0], ["▁Kasia", -9.0]]
    vocab += [["▁rules", -7.0], ["ing", -3.0], ["▁GLiNER", -11.0]]
    spec = json.dumps({"model": {"type": "Unigram", "vocab": vocab}})
    backend = SimpleNamespace(to_str=lambda: spec)
    tok = SimpleNamespace(backend_tokenizer=backend)
    model = SimpleNamespace(data_processor=SimpleNamespace(transformer_tokenizer=tok))
    assert common_words(model) == frozenset({"after", "rules"})


def test_unreadable_tokenizer_gives_no_common_words() -> None:
    from safe_share.gliner_detect import common_words

    assert common_words(object()) == frozenset()


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
