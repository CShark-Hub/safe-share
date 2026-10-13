"""Session memory of model-detected person names, matched on every later frame."""

from __future__ import annotations

import re
import threading
from itertools import pairwise

from .detect import Detection, Detector, _fold, _normalize

MAX_NAMES = 256
_EDGE_PUNCT = re.compile(r"^\W+|\W+$")


def _name_word(w: str) -> bool:
    """A word that may match on its own: alphabetic, 3+ letters, capitalised, no two capitals in a row."""
    return (
        w.isalpha()
        and len(w) >= 3
        and w[0].isupper()
        and not any(a.isupper() and b.isupper() for a, b in pairwise(w))
    )


class NameCacheDetector:
    """Wraps a detector; GLiNER person spans are remembered for the session.

    Multi-word names match case-insensitively; single words only where the line
    has them capitalised; words in `common` only as part of a multi-word name.
    Entries live in memory only, are never logged, and go on `clear()`. A full
    cache refuses new names rather than evicting old ones.
    """

    def __init__(
        self,
        inner: Detector,
        common: frozenset[str] = frozenset(),
        max_names: int = MAX_NAMES,
    ) -> None:
        self._inner = inner
        self._common = common
        self._max = max_names
        self._lock = threading.Lock()
        self._phrases: set[str] = set()  # folded, single-spaced, 2+ words
        # Casefolded, not OCR-folded: a fold match on one word ("Ali" ~ "All")
        # is too loose, so single-word hits are checked against these exactly.
        self._words: set[str] = set()
        self._pattern: re.Pattern[str] | None = None
        self._word_set: frozenset[str] = frozenset()
        self._dirty = False
        self._closed = False
        self.refused = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._phrases) + len(self._words)

    def clear(self) -> None:
        """Forget every name; later detections are no longer admitted."""
        with self._lock:
            self._phrases.clear()
            self._words.clear()
            self._pattern = None
            self._word_set = frozenset()
            self._closed = True

    def detect(self, texts: list[str]) -> list[Detection]:
        found = self._inner.detect(texts)
        lines = [_normalize(t) for t in texts]
        with self._lock:
            for d in found:
                if d.source == "gliner" and d.entity_type == "person":
                    self._admit(_normalize(texts[d.line_index][d.start : d.end]))
            pattern = self._compiled()
            words = self._word_set
        if pattern is None:
            return found
        out = list(found)
        for i, text in enumerate(lines):
            # _fold is 1:1 per character, so spans on the folded line index `text`.
            for m in pattern.finditer(_fold(text)):
                if len(m.group().split()) == 1 and (
                    not text[m.start()].isupper()
                    or text[m.start() : m.end()].casefold() not in words
                ):
                    continue
                if any(
                    d.line_index == i and d.start <= m.start() and m.end() <= d.end
                    for d in found
                ):
                    continue
                out.append(Detection(i, m.start(), m.end(), "person", "name_cache"))
        return out

    def _admit(self, span: str) -> None:
        if self._closed:
            return
        words = _EDGE_PUNCT.sub("", span).split()
        new_phrases = {_fold(" ".join(words))} if len(words) > 1 else set()
        new_words = (
            {
                w.casefold()
                for w in words
                if _name_word(w) and w.casefold() not in self._common
            }
            if len(words) <= 3
            else set()
        )
        new_phrases -= self._phrases
        new_words -= self._words
        if not new_phrases and not new_words:
            return
        if (
            len(self._phrases) + len(self._words) + len(new_phrases | new_words)
            > self._max
        ):
            self.refused += 1
            return
        self._phrases |= new_phrases
        self._words |= new_words
        self._dirty = True

    def _compiled(self) -> re.Pattern[str] | None:
        if self._dirty:
            folded = self._phrases | {_fold(w) for w in self._words}
            parts = sorted(folded, key=len, reverse=True)
            alts = [r"\s+".join(re.escape(w) for w in p.split()) for p in parts]
            self._pattern = (
                re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)")
                if alts
                else None
            )
            self._word_set = frozenset(self._words)
            self._dirty = False
        return self._pattern
