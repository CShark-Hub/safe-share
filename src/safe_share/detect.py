"""Rule-based sensitive-text detection over OCR line strings.

Pure text in, character spans out: no OCR or screen dependencies, so it is
testable on plain strings and a model detector (GLiNER, later) can implement the
same `detect` interface and have its spans merged with these. Overlap merging is
the caller's job (union, never intersection).

Structured entities are recognized by format; this is not password detection.
Covered:
  - emails
  - provider key shapes: AWS AK/AS/AG/AI/AR..IA (incl. ASIA temp creds),
    GitHub gh[pousr]_, Slack xox[baprs]-, Google AIza, Stripe [sprk]k_live_
  - PEM private-key blocks (BEGIN..END, spanning lines), three-segment JWTs
  - a secret value following a keyword (password/secret/token/api key = ...),
    including the value-on-the-next-line case after an OCR line split
Not covered: free-form passwords with no marker, Stripe test keys, values that
look like ordinary words. Misses here are misses, not "no sensitive text".
Cross-line PEM coverage depends on OCR reading the words "BEGIN ... PRIVATE KEY"
(dashes optional); if OCR garbles those words, the base64 body leaks.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Detection:
    """A sensitive span [start, end) within `line_index` of the input list."""

    line_index: int
    start: int
    end: int
    entity_type: str
    source: str = "rule"
    score: float | None = None


class Detector(Protocol):
    def detect(self, texts: list[str]) -> list[Detection]: ...


class UnionDetector:
    """Runs every detector; a span flagged by any of them is kept."""

    def __init__(self, *detectors: Detector) -> None:
        self._detectors = detectors

    def detect(self, texts: list[str]) -> list[Detection]:
        return [d for det in self._detectors for d in det.detect(texts)]


@dataclass
class RuleConfig:
    emails: bool = True
    credentials: bool = True
    # repr=False: the terms are the user's private words and must stay out of
    # logs, reprs and tracebacks.
    terms: list[str] = field(default_factory=list, repr=False)


_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Single-line, self-contained key shapes; the whole match is the sensitive span.
_KEY_SHAPES: list[tuple[str, re.Pattern[str]]] = [
    (
        "aws_key",
        re.compile(
            r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA)[0-9A-Z]{16}(?![0-9A-Za-z])"
        ),
    ),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}(?![0-9A-Za-z])")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}(?![0-9A-Za-z-])")),
    # Trailing (?!class) not \b: a \b after a class containing -/_ fails when the
    # key ends in one of those, dropping the whole match and leaking the key.
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])")),
    ("stripe_key", re.compile(r"\b(?:sk|pk|rk)_live_[0-9A-Za-z]{16,}(?![0-9A-Za-z])")),
    (
        "jwt",
        re.compile(
            r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?![0-9A-Za-z_-])"
        ),
    ),
]

_KEYED = re.compile(
    r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|apikey|token)"
    r"\s*[:=][ \t]*(?P<v>\S.*)?$"
)
_LEADING_TOKEN = re.compile(r"\S+")
# Key on the words, not the dashes: OCR renders "-----" unreliably (dropped,
# merged, or read as a rule), so requiring literal dashes leaks the block.
_PEM_BEGIN = re.compile(r"(?i)BEGIN[A-Z0-9 –—-]*PRIVATE KEY")
_PEM_END = re.compile(r"(?i)END[A-Z0-9 –—-]*PRIVATE KEY")

# Values after a keyword that are not secrets; masking these is only annoying,
# but skipping them keeps demo precision up. Placeholder runs (****) also skip.
_NON_SECRET = frozenset(
    {
        "true",
        "false",
        "null",
        "none",
        "nil",
        "yes",
        "no",
        "enabled",
        "disabled",
        "on",
        "off",
        "required",
        "optional",
        "empty",
    }
)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _is_placeholder(value: str) -> bool:
    v = value.strip().casefold()
    return v in _NON_SECRET or bool(v) and set(v) <= set("*•●·.x")


def _compile_terms(terms: list[str]) -> re.Pattern[str] | None:
    """Whole-word alternation, longest term first, interior whitespace flexible."""
    parts = []
    for term in sorted((t.strip() for t in terms if t.strip()), key=len, reverse=True):
        words = re.split(r"\s+", _normalize(term))
        parts.append(r"\s+".join(re.escape(w) for w in words))
    if not parts:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)", re.IGNORECASE)


class RuleDetector:
    """Deterministic detector for emails, configured terms and credential shapes."""

    def __init__(self, config: RuleConfig) -> None:
        self._config = config
        self._terms = _compile_terms(config.terms)

    def detect(self, texts: list[str]) -> list[Detection]:
        lines = [_normalize(t) for t in texts]
        out: list[Detection] = []
        for i, text in enumerate(lines):
            if self._config.emails:
                out += [
                    Detection(i, m.start(), m.end(), "email")
                    for m in _EMAIL.finditer(text)
                ]
            if self._terms is not None:
                out += [
                    Detection(i, m.start(), m.end(), "term")
                    for m in self._terms.finditer(text)
                ]
            if self._config.credentials:
                self._detect_credentials(i, text, lines, out)
        return out

    def _detect_credentials(
        self, i: int, text: str, lines: list[str], out: list[Detection]
    ) -> None:
        for etype, pat in _KEY_SHAPES:
            out += [Detection(i, m.start(), m.end(), etype) for m in pat.finditer(text)]

        if _PEM_BEGIN.search(text):
            out.append(Detection(i, 0, len(text), "pem_key"))
        elif _PEM_END.search(text) or self._inside_pem(i, lines):
            # Body/END line of a block whose BEGIN was on an earlier line.
            out.append(Detection(i, 0, len(text), "pem_key"))

        m = _KEYED.search(text)
        if m:
            value = m.group("v")
            if value and not _is_placeholder(value):
                out.append(Detection(i, m.start("v"), m.end("v"), "keyed_secret"))
            elif not value:
                # Value split onto the next line by OCR: cover the whole
                # remainder, not just the first token (a secret may contain
                # spaces, and a partial cover is a leak).
                nxt = self._next_value_span(i, lines)
                if nxt is not None:
                    j, s, e = nxt
                    out.append(Detection(j, s, e, "keyed_secret"))

    @staticmethod
    def _inside_pem(i: int, lines: list[str]) -> bool:
        for k in range(i - 1, -1, -1):
            if _PEM_END.search(lines[k]):
                return False
            if _PEM_BEGIN.search(lines[k]):
                return True
        return False

    @staticmethod
    def _next_value_span(i: int, lines: list[str]) -> tuple[int, int, int] | None:
        if i + 1 >= len(lines):
            return None
        m = _LEADING_TOKEN.search(lines[i + 1])
        return (i + 1, m.start(), len(lines[i + 1])) if m else None
