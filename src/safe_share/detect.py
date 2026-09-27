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
  - a secret value after a keyword (password/secret/token/api key = ...), incl.
    DB_PASSWORD=, quoted keys and a value on the next line; Authorization/Bearer
    values; passwords in scheme://user:pass@host
Credential rules match a copy of the line with case and OCR confusions folded.
Not covered: free-form passwords with no marker, Stripe test keys, values that
look like ordinary words. Misses here are misses, not "no sensitive text".
Cross-line PEM coverage depends on OCR reading the words "BEGIN ... PRIVATE KEY"
(dashes optional); if OCR garbles those words, the base64 body leaks.
Not covered either: camelCase keys (dbPassword =).
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
    """Runs every detector; drops a model span inside a rule span on the same line."""

    def __init__(self, *detectors: Detector) -> None:
        self._detectors = detectors

    def detect(self, texts: list[str]) -> list[Detection]:
        found = [d for det in self._detectors for d in det.detect(texts)]
        rules = [d for d in found if d.source == "rule"]
        return [
            d
            for d in found
            if d.source == "rule"
            or not any(
                r.line_index == d.line_index and r.start <= d.start and d.end <= r.end
                for r in rules
            )
        ]


@dataclass
class RuleConfig:
    emails: bool = True
    credentials: bool = True
    # repr=False: the terms are the user's private words and must stay out of
    # logs, reprs and tracebacks.
    terms: list[str] = field(default_factory=list, repr=False)


_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_FOLD_MAP = str.maketrans({"0": "o", "1": "l", "i": "l", "5": "s", "v": "y"})


def _fold(text: str) -> str:
    """Lowercase and fold OCR confusions, one char per char so spans map back."""
    low = "".join(c.lower() if len(c.lower()) == 1 else c for c in text)
    return low.translate(_FOLD_MAP)


# Matched on folded text; the whole match is the span. Fixed lengths allow +-1
# for a dropped or doubled OCR character.
_AWS_PREFIXES = ("AKIA", "ASIA", "AGPA", "AIDA", "AROA", "AIPA", "ANPA", "ANVA")
_KEY_SHAPES: list[tuple[str, re.Pattern[str]]] = [
    (
        "aws_key",
        re.compile(
            rf"\b(?:{'|'.join(map(_fold, _AWS_PREFIXES))})[0-9a-z]{{15,17}}(?![0-9a-z])"
        ),
    ),
    (
        "github_token",
        re.compile(rf"\b{_fold('gh')}[pousr]_[0-9a-z]{{20,}}(?![0-9a-z])"),
    ),
    (
        "slack_token",
        re.compile(rf"\b{_fold('xox')}[baprs]-[0-9a-z-]{{10,}}(?![0-9a-z-])"),
    ),
    # Trailing (?!class) not \b: a \b after a class containing -/_ fails when the
    # key ends in one of those, dropping the whole match and leaking the key.
    ("google_key", re.compile(rf"\b{_fold('AIza')}[0-9a-z_-]{{34,36}}(?![0-9a-z_-])")),
    (
        "stripe_key",
        re.compile(rf"\b(?:sk|pk|rk)_{_fold('live')}_[0-9a-z]{{15,}}(?![0-9a-z])"),
    ),
    (
        "jwt",
        re.compile(
            rf"\b{_fold('eyJ')}[0-9a-z_-]+\.[0-9a-z_-]+\.[0-9a-z_-]+(?![0-9a-z_-])"
        ),
    ),
]
CREDENTIAL_TYPES = frozenset(
    [name for name, _ in _KEY_SHAPES] + ["pem_key", "keyed_secret", "conn_password"]
)

_KEYWORDS = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "api[_-]?key",
    "apikey",
    "access[_-]?key",
    "private[_-]?key",
    "credentials?",
    "authorization",
)
# Keyword as a whole identifier segment, not a substring (tokenizer, max_tokens).
_KEYED = re.compile(
    r"(?<![0-9a-z])(?:[0-9a-z]+[_-])*"
    rf"(?:{'|'.join(_fold(k) for k in _KEYWORDS)})"
    r"(?:[_-][0-9a-z_-]*+)?[\"']?\s*[:=][ \t]*(?P<v>\S.*)?$"
)
_BEARER = re.compile(rf"\b{_fold('bearer')}\s+(?P<v>[0-9a-z._~+/=-]{{8,}})")
_CONN_PASSWORD = re.compile(r"://[^/\s:@]*:(?P<v>[^@\s/]+)@")
_LEADING_TOKEN = re.compile(r"\S+")
# YAML block scalars and line continuations: the value is on the next line.
_NEXT_LINE_MARKERS = frozenset({"|", "|-", ">", ">-", "\\"})
# Key on the words, not the dashes: OCR renders "-----" unreliably (dropped,
# merged, or read as a rule), so requiring literal dashes leaks the block.
_PEM_BEGIN = re.compile(rf"{_fold('BEGIN')}[a-z0-9 –—-]*{_fold('PRIVATE KEY')}")
_PEM_END = re.compile(rf"{_fold('END')}[a-z0-9 –—-]*{_fold('PRIVATE KEY')}")

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
    v = value.strip().strip("\"',").strip().casefold()
    return not v or v in _NON_SECRET or set(v) <= set("*•●·.x")


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
        folded = [_fold(t) for t in lines]
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
                self._detect_credentials(i, text, folded, out)
        return out

    def _detect_credentials(
        self, i: int, text: str, folded: list[str], out: list[Detection]
    ) -> None:
        # Patterns run on the folded line; spans index the raw line 1:1.
        f = folded[i]
        for etype, pat in _KEY_SHAPES:
            out += [Detection(i, m.start(), m.end(), etype) for m in pat.finditer(f)]

        if _PEM_BEGIN.search(f):
            out.append(Detection(i, 0, len(text), "pem_key"))
        elif _PEM_END.search(f) or self._inside_pem(i, folded):
            # Body/END line of a block whose BEGIN was on an earlier line.
            out.append(Detection(i, 0, len(text), "pem_key"))

        for pat in (_BEARER, _CONN_PASSWORD):
            m = pat.search(f)
            if m:
                etype = "conn_password" if pat is _CONN_PASSWORD else "keyed_secret"
                out.append(Detection(i, m.start("v"), m.end("v"), etype))

        m = _KEYED.search(f)
        if m:
            value = text[m.start("v") : m.end("v")] if m.group("v") else None
            if value is not None and value.strip() in _NEXT_LINE_MARKERS:
                value = None
            if value and not _is_placeholder(value):
                out.append(Detection(i, m.start("v"), m.end("v"), "keyed_secret"))
            elif not value:
                # Value split onto the next line by OCR: cover the whole
                # remainder, not just the first token (a secret may contain
                # spaces, and a partial cover is a leak).
                nxt = self._next_value_span(i, folded)
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
