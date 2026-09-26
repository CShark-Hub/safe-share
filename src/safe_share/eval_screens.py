"""Image-level eval: OCR -> detection -> redaction on screenshots, then OCR again.

`scene_<n>_*.png` is checked against case n of `screens.json` plus the local
`screens_credentials.json`. A truth string readable after masking is a leak, by
stage: detection (read, not flagged), mask (flagged, still readable), ocr (not
read in pass 1). Vision-readable only: a lower bound on what a person can read.
Misread truth is "garbled", never-read truth "unread". Masked harmless lines
print their OCR text only with --show-text. Nothing is written to disk.
Run: python -m safe_share.eval_screens [dir] [--detector rules|union] [--scale 0.5]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image

from .detect import Detector, RuleConfig, RuleDetector, UnionDetector
from .ocr import TextRecognizer
from .redact import redact
from .types import OcrLine

_ROOT = Path(__file__).resolve().parents[2]
_SCREENS = _ROOT / "fixtures" / "screens"
_TRUTH = _ROOT / "fixtures" / "detect" / "screens.json"
_LOCAL_TRUTH = _ROOT / "fixtures" / "detect" / "screens_credentials.json"
_NAME = re.compile(r"scene_(\d+)_(\w+)\.png")
_FUZZY = 0.75
_FUZZY_MIN_LEN = 5  # shorter names fuzzy-match ordinary words
_WORD = re.compile(r"[^\W_]+")


@dataclass
class SceneResult:
    name: str
    tags: list[str]
    seen: Counter = field(default_factory=Counter)  # type -> occurrences in pass 1
    covered: Counter = field(default_factory=Counter)  # type -> on a flagged line
    leaks: list[tuple[str, str, str]] = field(default_factory=list)  # text, type, stage
    garbled: list[tuple[str, bool]] = field(default_factory=list)  # truth, masked
    unread: list[str] = field(default_factory=list)
    harmless: int = 0
    harmless_masked: list[tuple[int, tuple[int, int, int, int], str]] = field(
        default_factory=list
    )
    ocr_ms: float = 0.0
    det_ms: float = 0.0


class Truth:
    """Expected strings of one case. Persons: case-sensitive whole word. Terms:
    case-insensitive; boundaries are ASCII alphanumerics, so `_` separates."""

    def __init__(self, case: dict, terms: list[str]) -> None:
        types = {t: "term" for t in terms}
        types.update({_text(case, e): e["type"] for e in case["expect"]})
        self.strings = sorted(types, key=len, reverse=True)
        self.types = [types[s] for s in self.strings]
        self._norms = [_norm(s) for s in self.strings]
        alts = []
        for i, s in enumerate(self.strings):
            body = r"\s+".join(map(re.escape, s.split()))
            if self.types[i] == "term":
                alts.append(rf"(?P<t{i}>(?<![A-Za-z0-9])(?i:{body})(?![A-Za-z0-9]))")
            else:
                alts.append(rf"(?P<t{i}>(?<!\w){body}(?!\w))")
        self._pat = re.compile("|".join(alts)) if alts else None

    def find(self, text: str) -> list[tuple[str, str]]:
        """(truth string, type) for each exact occurrence in `text`."""
        if self._pat is None:
            return []
        out = []
        for m in self._pat.finditer(text):
            assert m.lastgroup is not None  # every alternative is a named group
            i = int(m.lastgroup[1:])
            out.append((self.strings[i], self.types[i]))
        return out

    def fuzzy(self, text: str) -> str | None:
        """The closest truth string that `text` contains in misread form, if any."""
        words = _norm(text).split()
        best, best_ratio = None, _FUZZY
        for s, target in zip(self.strings, self._norms):
            if len(s) < _FUZZY_MIN_LEN:
                continue
            k = len(target.split())
            for i in range(max(1, len(words) - k + 1)):
                window = " ".join(words[i : i + k])
                ratio = SequenceMatcher(None, window, target).ratio()
                if ratio >= best_ratio:
                    best, best_ratio = s, ratio
        return best


def _text(case: dict, e: dict) -> str:
    return e["text"] if "text" in e else case["lines"][e["line"]]


def _load(path: Path, scale: float) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    if scale != 1.0:
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
        )
    return np.ascontiguousarray(np.asarray(img)[..., [2, 1, 0, 3]])


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def _stage(
    box: tuple[int, int, int, int], text: str, raw: list[OcrLine], flagged: set[int]
) -> str:
    """Approximate: pass 2 may split or merge lines differently from pass 1."""
    best = max(range(len(raw)), key=lambda i: _overlap(box, raw[i].box), default=None)
    if best is None or _overlap(box, raw[best].box) == 0:
        return "ocr"
    if best in flagged:
        return "mask"
    hit = f" {_norm(text)} " in f" {_norm(raw[best].text)} "
    return "detection" if hit else "ocr"


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.casefold()))


def evaluate(
    path: Path,
    case: dict,
    terms: list[str],
    ocr: TextRecognizer,
    detector: Detector,
    scale: float = 1.0,
) -> SceneResult:
    truth = Truth(case, terms)
    r = SceneResult(path.stem, case.get("tags", []))
    frame = _load(path, scale)

    t0 = time.perf_counter()
    raw = ocr.recognize(frame)
    t1 = time.perf_counter()
    detections = detector.detect([ln.text for ln in raw])
    flagged = {d.line_index for d in detections}
    t2 = time.perf_counter()
    r.ocr_ms, r.det_ms = (t1 - t0) * 1000, (t2 - t1) * 1000

    keep = [_norm(k) for k in case.get("keep", [])]
    read: set[str] = set()
    for i, ln in enumerate(raw):
        hits = truth.find(ln.text)
        for s, t in hits:
            read.add(s)
            r.seen[t] += 1
            r.covered[t] += i in flagged
        if hits:
            continue
        is_keep = any(k in _norm(ln.text) for k in keep)
        near = None if is_keep else truth.fuzzy(ln.text)
        if near is not None:
            read.add(near)
            r.garbled.append((near, i in flagged))
            continue
        r.harmless += 1
        if i in flagged:
            r.harmless_masked.append((i, ln.box, ln.text))
    r.unread = sorted({_text(case, e) for e in case["expect"]} - read)

    masked = redact(frame.copy(), raw, detections)
    for ln in ocr.recognize(masked):
        for s, t in truth.find(ln.text):
            r.leaks.append((s, t, _stage(ln.box, s, raw, flagged)))
    return r


def _ms(xs: list[float]) -> str:
    return f"p50 {statistics.median(xs):.0f} / max {max(xs):.0f} ms"


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else "-"


def print_report(results: list[SceneResult], show_text: bool = False) -> None:
    print(
        f"{'screen':<16}{'seen':>6}{'covered':>9}{'leaks':>7}"
        f"{'garbled':>9}{'harmless masked':>18}"
    )
    for r in results:
        seen, cov = sum(r.seen.values()), sum(r.covered.values())
        g = f"{sum(m for _, m in r.garbled)}/{len(r.garbled)}"
        print(
            f"{r.name:<16}{seen:>6}{cov:>9}{len(r.leaks):>7}{g:>9}"
            f"{f'{len(r.harmless_masked)}/{r.harmless}':>18}"
        )

    seen, cov = Counter(), Counter()
    for r in results:
        seen.update(r.seen)
        cov.update(r.covered)
    print(f"\n{'type':<16}{'covered / seen in OCR pass 1':>30}")
    for t in sorted(seen):
        print(f"{t:<16}{_pct(cov[t], seen[t]):>30}")

    # Leak, garbled and unread strings are fixture truth, not OCR output.
    leaks = Counter((r.name, *leak) for r in results for leak in r.leaks)
    stages = Counter()
    for (*_, stage), n in leaks.items():
        stages[stage] += n
    print(
        f"\nleaks: {sum(leaks.values())} "
        f"({', '.join(f'{k} {v}' for k, v in sorted(stages.items()))})"
    )
    for (name, text, etype, stage), n in leaks.items():
        print(
            f"  - {name}: {text!r} ({etype}, {stage} miss){f' x{n}' if n > 1 else ''}"
        )

    garbled = [(r.name, s, m) for r in results for s, m in r.garbled]
    print(
        f"garbled by OCR (misread truth; masked/total): "
        f"{sum(m for *_, m in garbled)}/{len(garbled)}"
    )
    for name, s, m in garbled:
        print(f"  - {name}: {s!r} {'masked' if m else 'NOT masked'}")
    print("unread (distinct truth strings not read in pass 1):")
    for r in results:
        if r.unread:
            print(f"  - {r.name}: {', '.join(map(repr, r.unread))}")

    for tag in sorted({t for r in results for t in r.tags}):
        sub = [r for r in results if tag in r.tags]
        n = sum(len(r.harmless_masked) for r in sub)
        print(f"harmless lines masked [{tag}]: {_pct(n, sum(r.harmless for r in sub))}")
    for r in results:
        for i, box, t in r.harmless_masked:
            print(
                f"  - {r.name}: {t[:60]!r}"
                if show_text
                else f"  - {r.name}: line {i} {box}"
            )
    print(
        f"\nOCR {_ms([r.ocr_ms for r in results])}; "
        f"detection {_ms([r.det_ms for r in results])} "
        "(offline, per image, one pass; live latency comes from the app's timing)"
    )


def _build_detector(kind: str, terms: list[str]) -> Detector:
    rules = RuleDetector(RuleConfig(terms=terms))
    if kind == "rules":
        return rules
    from .gliner_detect import GlinerDetector, load_best

    model, where = load_best()
    print(f"[detect] GLiNER on {where}", file=sys.stderr)
    return UnionDetector(rules, GlinerDetector(model))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m safe_share.eval_screens")
    ap.add_argument("screens", nargs="?", type=Path, default=_SCREENS)
    ap.add_argument("--detector", choices=["rules", "union"], default="union")
    ap.add_argument(
        "--scale", type=float, default=1.0, help="downscale images (small-text check)"
    )
    ap.add_argument(
        "--show-text",
        action="store_true",
        help="print OCR text of masked harmless lines",
    )
    args = ap.parse_args(argv)
    if not 0 < args.scale <= 1:
        ap.error("--scale must be in (0, 1]")

    files = sorted(
        (int(m[1]), p.name, p)
        for p in args.screens.glob("scene_*.png")
        if (m := _NAME.fullmatch(p.name))
    )
    if not files:
        print(f"no scene_<n>_<variant>.png in {args.screens}", file=sys.stderr)
        return 2
    if not _TRUTH.exists():
        print(f"no ground truth at {_TRUTH}", file=sys.stderr)
        return 2
    data = json.loads(_TRUTH.read_text())
    cases, terms = data["cases"], data.get("terms", [])
    if _LOCAL_TRUTH.exists():
        local = json.loads(_LOCAL_TRUTH.read_text())
        cases = cases + local["cases"]
        terms = terms + local.get("terms", [])
    detector = _build_detector(args.detector, terms)
    ocr = TextRecognizer()

    results: list[SceneResult] = []
    for n, _, p in files:
        if not 1 <= n <= len(cases):
            print(f"skip {p.name}: no case {n}", file=sys.stderr)
            continue
        results.append(evaluate(p, cases[n - 1], terms, ocr, detector, args.scale))
    if not results:
        return 2
    print_report(results, args.show_text)
    # Garbled truth left unmasked also fails.
    visible = any(r.leaks or any(not m for _, m in r.garbled) for r in results)
    return 1 if visible else 0


if __name__ == "__main__":
    raise SystemExit(main())
