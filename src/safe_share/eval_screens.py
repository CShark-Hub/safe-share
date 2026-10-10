"""Image-level eval: OCR -> detection -> redaction on screenshots, then OCR again.

`scene_<n>_*.png` is checked against case n of `screens.json` plus the local
`screens_credentials.json`. A truth string readable after masking is a leak, by
stage: detection (read, not flagged), mask (flagged, still readable), ocr (not
read in pass 1). Vision-readable only: a lower bound on what a person can read.
Misread truth is "garbled", never-read truth "unread". "Collateral" counts
unflagged harmless lines at least half under a pill. Masked harmless lines print
their OCR text only with --show-text. Nothing is written to disk unless
--save DIR is given: masked screenshots as <name>_masked.png, for visual review.
They still show any leak the report lists; treat them like the input.
--jitter re-runs each image shifted by 1-3 px and reports, as counts only, how
often flagged lines are lost and how much pill grouping and label size change.
--context shifts the OCR lines on one side of a random vertical cut, as if a pane
scrolled, and counts truth occurrences flagged in all, some or none of the runs.
--name-cache image|session wraps the detector in the session name cache, fresh per
image or one for the whole run; --context always uses a fresh cache per image.
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
from .name_cache import NameCacheDetector
from .ocr import TextRecognizer
from .redact import GRAPHITE, fit_label, label_for, label_px, plan, redact
from .types import OcrLine

_ROOT = Path(__file__).resolve().parents[2]
_SCREENS = _ROOT / "fixtures" / "screens"
_TRUTH = _ROOT / "fixtures" / "detect" / "screens.json"
_LOCAL_TRUTH = _ROOT / "fixtures" / "detect" / "screens_credentials.json"
_NAME = re.compile(r"scene_(\d+)_(\w+)(?<!_masked)\.png")
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
    collateral: int = 0
    ocr_ms: float = 0.0
    det_ms: float = 0.0
    mask_ms: float = 0.0
    pills: int = 0
    wrong_label: int = 0  # label type matches no truth type of its lines
    no_label: int = 0  # label omitted: did not fit


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
        return [(s, t) for s, t, _, _ in self.spans(text)]

    def spans(self, text: str) -> list[tuple[str, str, int, int]]:
        """(truth string, type, start, end) for each exact occurrence in `text`."""
        if self._pat is None:
            return []
        out = []
        for m in self._pat.finditer(text):
            assert m.lastgroup is not None  # every alternative is a named group
            i = int(m.lastgroup[1:])
            out.append((self.strings[i], self.types[i], m.start(), m.end()))
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
    save: Path | None = None,
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
    harmless: list[int] = []
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
        harmless.append(i)
        if i in flagged:
            r.harmless_masked.append((i, ln.box, ln.text))
    r.unread = sorted({_text(case, e) for e in case["expect"]} - read)

    h, w = frame.shape[:2]
    pills = plan(raw, detections, w, h)
    for i in harmless:
        box = raw[i].box
        area = _overlap(box, box)
        if i in flagged or not area:
            continue
        if max((_overlap(box, p.rect) for p in pills), default=0) * 2 >= area:
            r.collateral += 1
    px = label_px(raw) if raw else 0
    for p in pills:
        r.pills += 1
        r.no_label += fit_label(p, px, *GRAPHITE) is None
        want = {label_for(t) for i in p.lines for _, t in truth.find(raw[i].text)}
        r.wrong_label += bool(want) and p.label not in want

    t3 = time.perf_counter()
    masked = redact(frame.copy(), raw, detections)
    r.mask_ms = (time.perf_counter() - t3) * 1000
    if save is not None:
        # _NAME excludes the suffix, so the output is never read as a fixture.
        Image.fromarray(masked[..., [2, 1, 0]]).save(save / f"{path.stem}_masked.png")
    for ln in ocr.recognize(masked):
        for s, t in truth.find(ln.text):
            r.leaks.append((s, t, _stage(ln.box, s, raw, flagged)))
    return r


_SHIFTS = ((1, 0), (0, 1), (1, 1), (2, 3), (3, 2))


@dataclass
class Jitter:
    runs: int = 0
    affected: int = 0  # runs that lost at least one flagged line
    flagged: int = 0  # flagged lines in the unshifted frames, summed over runs
    lost_unflagged: int = 0  # line still read, no longer flagged
    lost_split: int = 0  # no OCR line in the same place
    comparable: int = 0  # runs with the same flagged lines
    regrouped: int = 0
    label_px: int = 0  # comparable runs whose frame label size changed
    edges: list[int] = field(default_factory=list)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    inter = _overlap(a, b)
    union = _overlap(a, a) + _overlap(b, b) - inter
    return inter / union if union else 0.0


def jitter(
    path: Path, ocr: TextRecognizer, detector: Detector, scale: float, j: Jitter
) -> None:
    """Compare detection and pill geometry of `path` against 1-3 px shifted copies."""
    frame = _load(path, scale)
    h, w = frame.shape[:2]
    base = ocr.recognize(frame)
    base_det = detector.detect([ln.text for ln in base])
    base_flag = [base[i].box for i in sorted({d.line_index for d in base_det})]
    base_pills = plan(base, base_det, w, h)
    for dx, dy in _SHIFTS:
        pad = np.pad(frame, ((dy, 0), (dx, 0), (0, 0)), mode="edge")[:h, :w]
        lines = [
            OcrLine(ln.text, (x0 - dx, y0 - dy, x1 - dx, y1 - dy), ln.confidence)
            for ln in ocr.recognize(np.ascontiguousarray(pad))
            for x0, y0, x1, y1 in [ln.box]
        ]
        det = detector.detect([ln.text for ln in lines])
        flag = [lines[i].box for i in sorted({d.line_index for d in det})]
        lost = [
            b for b in base_flag if max((_iou(b, f) for f in flag), default=0) <= 0.5
        ]
        j.runs += 1
        j.flagged += len(base_flag)
        j.affected += bool(lost)
        for b in lost:
            read = any(_iou(b, ln.box) > 0.5 for ln in lines)
            j.lost_unflagged += read
            j.lost_split += not read
        if lost or len(flag) != len(base_flag):
            continue
        j.comparable += 1
        pills = plan(lines, det, w, h)
        shape = sorted(len(p.covers) for p in pills)
        j.regrouped += shape != sorted(len(p.covers) for p in base_pills)
        j.label_px += label_px(lines) != label_px(base)
        for p in pills:
            q = max(base_pills, key=lambda q: _iou(p.rect, q.rect))
            j.edges.append(max(abs(a - b) for a, b in zip(p.rect, q.rect, strict=True)))


_CONTEXT_RUNS = 10
_CONTEXT_MAX_SHIFT = 6  # line heights


@dataclass
class Context:
    runs: int = 0
    missed: int = 0  # detector runs that missed a truth occurrence, summed
    harmless_flagged: int = 0  # summed over runs
    # type -> [stable, flaky, never]; flaky entries are (scene, truth, hits)
    counts: dict[str, list[int]] = field(default_factory=dict)
    flaky: list[tuple[str, str, int]] = field(default_factory=list)
    never: list[tuple[str, str]] = field(default_factory=list)


def _scrolled(lines: list[OcrLine], w: int, rng: np.random.Generator) -> list[OcrLine]:
    """`lines` with those on one side of a random vertical cut moved vertically."""
    lh = float(np.median([ln.box[3] - ln.box[1] for ln in lines]))
    cut = rng.uniform(0.2, 0.8) * w
    right = bool(rng.random() < 0.5)
    dy = round(rng.choice((-1, 1)) * rng.integers(1, _CONTEXT_MAX_SHIFT + 1) * lh)
    out = []
    for ln in lines:
        x0, y0, x1, y1 = ln.box
        if ((x0 + x1) / 2 >= cut) == right:
            ln = OcrLine(ln.text, (x0, y0 + dy, x1, y1 + dy), ln.confidence)
        out.append(ln)
    return out


def context(
    path: Path,
    case: dict,
    terms: list[str],
    ocr: TextRecognizer,
    detector: Detector,
    scale: float,
    seed: int,
    c: Context,
) -> None:
    """Stability of each truth occurrence under reordered lines; a hit must overlap it.
    Variants run in order (unscrolled first), so a name cache carries between them."""
    truth = Truth(case, terms)
    frame = _load(path, scale)
    w = frame.shape[1]
    base = ocr.recognize(frame)
    if not base:
        return
    keep = [_norm(k) for k in case.get("keep", [])]
    harmless = [
        i
        for i, ln in enumerate(base)
        if not truth.find(ln.text)
        and (any(k in _norm(ln.text) for k in keep) or truth.fuzzy(ln.text) is None)
    ]
    rng = np.random.default_rng(seed)
    variants = [base] + [_scrolled(base, w, rng) for _ in range(_CONTEXT_RUNS)]
    hits: list[set[tuple[int, int, int]]] = []  # (base line, start, end)
    for lines in variants:
        order = sorted(
            range(len(lines)), key=lambda i: (lines[i].box[1], lines[i].box[0])
        )
        det = detector.detect([lines[i].text for i in order])
        hits.append({(order[d.line_index], d.start, d.end) for d in det})
        flagged = {j for j, _, _ in hits[-1]}
        c.harmless_flagged += sum(i in flagged for i in harmless)
    c.runs += len(variants)
    for i, ln in enumerate(base):
        for s, t, a, b in truth.spans(ln.text):
            n = sum(any(j == i and lo < b and a < hi for j, lo, hi in h) for h in hits)
            c.missed += len(variants) - n
            row = c.counts.setdefault(t, [0, 0, 0])
            if n == len(variants):
                row[0] += 1
            elif n:
                row[1] += 1
                c.flaky.append((path.stem, s, n))
            else:
                row[2] += 1
                c.never.append((path.stem, s))


def print_context(c: Context) -> None:
    # Truth strings are fixture text, not OCR output.
    total = [sum(col) for col in zip(*c.counts.values())] or [0, 0, 0]
    print(
        f"\ncontext ({c.runs} detector runs, {_CONTEXT_RUNS} scrolled variants per "
        f"image): truth occurrences stable {total[0]}, flaky {total[1]}, never {total[2]}"
    )
    per_run = c.harmless_flagged / c.runs if c.runs else 0.0
    print(
        f"  missed occurrence-runs {c.missed}; "
        f"harmless lines flagged {per_run:.1f} per run"
    )
    for t, (st, fl, nv) in sorted(c.counts.items()):
        print(f"  {t:<14} stable {st:>3}  flaky {fl:>3}  never {nv:>3}")
    runs = _CONTEXT_RUNS + 1
    for name, s, n in c.flaky:
        print(f"  - flaky {name}: {s!r} {n}/{runs}")
    for name, s in c.never:
        print(f"  - never {name}: {s!r}")


def print_jitter(j: Jitter) -> None:
    lost = j.lost_unflagged + j.lost_split
    print(
        f"\njitter ({j.runs} runs, 1-3 px shifts): flagged lines lost "
        f"{_pct(lost, j.flagged)} in {j.affected} runs "
        f"({j.lost_unflagged} no longer flagged, {j.lost_split} OCR split differently)"
    )
    if j.comparable:
        e = np.array(j.edges) if j.edges else np.zeros(1)
        print(
            f"  same flagged lines in {j.comparable} runs: grouping changed "
            f"{j.regrouped}, label size changed {j.label_px}, pill edge moved "
            f"p95 {np.percentile(e, 95):.0f} / max {e.max():.0f} px"
        )


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
    n = sum(r.collateral for r in results)
    print(
        f"collateral (unflagged harmless lines under a pill): "
        f"{_pct(n, sum(r.harmless for r in results))}"
    )
    pills = sum(r.pills for r in results)
    print(
        f"labels: wrong type {_pct(sum(r.wrong_label for r in results), pills)}, "
        f"omitted {_pct(sum(r.no_label for r in results), pills)}"
    )
    print(
        f"\nOCR {_ms([r.ocr_ms for r in results])}; "
        f"detection {_ms([r.det_ms for r in results])}; "
        f"redact {_ms([r.mask_ms for r in results])} "
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
    ap.add_argument(
        "--jitter",
        action="store_true",
        help="also re-run each image shifted by 1-3 px (stability counts)",
    )
    ap.add_argument("--name-cache", choices=["off", "image", "session"], default="off")
    ap.add_argument("--seed", type=int, default=0, help="--context seed offset")
    ap.add_argument(
        "--context",
        action="store_true",
        help="also re-run detection with one pane scrolled (context stability)",
    )
    ap.add_argument(
        "--save",
        type=Path,
        help="write masked screenshots here; they may still show leaked text",
    )
    args = ap.parse_args(argv)
    if not 0 < args.scale <= 1:
        ap.error("--scale must be in (0, 1]")
    if args.save is not None:
        if not args.save.is_dir():
            ap.error(f"--save: {args.save} is not a directory")
        if args.save.resolve() == args.screens.resolve():
            ap.error("--save must not be the screenshots directory")

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
    base = _build_detector(args.detector, terms)
    session = NameCacheDetector(base) if args.name_cache == "session" else None

    def fresh() -> Detector:
        return base if args.name_cache == "off" else NameCacheDetector(base)

    ocr = TextRecognizer()

    results: list[SceneResult] = []
    for n, _, p in files:
        if not 1 <= n <= len(cases):
            print(f"skip {p.name}: no case {n}", file=sys.stderr)
            continue
        results.append(
            evaluate(
                p,
                cases[n - 1],
                terms,
                ocr,
                session if session is not None else fresh(),
                args.scale,
                args.save,
            )
        )
    if not results:
        return 2
    print_report(results, args.show_text)
    if args.jitter:
        j = Jitter()
        for n, _, p in files:
            if 1 <= n <= len(cases):
                jitter(
                    p, ocr, session if session is not None else fresh(), args.scale, j
                )
        print_jitter(j)
    if args.context:
        c = Context()
        for k, (n, _, p) in enumerate(files):
            if 1 <= n <= len(cases):
                context(
                    p, cases[n - 1], terms, ocr, fresh(), args.scale, args.seed + k, c
                )
        print_context(c)
    # Garbled truth left unmasked also fails.
    visible = any(r.leaks or any(not m for _, m in r.garbled) for r in results)
    return 1 if visible else 0


if __name__ == "__main__":
    raise SystemExit(main())
