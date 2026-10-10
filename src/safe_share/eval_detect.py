"""String-level evaluation of a detector against repo fixtures.

Pure text, no screen. Reports per-type span recall/precision, false positives, an
exposure count (expected spans not fully covered by the union of detections),
and line-level numbers, since redaction covers a whole line on any hit.
`--perturb` adds OCR-style variants of every credential span and reports them apart.
Run: python -m safe_share.eval_detect [fixtures.json] [--detector gliner|union] [--perturb]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .detect import Detection, Detector, RuleConfig, RuleDetector, UnionDetector
from .gliner_detect import THRESHOLD

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "detect" / "cases.json"


@dataclass(frozen=True)
class Expected:
    line: int
    start: int
    end: int
    type: str


def _resolve_expected(exp: dict, lines: list[str]) -> Expected:
    line = exp["line"]
    if exp.get("whole_line"):
        return Expected(line, 0, len(lines[line]), exp["type"])
    text = exp["text"]
    start = lines[line].index(text)
    return Expected(line, start, start + len(text), exp["type"])


_NOT_CREDENTIAL = {"email", "term", "person"}


def _swap(s: str, pairs: list[tuple[str, str]]) -> str:
    """Replace the first character found from `pairs` (tried in order)."""
    for a, b in pairs:
        i = s.find(a)
        if i >= 0:
            return s[:i] + b + s[i + 1 :]
    return s


def _flip_case(s: str) -> str:
    for i in range(min(3, len(s)), len(s)):
        if s[i].isalpha():
            return s[:i] + s[i].swapcase() + s[i + 1 :]
    return s


_PERTURBATIONS = {
    "case": _flip_case,
    "y/v": lambda s: _swap(s, [("y", "v")]),
    "O/0": lambda s: _swap(s, [("O", "0"), ("0", "O")]),
    "l/1/I": lambda s: _swap(s, [("l", "1"), ("I", "l"), ("1", "l")]),
    "S/5": lambda s: _swap(s, [("S", "5"), ("5", "S")]),
    "drop": lambda s: s[: len(s) // 2] + s[len(s) // 2 + 1 :],
}


def perturb(data: dict) -> dict:
    """One case per (credential span, OCR confusion) that changes the span."""
    out = []
    for case in data["cases"]:
        for k, exp in enumerate(case["expect"]):
            if exp["type"] in _NOT_CREDENTIAL:
                continue
            line = case["lines"][exp["line"]]
            orig = line if exp.get("whole_line") else exp["text"]
            for kind, fn in _PERTURBATIONS.items():
                new = fn(orig)
                if new == orig:
                    continue
                c = json.loads(json.dumps(case))
                c["id"] = f"{case['id']}~{k}~{kind}"
                c["lines"][exp["line"]] = line.replace(orig, new, 1)
                # Score only the perturbed span; its siblings are the clean run's.
                c["expect"] = [c["expect"][k]]
                if not exp.get("whole_line"):
                    c["expect"][0]["text"] = new
                out.append(c)
    return {**data, "cases": out}


def _overlap(a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
    return a_s < b_e and b_s < a_e


def _fully_covered(e: Expected, dets: list[Detection], same_type: bool) -> bool:
    """True if [e.start, e.end) is entirely covered by the union of matching
    detections. Full containment, not mere overlap: a partial cover is a leak."""
    spans = sorted(
        (d.start, d.end)
        for d in dets
        if d.line_index == e.line and (not same_type or d.entity_type == e.type)
    )
    cursor = e.start
    for s, en in spans:
        if s > cursor:
            break
        cursor = max(cursor, en)
        if cursor >= e.end:
            return True
    return cursor >= e.end


@dataclass
class _LineTally:
    pos_covered: int = 0
    pos_total: int = 0
    neg_flagged: int = 0
    neg_total: int = 0

    def add(self, positive: bool, hit: bool) -> None:
        if positive:
            self.pos_total += 1
            self.pos_covered += hit
        else:
            self.neg_total += 1
            self.neg_flagged += hit


def _ratio(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else "-"


@dataclass
class Report:
    recall_hit: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recall_tot: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    prec_hit: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    prec_tot: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    exposures: list[str] = field(default_factory=list)
    false_positives: list[tuple[str, str]] = field(default_factory=list)
    lines_all: _LineTally = field(default_factory=_LineTally)
    lines_by_type: dict[str, _LineTally] = field(
        default_factory=lambda: defaultdict(_LineTally)
    )
    lines_by_tag: dict[str, _LineTally] = field(
        default_factory=lambda: defaultdict(_LineTally)
    )


def evaluate(data: dict, detector: Detector) -> Report:
    r = Report()
    for case in data["cases"]:
        lines = case["lines"]
        expected = [_resolve_expected(e, lines) for e in case["expect"]]
        found = detector.detect(lines)

        positive_lines = {e.line for e in expected}
        hit_lines = {d.line_index for d in found}
        for i in range(len(lines)):
            positive, hit = i in positive_lines, i in hit_lines
            r.lines_all.add(positive, hit)
            for tag in case.get("tags", []):
                r.lines_by_tag[tag].add(positive, hit)
        for etype, line in {(e.type, e.line) for e in expected}:
            r.lines_by_type[f"type:{etype}"].add(True, line in hit_lines)

        for e in expected:
            r.recall_tot[e.type] += 1
            if _fully_covered(e, found, same_type=True):
                r.recall_hit[e.type] += 1
            if not _fully_covered(e, found, same_type=False):
                r.exposures.append(f"{case['id']} (line {e.line}, {e.type})")

        all_negative = not expected
        for d in found:
            r.prec_tot[d.entity_type] += 1
            hit = any(
                d.line_index == e.line
                and _overlap(d.start, d.end, e.start, e.end)
                and d.entity_type == e.type
                for e in expected
            )
            if hit:
                r.prec_hit[d.entity_type] += 1
            elif not any(
                d.line_index == e.line and _overlap(d.start, d.end, e.start, e.end)
                for e in expected
            ):
                # Snippets only for all-negative cases, which are synthetic
                # non-secrets by construction (privacy rule 6).
                if all_negative:
                    snippet = lines[d.line_index][d.start : d.end][:32]
                    r.false_positives.append((case["id"], f"{d.entity_type}:{snippet}"))
                else:
                    r.false_positives.append(
                        (
                            case["id"],
                            f"{d.entity_type}@L{d.line_index}:{d.start}-{d.end}",
                        )
                    )
    return r


def print_report(r: Report) -> None:
    types = sorted(set(r.recall_tot) | set(r.prec_tot))
    print(f"{'type':<16}{'recall':>12}{'precision':>14}")
    for t in types:
        rt, ph, pt = r.recall_tot[t], r.prec_hit[t], r.prec_tot[t]
        rec = f"{r.recall_hit[t]}/{rt}" if rt else "-"
        prc = f"{ph}/{pt}" if pt else "-"
        print(f"{t:<16}{rec:>12}{prc:>14}")

    print(f"\n{'lines':<16}{'covered':>18}{'wrongly masked':>20}")
    rows = [
        ("all", r.lines_all),
        *sorted(r.lines_by_type.items()),
        *sorted(r.lines_by_tag.items()),
    ]
    for name, t in rows:
        print(
            f"{name:<16}{_ratio(t.pos_covered, t.pos_total):>18}"
            f"{_ratio(t.neg_flagged, t.neg_total):>20}"
        )

    print(f"\nexposures (leaks): {len(r.exposures)}")
    for x in r.exposures:
        print(f"  - {x}")
    print(f"false positives: {len(r.false_positives)}")
    for cid, s in r.false_positives:
        print(f"  - {cid}: {s}")


def run(
    fixtures: Path = _FIXTURES,
    detector: Detector | None = None,
    with_perturb: bool = False,
) -> int:
    if not fixtures.exists():
        print(
            f"no fixtures at {fixtures}\n"
            "copy fixtures/detect/cases.example.json to cases.json and fill in "
            "your own synthetic values (cases.json is gitignored).",
            file=sys.stderr,
        )
        return 2
    data = json.loads(fixtures.read_text())
    if detector is None:
        detector = RuleDetector(RuleConfig(terms=data.get("terms", [])))
    r = evaluate(data, detector)
    print_report(r)
    if with_perturb:
        _print_perturbed(evaluate(perturb(data), detector))
    return 1 if r.exposures or r.false_positives else 0


def _print_perturbed(r: Report) -> None:
    by_kind: dict[str, int] = defaultdict(int)
    for x in r.exposures:
        by_kind[x.split(" ")[0].rsplit("~", 1)[1]] += 1
    total = sum(r.recall_tot.values())
    print(
        f"\nOCR-perturbed credential spans: {total - len(r.exposures)}/{total} covered"
    )
    print(
        "  exposures by confusion: "
        + ", ".join(f"{k} {v}" for k, v in sorted(by_kind.items()))
    )
    for x in r.exposures:
        print(f"  - {x}")


def _build_detector(args: argparse.Namespace, fixtures: Path) -> Detector | None:
    if args.detector == "rules":
        return None
    from .gliner_detect import GlinerDetector, load_model

    gliner = GlinerDetector(
        load_model(device=args.device, dtype=args.dtype),
        threshold=args.threshold,
        lines_per_chunk=args.lines_per_chunk,
    )
    if args.detector == "gliner":
        return gliner
    terms = (
        json.loads(fixtures.read_text()).get("terms", []) if fixtures.exists() else []
    )
    return UnionDetector(RuleDetector(RuleConfig(terms=terms)), gliner)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m safe_share.eval_detect")
    ap.add_argument("fixtures", nargs="?", type=Path, default=_FIXTURES)
    ap.add_argument("--detector", choices=["rules", "gliner", "union"], default="rules")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--lines-per-chunk", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--perturb", action="store_true")
    ap.add_argument(
        "--name-cache",
        action="store_true",
        help="one session name cache across all cases, in file order",
    )
    args = ap.parse_args(argv)
    if args.name_cache and (args.detector != "union" or args.perturb):
        ap.error("--name-cache needs --detector union and no --perturb")
    if not args.fixtures.exists():
        return run(args.fixtures)  # reports the missing file without loading a model
    detector = _build_detector(args, args.fixtures)
    if args.name_cache:
        from .name_cache import NameCacheDetector

        detector = NameCacheDetector(detector)
    return run(args.fixtures, detector, args.perturb)


if __name__ == "__main__":
    raise SystemExit(main())
