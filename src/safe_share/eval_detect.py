"""String-level evaluation of a detector against repo fixtures.

Pure text, no screen. Reports per-type span recall/precision, false positives, an
exposure count (expected spans not fully covered by the union of detections),
and line-level numbers, since redaction covers a whole line on any hit.
Run: python -m safe_share.eval_detect [fixtures.json]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .detect import Detection, Detector, RuleConfig, RuleDetector

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


def run(fixtures: Path = _FIXTURES, detector: Detector | None = None) -> int:
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

    recall_hit: dict[str, int] = defaultdict(int)
    recall_tot: dict[str, int] = defaultdict(int)
    prec_hit: dict[str, int] = defaultdict(int)
    prec_tot: dict[str, int] = defaultdict(int)
    exposures: list[str] = []
    false_positives: list[tuple[str, str]] = []  # (case id, snippet)
    lines_all = _LineTally()
    lines_by_tag: dict[str, _LineTally] = defaultdict(_LineTally)
    lines_by_type: dict[str, _LineTally] = defaultdict(_LineTally)

    for case in data["cases"]:
        lines = case["lines"]
        expected = [_resolve_expected(e, lines) for e in case["expect"]]
        found = detector.detect(lines)

        positive_lines = {e.line for e in expected}
        hit_lines = {d.line_index for d in found}
        for i in range(len(lines)):
            positive, hit = i in positive_lines, i in hit_lines
            lines_all.add(positive, hit)
            for tag in case.get("tags", []):
                lines_by_tag[tag].add(positive, hit)
        for etype, line in {(e.type, e.line) for e in expected}:
            lines_by_type[f"type:{etype}"].add(True, line in hit_lines)

        for e in expected:
            recall_tot[e.type] += 1
            if _fully_covered(e, found, same_type=True):
                recall_hit[e.type] += 1
            if not _fully_covered(e, found, same_type=False):
                exposures.append(f"{case['id']} (line {e.line}, {e.type})")

        all_negative = not expected
        for d in found:
            prec_tot[d.entity_type] += 1
            hit = any(
                d.line_index == e.line
                and _overlap(d.start, d.end, e.start, e.end)
                and d.entity_type == e.type
                for e in expected
            )
            if hit:
                prec_hit[d.entity_type] += 1
            elif not any(
                d.line_index == e.line and _overlap(d.start, d.end, e.start, e.end)
                for e in expected
            ):
                # A detection over clean content is a false positive. Snippet
                # text is logged only for all-negative cases, whose lines are
                # synthetic non-secrets by construction (privacy rule 6);
                # mixed cases report type + coordinates only.
                if all_negative:
                    snippet = lines[d.line_index][d.start : d.end][:32]
                    false_positives.append((case["id"], f"{d.entity_type}:{snippet}"))
                else:
                    false_positives.append(
                        (
                            case["id"],
                            f"{d.entity_type}@L{d.line_index}:{d.start}-{d.end}",
                        )
                    )

    types = sorted(set(recall_tot) | set(prec_tot))
    print(f"{'type':<16}{'recall':>12}{'precision':>14}")
    for t in types:
        rt, ph, pt = recall_tot[t], prec_hit[t], prec_tot[t]
        rec = f"{recall_hit[t]}/{rt}" if rt else "-"
        prc = f"{ph}/{pt}" if pt else "-"
        print(f"{t:<16}{rec:>12}{prc:>14}")

    print(f"\n{'lines':<16}{'covered':>18}{'wrongly masked':>20}")
    rows = [
        ("all", lines_all),
        *sorted(lines_by_type.items()),
        *sorted(lines_by_tag.items()),
    ]
    for name, t in rows:
        print(
            f"{name:<16}{_ratio(t.pos_covered, t.pos_total):>18}"
            f"{_ratio(t.neg_flagged, t.neg_total):>20}"
        )

    print(f"\nexposures (leaks): {len(exposures)}")
    for x in exposures:
        print(f"  - {x}")
    print(f"false positives: {len(false_positives)}")
    for cid, s in false_positives:
        print(f"  - {cid}: {s}")

    return 1 if exposures or false_positives else 0


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1])) if len(sys.argv) > 1 else run())
