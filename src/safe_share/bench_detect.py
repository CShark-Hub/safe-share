"""Latency of GLiNER detection on a synthetic ~100-line frame built from fixtures.

Run: python -m safe_share.bench_detect [--device mps] [--dtype float16] [--lines-per-chunk 1]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from .gliner_detect import GlinerDetector, load_model

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "detect" / "names.json"


def frame_lines(n: int, seed: int = 0) -> list[str]:
    lines = [
        ln for c in json.loads(_FIXTURES.read_text())["cases"] for ln in c["lines"]
    ]
    return random.Random(seed).sample(lines, n)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m safe_share.bench_detect")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--lines-per-chunk", type=int, default=None)
    ap.add_argument("--lines", type=int, default=100)
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args(argv)

    t = time.perf_counter()
    model = load_model(device=args.device, dtype=args.dtype)
    load_s = time.perf_counter() - t

    det = GlinerDetector(model, lines_per_chunk=args.lines_per_chunk)
    texts = frame_lines(args.lines)
    for _ in range(3):
        det.detect(texts)
    times = []
    for _ in range(args.runs):
        t = time.perf_counter()
        det.detect(texts)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    p95 = times[min(len(times) - 1, round(0.95 * len(times)) - 1)]
    print(
        f"device={args.device} dtype={args.dtype} lines_per_chunk={args.lines_per_chunk} "
        f"lines={args.lines} load={load_s:.1f}s "
        f"p50={statistics.median(times):.0f}ms p95={p95:.0f}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
