"""GLiNER person/organization detection. Needs the `ml` extra; loads from the
local cache only, except `fetch_model()`."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from .detect import Detection

MODEL_ID = "gliner-community/gliner_small-v2.5"
MODEL_REVISION = "f227d3cd637bd4e6757ae143935316d062393341"
MODEL_VARIANT = "fp16"  # safetensors; the repo default is a pickle
# gliner loads this config at `main`; the cached ref is checked instead.
BACKBONE_ID = "microsoft/deberta-v3-small"
BACKBONE_REVISION = "a36c739020e01763fe789b4b85e2df55d6180012"

LABELS = ("person", "organization")
# Far under the 768-word max_len, so gliner never truncates (a silent miss);
# small chunks are also faster (bench_detect).
MAX_CHUNK_WORDS = 75
THRESHOLD = 0.7  # measured on fixtures/detect/names.json


class GlinerModel(Protocol):
    def inference(
        self,
        texts: list[str],
        labels: list[str],
        threshold: float = 0.5,
        batch_size: int = 8,
    ) -> list[list[dict[str, Any]]]: ...


class ModelUnavailable(RuntimeError):
    """Model or pinned backbone config missing from the local cache."""


class GlinerDetector:
    """Packs whole lines into newline-joined chunks, so the model sees context."""

    def __init__(
        self,
        model: GlinerModel,
        threshold: float = THRESHOLD,
        labels: Sequence[str] = LABELS,
        max_chunk_words: int = MAX_CHUNK_WORDS,
        lines_per_chunk: int | None = None,
        batch_size: int = 16,
    ) -> None:
        self._model = model
        self._threshold = threshold
        self._labels = list(labels)
        self._max_words = max_chunk_words
        self._lines_per_chunk = lines_per_chunk
        self._batch_size = batch_size

    def detect(self, texts: list[str]) -> list[Detection]:
        chunks = self._chunk(texts)
        if not chunks:
            return []
        results = self._model.inference(
            [text for text, _ in chunks],
            self._labels,
            threshold=self._threshold,
            batch_size=self._batch_size,
        )
        if len(results) != len(chunks):
            raise RuntimeError("model returned a different number of results")
        out: list[Detection] = []
        for (_, starts), entities in zip(chunks, results):
            for ent in entities:
                out += _split_to_lines(ent, starts, texts)
        return out

    def _chunk(self, texts: list[str]) -> list[tuple[str, list[tuple[int, int]]]]:
        """[(chunk text, [(line index, offset in chunk)])]."""
        chunks: list[tuple[str, list[tuple[int, int]]]] = []
        parts: list[str] = []
        starts: list[tuple[int, int]] = []
        words = 0
        offset = 0

        def flush() -> None:
            nonlocal parts, starts, words, offset
            if parts:
                chunks.append(("\n".join(parts), starts))
            parts, starts, words, offset = [], [], 0, 0

        for i, line in enumerate(texts):
            n = len(line.split())
            if n == 0:
                continue
            if n > self._max_words:
                # Fail, never truncate; the pipeline then publishes neutral.
                raise ValueError("OCR line exceeds the model input budget")
            full = words + n > self._max_words or (
                self._lines_per_chunk is not None
                and len(parts) >= self._lines_per_chunk
            )
            if full:
                flush()
            starts.append((i, offset))
            parts.append(line)
            words += n
            offset += len(line) + 1
        flush()
        return chunks


def _split_to_lines(
    ent: dict[str, Any], starts: list[tuple[int, int]], texts: list[str]
) -> list[Detection]:
    """Chunk entity -> per-line spans; one crossing a line break marks each line."""
    s, e = int(ent["start"]), int(ent["end"])
    out = []
    for line_index, line_start in starts:
        line_end = line_start + len(texts[line_index])
        lo, hi = max(s, line_start), min(e, line_end)
        if lo < hi:
            out.append(
                Detection(
                    line_index,
                    lo - line_start,
                    hi - line_start,
                    str(ent["label"]),
                    source="gliner",
                    score=float(ent["score"]),
                )
            )
    return out


def _check_backbone_pinned() -> None:
    from huggingface_hub.constants import HF_HUB_CACHE

    ref = (
        Path(HF_HUB_CACHE)
        / f"models--{BACKBONE_ID.replace('/', '--')}"
        / "refs"
        / "main"
    )
    try:
        cached = ref.read_text().strip()
    except OSError as exc:
        raise ModelUnavailable("backbone config is not cached") from exc
    if cached != BACKBONE_REVISION:
        raise ModelUnavailable("cached backbone config is not the pinned revision")


def _quiet_libraries() -> None:
    import warnings

    import transformers

    transformers.logging.set_verbosity_error()
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.")


def load_model(device: str = "cpu", dtype: str = "float32") -> Any:
    """Load the pinned model from the local cache; `dtype` is float16 or float32."""
    # Read by huggingface_hub at import time.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    _quiet_libraries()
    _check_backbone_pinned()

    import torch
    from gliner import GLiNER

    try:
        model = GLiNER.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            variant=MODEL_VARIANT,
            local_files_only=True,
            map_location=device,
        )
    except OSError as exc:
        raise ModelUnavailable("model is not cached") from exc
    # gliner rejects a dtype that differs from the variant's, so cast after load.
    model.to(dtype=getattr(torch, dtype))
    model.eval()
    return model


def fetch_model() -> None:
    """Download the pinned model into the Hugging Face cache (uses network)."""
    from gliner import GLiNER

    GLiNER.from_pretrained(MODEL_ID, revision=MODEL_REVISION, variant=MODEL_VARIANT)
