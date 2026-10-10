"""GLiNER person-name detection. Needs the `ml` extra; loads from the
local cache only. Downloads run only in a child process (`fetch_in_subprocess`)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from .detect import Detection

MODEL_ID = "gliner-community/gliner_small-v2.5"
MODEL_REVISION = "f227d3cd637bd4e6757ae143935316d062393341"
MODEL_VARIANT = "fp16"  # safetensors; the repo default is a pickle
# gliner loads this config at `main`; the cached ref is checked instead.
BACKBONE_ID = "microsoft/deberta-v3-small"
BACKBONE_REVISION = "a36c739020e01763fe789b4b85e2df55d6180012"

LABELS = (
    "person",
)  # organization also hits public brands and UI; clients go in `terms`
# Words as gliner counts them (its WhitespaceTokenSplitter): punctuation is a word.
_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")
# Far under gliner's 768-word max_len, which truncates (a silent miss); small
# chunks are also faster (bench_detect).
MAX_CHUNK_WORDS = 75
# The backbone has 512 positions; the label prompt takes 7 subwords.
MAX_CHUNK_SUBWORDS = 480
THRESHOLD = 0.7  # measured on fixtures/detect/names.json
FETCH_TIMEOUT = 900.0  # seconds
# Read before _force_offline overwrites it: a user's own offline setting wins.
_USER_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_ONLINE_BLOCKERS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_PROGRESS_BARS",
)


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
        self._subwords = _subword_counter(model)

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
        words = subwords = offset = 0

        def flush() -> None:
            nonlocal parts, starts, words, subwords, offset
            if parts:
                chunks.append(("\n".join(parts), starts))
            parts, starts, words, subwords, offset = [], [], 0, 0, 0

        split = [(i, _WORD.findall(line)) for i, line in enumerate(texts)]
        split = [(i, w) for i, w in split if w]
        sub = (
            self._subwords([w for _, w in split])
            if self._subwords
            else [0] * len(split)
        )
        for (i, w), m in zip(split, sub, strict=True):
            n = len(w)
            if n > self._max_words or m > MAX_CHUNK_SUBWORDS:
                # Fail, never truncate; the pipeline then publishes neutral.
                raise ValueError("OCR line exceeds the model input budget")
            full = (
                words + n > self._max_words
                or subwords + m > MAX_CHUNK_SUBWORDS
                or (
                    self._lines_per_chunk is not None
                    and len(parts) >= self._lines_per_chunk
                )
            )
            if full:
                flush()
            starts.append((i, offset))
            parts.append(texts[i])
            words += n
            subwords += m
            offset += len(texts[i]) + 1
        flush()
        return chunks


def _subword_counter(model: object) -> Callable[[list[list[str]]], list[int]] | None:
    """Subwords per line from the model's tokenizer; None for models without one."""
    tok = getattr(getattr(model, "data_processor", None), "transformer_tokenizer", None)
    if tok is None:
        return None

    def count(lines: list[list[str]]) -> list[int]:
        if not lines:
            return []
        ids = tok(lines, is_split_into_words=True, add_special_tokens=False)[
            "input_ids"
        ]
        return [len(x) for x in ids]

    return count


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


def _check_cache() -> None:
    """Offline mode active, backbone ref pinned, fp16 weights present (so gliner
    cannot fall back to the repo's pickle)."""
    from huggingface_hub import constants
    from huggingface_hub.constants import HF_HUB_CACHE

    if not constants.HF_HUB_OFFLINE:
        raise ModelUnavailable("offline mode is not active")
    # gliner loads a relative directory named like the repo id before the cache.
    if Path(MODEL_ID).exists():
        raise ModelUnavailable(f"a local ./{MODEL_ID} would shadow the cached model")
    weights = (
        Path(HF_HUB_CACHE)
        / f"models--{MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / MODEL_REVISION
        / f"model.{MODEL_VARIANT}.safetensors"
    )
    if not weights.exists():
        raise ModelUnavailable("model is not cached")
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

    # Filters first: importing transformers imports torch, which warns.
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"huggingface_hub\.")
    import transformers

    transformers.logging.set_verbosity_error()


def _force_offline() -> None:
    # Read by huggingface_hub at import time; _check_cache verifies it took.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def load_model(device: str = "cpu", dtype: str = "float32") -> Any:
    """Load the pinned model from the local cache; `dtype` is float16 or float32."""
    _force_offline()
    _quiet_libraries()
    _check_cache()

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


def load_best() -> tuple[Any, str]:
    """MPS fp16 if it loads and finds a name, else CPU fp32. Returns (model, "device/dtype")."""
    _force_offline()
    import torch

    options = [("cpu", "float32")]
    if torch.backends.mps.is_available():
        options.insert(0, ("mps", "float16"))
    for device, dtype in options:
        try:
            model = load_model(device, dtype)
            found = GlinerDetector(model).detect(["Meeting with Priya Venkat at noon"])
            # A model that runs but finds nothing (e.g. fp16 overflow) is a failure.
            if not any(d.entity_type == "person" for d in found):
                raise RuntimeError("warm-up found no person")
            return model, f"{device}/{dtype}"
        except ModelUnavailable:
            raise  # another device will not find a missing file either
        except Exception as exc:  # noqa: BLE001 - try the next device
            print(f"[detect] GLiNER failed on {device}/{dtype}: {type(exc).__name__}")
    raise ModelUnavailable("model did not run on any device")


def model_cached() -> bool:
    """True if the pinned model is in the local cache. Forces offline mode here."""
    _force_offline()
    try:
        _check_cache()
    except ModelUnavailable:
        return False
    return True


def fetch_in_subprocess(timeout: float = FETCH_TIMEOUT) -> bool:
    """Download the pinned model in a child process, so this one never goes online.
    True only if the cache then holds it."""
    if _USER_OFFLINE:
        print("[detect] HF_HUB_OFFLINE is set: not downloading")
        return False
    env = {k: v for k, v in os.environ.items() if k not in _ONLINE_BLOCKERS}
    try:
        done = subprocess.run(
            # -P: the current directory must not shadow installed modules.
            [sys.executable, "-P", "-m", "safe_share.gliner_detect", "--fetch"],
            env=env,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0 and model_cached()


def fetch_model() -> None:
    """Download the pinned model into the Hugging Face cache (uses network)."""
    from gliner import GLiNER

    GLiNER.from_pretrained(MODEL_ID, revision=MODEL_REVISION, variant=MODEL_VARIANT)


if __name__ == "__main__":
    if sys.argv[1:] != ["--fetch"]:
        raise SystemExit("usage: python -m safe_share.gliner_detect --fetch")
    fetch_model()
