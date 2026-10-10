# Third-party licenses

Safe Share is licensed under Apache 2.0. No dependency is under GPL, AGPL or SSPL, and
every dependency installed on macOS has a declared license. Checked on 2026-10-10 from
the installed package metadata (`uv sync --extra ml`, 51 packages).

## Direct dependencies

| Package | License | Used for |
|---|---|---|
| numpy | BSD-3-Clause (with 0BSD, MIT, Zlib, CC0-1.0 parts) | Image arrays |
| pillow | MIT-CMU | Image handling |
| pyobjc-core, pyobjc-framework-* | MIT | macOS APIs: ScreenCaptureKit, Vision, AppKit |
| gliner (`ml` extra) | Apache-2.0 | Name detection |
| pytest (dev) | MIT | Tests |
| ruff (dev) | MIT | Linting |

## AI models

| Model | License |
|---|---|
| [gliner-community/gliner_small-v2.5](https://huggingface.co/gliner-community/gliner_small-v2.5) | Apache-2.0 |
| [microsoft/deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small) (backbone config) | MIT |

The models are downloaded on first run and are not part of this repository.

## Indirect dependencies

All permissive (MIT, BSD, Apache-2.0, ISC, PSF-2.0, CNRI-Python, BSL-1.0), with two
packages under MPL-2.0:

| Package | License | Pulled in by |
|---|---|---|
| certifi | MPL-2.0 | huggingface_hub (via httpx, httpcore) |
| tqdm | MPL-2.0 AND MIT | gliner, huggingface_hub, transformers |

MPL-2.0 is a weak, file-level copyleft, not one of the excluded licenses. It applies to
changes made to those packages' own files. Safe Share uses both unmodified, so its own
code stays Apache 2.0.

Large indirect dependencies: torch (Apache-2.0, BSD and MIT parts), transformers,
tokenizers, safetensors and sentencepiece (Apache-2.0).

## Not installed on macOS

`uv.lock` also lists 20 Linux- and Windows-only dependencies of PyTorch and pytest
(CUDA and NVIDIA libraries, triton, colorama). uv never installs them on macOS, the only
platform Safe Share runs on.
