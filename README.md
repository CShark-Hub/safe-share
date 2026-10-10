# Safe Share

On-device redaction for screen sharing on macOS. Safe Share captures your screen,
finds names, emails, credentials and client terms with OCR and a local NER model,
covers them with opaque labels, and shows the result in a window you share in Teams
instead of your real screen. Screen content never leaves your Mac.

Full documentation is in progress.

## Quickstart

Tested on macOS 27, Apple silicon. Needs [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd <repo-dir>
uv sync --extra ml
uv run safe-share        # first run downloads the GLiNER model once, before capture
```

Grant Screen Recording to your terminal when macOS asks, then run it again. In Teams,
share the **Safe Share** window, not your screen.

## License

Licensed under the [Apache License 2.0](LICENSE).

Copyright (c) 2026 CSHARK Commit Hackathon Contributors

## Team

- Krzysiek Kulig
