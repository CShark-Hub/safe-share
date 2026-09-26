"""Load the app configuration from a TOML file.

Resolution order: an explicit path, then $SAFE_SHARE_CONFIG, then
`safe_share.toml` at the repo root, then the committed `safe_share.example.toml`,
then built-in defaults (emails + credentials on, no terms, no GLiNER).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .detect import RuleConfig

_ROOT = Path(__file__).resolve().parents[2]
_LOCAL = _ROOT / "safe_share.toml"
_EXAMPLE = _ROOT / "safe_share.example.toml"


@dataclass(frozen=True)
class AppConfig:
    rules: RuleConfig
    gliner: bool  # add model-based person-name detection
    source: str  # file used, or "defaults"


def load_config(path: Path | None = None) -> AppConfig:
    chosen = (
        path
        or _env_path()
        or (_LOCAL if _LOCAL.exists() else None)
        or (_EXAMPLE if _EXAMPLE.exists() else None)
    )
    if chosen is None:
        return AppConfig(RuleConfig(), gliner=False, source="defaults")
    data = tomllib.loads(chosen.read_text())
    rules = RuleConfig(
        emails=bool(data.get("emails", True)),
        credentials=bool(data.get("credentials", True)),
        terms=[str(t) for t in data.get("terms", [])],
    )
    return AppConfig(rules, gliner=bool(data.get("gliner", False)), source=str(chosen))


def _env_path() -> Path | None:
    raw = os.environ.get("SAFE_SHARE_CONFIG")
    return Path(raw) if raw else None
