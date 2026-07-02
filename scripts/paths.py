"""Centralized path configuration for the secure-inference benchmark scripts.

All external dependencies (the SHAFT/CrypTen checkout, the text-classification
data/models) are resolved from environment variables so the repository is
relocatable. Set them once before running anything:

    export SHAFT_ROOT=/path/to/SHAFT            # the SHAFT checkout (vendored modified CrypTen)
    export DATA_ROOT=/path/to/text-classification  # models + GLUE data

`QUEST_ROOT` (this repository's root) is derived from this file's location, so
it needs no configuration. Importing this module also inserts SHAFT_ROOT,
DATA_ROOT and QUEST_ROOT at the front of ``sys.path`` so that ``import crypten``
resolves to SHAFT's modified CrypTen and the ``scripts`` namespace package is
importable from any working directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# This repository's root (this file lives at <root>/scripts/paths.py).
QUEST_ROOT = Path(__file__).resolve().parents[1]

# External dependencies — MUST be provided by the user via environment variables.
SHAFT_ROOT = Path(os.environ["SHAFT_ROOT"]) if os.environ.get("SHAFT_ROOT") else None
DATA_ROOT = Path(os.environ["DATA_ROOT"]) if os.environ.get("DATA_ROOT") else None


def _require(name: str, value: Path | None) -> Path:
    if value is None:
        raise EnvironmentError(
            f"Environment variable {name} is not set. "
            f"Please `export {name}=/path/to/...` before running. See README.md."
        )
    return value


def get_shaft_root() -> Path:
    return _require("SHAFT_ROOT", SHAFT_ROOT)


def get_data_root() -> Path:
    return _require("DATA_ROOT", DATA_ROOT)


# Default model / data locations under DATA_ROOT (override via CLI if needed).
def default_bert_base_sst2() -> Path:
    return get_data_root() / "bert-base-cased-sst2"


def default_validation_file() -> Path:
    return get_data_root() / "glue" / "sst2" / "validation.parquet"


# Make imports work from any cwd. QUEST_ROOT is always safe to insert.
for _p in (str(QUEST_ROOT), str(SHAFT_ROOT) if SHAFT_ROOT else "", str(DATA_ROOT) if DATA_ROOT else ""):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
