"""Centralized path bootstrap for core/ subpackages.

Allows both direct CLI execution (``python core/substrate_nodes.py``) and
legacy flat imports (``from node_runtime import ...``) to work seamlessly
alongside structured subpackage layout under ``core/{env,memory,stage,planning,runner,legacy}``.
"""
from __future__ import annotations

from pathlib import Path
import sys

CORE_ROOT = Path(__file__).resolve().parent
SUBPACKAGES = ("env", "memory", "stage", "planning", "runner", "legacy")


def ensure_core_paths() -> None:
    """Register ``core/`` and all six architectural subpackages on ``sys.path``."""
    for target in (CORE_ROOT, *(CORE_ROOT / sub for sub in SUBPACKAGES)):
        path_str = str(target)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def find_core_file(filename: str) -> Path:
    """Locate a module file across ``core/`` and its subpackages."""
    direct = CORE_ROOT / filename
    if direct.exists():
        return direct
    for sub in SUBPACKAGES:
        cand = CORE_ROOT / sub / filename
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{filename} not found under {CORE_ROOT}")
