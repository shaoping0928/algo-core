"""Configure import paths so unit tests run directly without manual PYTHONPATH."""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("core", "modules", "probes"):
    _path = str(_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)
