"""Hard Memory Budget Governor with pressure-scaled forgetting and eviction.

Monitors live OS process RSS (Windows ``GetProcessMemoryInfo`` via ``ctypes``,
Linux ``/proc/self/statm``, or ``tracemalloc`` fallback — zero external deps)
against a configurable ceiling (default **16 GB**).

As RSS approaches ``max_bytes``, computes a convex pressure curve ``p in [0, 1]``:
  - ``ratio <= soft_ratio`` (default 50% / 8 GB): ``pressure = 0.0`` (no forced forgetting)
  - ``soft_ratio < ratio < hard_ratio`` (default 92%): smooth quadratic ramp
    ``((ratio - soft) / (hard - soft)) ** 2`` — the closer to 16 GB, the stronger
    forgetting and eviction become!
  - ``ratio >= hard_ratio``: ``pressure = 1.0`` (maximum aggressive pruning + GC)
"""
from __future__ import annotations

import ctypes
import gc
import os
import sys
import tracemalloc
from typing import Dict, List, Optional, Sequence


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def current_rss_bytes() -> int:
    """Return current process resident set size (RSS) in bytes using OS APIs."""
    if sys.platform == "win32":
        try:
            psapi = ctypes.WinDLL("psapi.dll")
            kernel32 = ctypes.WinDLL("kernel32.dll")
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                ctypes.c_uint32,
            ]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int
            hproc = kernel32.GetCurrentProcess()
            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            if psapi.GetProcessMemoryInfo(hproc, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except Exception:
            pass
    else:
        try:
            with open("/proc/self/statm", "r", encoding="ascii") as f:
                parts = f.read().split()
                if len(parts) >= 2:
                    return int(parts[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except Exception:
            pass
    if tracemalloc.is_tracing():
        cur, _peak = tracemalloc.get_traced_memory()
        return int(cur)
    return 0


class MemoryBudgetGovernor:
    """Enforces a hard memory ceiling (default 16 GB) via convex pressure-scaled eviction."""

    def __init__(
        self,
        max_bytes: int = 16 * (1024 ** 3),
        *,
        soft_ratio: float = 0.50,
        hard_ratio: float = 0.92,
        power: float = 2.0,
    ):
        self.max_bytes = max(int(max_bytes), 1024 * 1024)
        self.soft_ratio = float(min(max(soft_ratio, 0.05), 0.90))
        self.hard_ratio = float(min(max(hard_ratio, self.soft_ratio + 0.05), 0.99))
        self.power = max(float(power), 1.0)
        self.total_govern_calls: int = 0
        self.total_evicted_items: int = 0
        self.last_pressure: float = 0.0
        self.last_rss_bytes: int = 0

    def pressure(self, rss_bytes: Optional[int] = None) -> float:
        """Compute memory pressure in [0.0, 1.0] (convex acceleration near ``max_bytes``)."""
        rss = current_rss_bytes() if rss_bytes is None else max(int(rss_bytes), 0)
        self.last_rss_bytes = rss
        ratio = float(rss) / float(self.max_bytes)
        if ratio <= self.soft_ratio:
            p = 0.0
        elif ratio >= self.hard_ratio:
            p = 1.0
        else:
            frac = (ratio - self.soft_ratio) / (self.hard_ratio - self.soft_ratio)
            p = float(frac ** self.power)
        self.last_pressure = p
        return p

    def govern(
        self,
        targets: Sequence[object],
        *,
        rss_bytes: Optional[int] = None,
    ) -> Dict[str, float]:
        """Measure pressure and invoke ``prune_under_pressure(p)`` on all registered targets."""
        self.total_govern_calls += 1
        p = self.pressure(rss_bytes=rss_bytes)
        evicted = 0
        if p > 0.0:
            for obj in targets:
                fn = getattr(obj, "prune_under_pressure", None)
                if callable(fn):
                    evicted += int(fn(p))
            if p >= 0.45 and evicted > 0:
                gc.collect()
            self.total_evicted_items += evicted
        rss_after = current_rss_bytes() if rss_bytes is None else int(rss_bytes)
        return dict(
            rss_mb=round(rss_after / (1024.0 * 1024.0), 1),
            max_mb=round(self.max_bytes / (1024.0 * 1024.0), 1),
            ratio=round(rss_after / float(self.max_bytes), 4),
            pressure=round(p, 4),
            evicted_items=int(evicted),
            total_evicted=int(self.total_evicted_items),
        )
