"""Reuse one fitted probe model across diagnostic arms.

Probes often vary only the rule margin, the status callback, or a reporting
option while the fitted world model and agent are identical.  Refitting for
every arm wastes CPU.  ``load_or_fit`` stores the fitted pair together with
the configuration and reuses it only when the configuration matches.

The cache is deliberately probe-side: it changes no training or decision
path, and a corrupt or mismatched cache falls back to a fresh fit.
"""
from __future__ import annotations

import os
import pickle
import tempfile
from typing import Callable, Optional, Tuple


def load_or_fit(config: dict, path: Optional[str] = None,
                refit: bool = False,
                fit_fn: Optional[Callable] = None) -> Tuple[object, object, bool]:
    """Return ``(world, agent, loaded)`` from a cache or a fresh fit."""
    if fit_fn is None:
        from substrate_nodes import fit as fit_fn
    if path and not refit and os.path.exists(path):
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            if payload.get("config") == config:
                return payload["world"], payload["agent"], True
        except Exception:
            # A partial/corrupt cache must never block a probe.
            pass
    world, agent = fit_fn(**config)
    if path:
        directory = os.path.dirname(os.path.abspath(path))
        try:
            os.makedirs(directory, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".probe-cache-", dir=directory
            )
            with os.fdopen(descriptor, "wb") as handle:
                pickle.dump(
                    dict(config=config, world=world, agent=agent),
                    handle,
                )
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except Exception:
                pass
    return world, agent, False
