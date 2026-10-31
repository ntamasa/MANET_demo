"""Algorithm registry with lazy imports, so a missing module does not break the others."""
from __future__ import annotations

from importlib import import_module

_REGISTRY = {
    "CADENCE": ("manet.algorithms.cadence", "Cadence"),
    "DADENCE": ("manet.algorithms.dadence", "Dadence"),
    "ISDA": ("manet.algorithms.isda", "Isda"),
}

ALGORITHM_NAMES: list[str] = list(_REGISTRY)


def get_algorithm(name: str):
    """Return the Algorithm subclass registered under `name` (case-insensitive)."""
    module, cls = _REGISTRY[name.upper()]
    return getattr(import_module(module), cls)
