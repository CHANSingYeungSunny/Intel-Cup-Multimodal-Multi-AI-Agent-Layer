"""
Centralised helper to force-load Multi-layer shadow modules into sys.modules.

Import this FIRST (before any other Multi-layer import) to guarantee that
``config``, ``database``, ``schemas``, and ``models`` resolve to the
Multi-layer versions (which re-export everything from the Single-layer
equivalents).
"""

import importlib.util
import os
import sys

# Absolute path to this (Multi) layer
_MULTI_ROOT = os.path.dirname(os.path.abspath(__file__))

_MODULES = ["config", "database", "schemas", "models", "main"]


def ensure_multi_modules():
    """Force-load Multi-layer config, database, schemas, models into sys.modules."""
    for _name in _MODULES:
        _path = os.path.join(_MULTI_ROOT, f"{_name}.py")
        if not os.path.exists(_path):
            continue
        _spec = importlib.util.spec_from_file_location(_name, _path)
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_name] = _mod
        _spec.loader.exec_module(_mod)


# Run immediately on import so that any subsequent ``import config`` etc.
# finds the Multi-layer version.
ensure_multi_modules()
