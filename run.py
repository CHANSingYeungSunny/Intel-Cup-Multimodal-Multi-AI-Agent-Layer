"""
Multi AI Agent Layer — launcher script.

Fixes sys.path so the Multi-layer modules are found before the Single-layer
equivalents, then starts uvicorn with the multi-agent FastAPI app.
"""

import os
import sys

# Ensure the Multi-layer root is first in sys.path so that ``import main``
# finds THIS layer's main.py (with create_multi_app), not the Single-layer
# main.py (which only has create_app).
_MULTI_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MULTI_ROOT not in sys.path:
    sys.path.insert(0, _MULTI_ROOT)

# Force-load all Multi-layer shadow modules BEFORE uvicorn imports anything.
# This caches config, database, schemas, models, and main as the Multi-layer
# versions in sys.modules.
import _ensure_imports  # noqa: E402

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:create_multi_app",
        host="0.0.0.0",
        port=8000,
        factory=True,
        reload=True,
    )
