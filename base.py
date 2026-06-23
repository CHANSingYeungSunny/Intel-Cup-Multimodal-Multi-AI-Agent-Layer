"""
Shared base for Multi-layer ORM models.

Separated from database.py so that models.py can import MultiBase
without triggering a name collision with the Single-layer database module.
"""

from sqlalchemy.orm import DeclarativeBase


class MultiBase(DeclarativeBase):
    """Base class for all Multi-layer ORM models (separate from Single-layer Base)."""
    pass
