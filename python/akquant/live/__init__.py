"""Live/paper trading entry points.

Public surface: :func:`run_live` (added in the symmetry refactor) mirrors
``run_backtest``. ``LiveRunner`` remains importable during the migration and is
re-exported here for backward compatibility.
"""

from ._facade import run_live
from ._runner import LiveRunner

__all__ = ["LiveRunner", "run_live"]
