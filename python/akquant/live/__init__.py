"""Live/paper trading entry points.

Public surface: :func:`run_live` mirrors ``run_backtest`` — a single top-level
function for live/paper sessions. The former ``LiveRunner`` class is now an
internal implementation detail (``akquant.live._runner.LiveRunner``) and is no
longer part of the public API.
"""

from ._facade import run_live

__all__ = ["run_live"]
