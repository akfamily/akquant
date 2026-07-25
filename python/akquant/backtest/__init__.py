from .engine import (
    BacktestStreamEvent,
    FunctionalStrategy,
    make_fill_policy,  # kept as a raising stub (Task 5) for a helpful error
    run_backtest,
    run_from_checkpoint,
)
from .fill_mode import (
    CurrentClose,
    FillMode,
    NextAverage,
    NextClose,
    NextHighLowMid,
    NextOpen,
)
from .merge import MergedResult, merge_results
from .result import BacktestResult

__all__ = [
    "BacktestResult",
    "BacktestStreamEvent",
    "MergedResult",
    "run_backtest",
    "run_from_checkpoint",
    "merge_results",
    "make_fill_policy",
    "FillMode",
    "NextOpen",
    "NextClose",
    "NextAverage",
    "NextHighLowMid",
    "CurrentClose",
    "FunctionalStrategy",
]
