#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Strategy runtime_config demo.

Scenarios:
1) run_backtest with strategy_runtime_config override enabled.
2) run_backtest with runtime_config_override=False.
3) run_from_checkpoint with strategy_runtime_config override enabled.
"""

import os
from typing import List

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy, StrategyRuntimeConfig


class RuntimeBacktestConflictStrategy(Strategy):
    """Backtest conflict strategy."""

    def __init__(self) -> None:
        """Initialize state."""
        self.errors: list[str] = []
        self.runtime_config = StrategyRuntimeConfig(error_mode="raise")

    def on_bar(self, bar: Bar) -> None:
        """Raise on each bar."""
        raise ValueError("backtest_conflict")

    def on_error(self, error: Exception, source: str, payload: object = None) -> None:
        """Record error callback source."""
        self.errors.append(source)


class RuntimeWarmConflictStrategy(Strategy):
    """Warm start conflict strategy."""

    def __init__(self) -> None:
        """Initialize state."""
        self.errors: list[str] = []
        self.runtime_config = StrategyRuntimeConfig(error_mode="raise")

    def on_bar(self, bar: Bar) -> None:
        """Raise only in restored phase."""
        if self.is_restored:
            raise ValueError("warm_conflict")

    def on_error(self, error: Exception, source: str, payload: object = None) -> None:
        """Record error callback source."""
        self.errors.append(source)


def make_bars(start: str, periods: int, symbol: str = "TEST") -> List[Bar]:
    """Build synthetic daily bars."""
    idx = pd.date_range(start=start, periods=periods, freq="D")
    bars: list[Bar] = []
    for i, ts in enumerate(idx):
        price = 100.0 + i
        bars.append(
            Bar(
                timestamp=int(ts.value),
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price + 0.5,
                volume=1000.0 + i,
                symbol=symbol,
            )
        )
    return bars


def scenario_backtest_override_true() -> None:
    """Run override=true scenario and warning dedup demo."""
    bars = make_bars("2023-01-01", 2)
    shared_strategy = RuntimeBacktestConflictStrategy()
    aq.run_backtest(
        data=bars,
        strategy=shared_strategy,
        symbols="TEST",
        show_progress=False,
        strategy_runtime_config={"error_mode": "continue"},
    )
    aq.run_backtest(
        data=bars,
        strategy=shared_strategy,
        symbols="TEST",
        show_progress=False,
        strategy_runtime_config={"error_mode": "continue"},
    )
    print("scenario1_done")
    print(f"scenario1_errors={len(shared_strategy.errors)}")


def scenario_backtest_override_false() -> None:
    """Run override=false scenario."""
    bars = make_bars("2023-02-01", 1)
    try:
        aq.run_backtest(
            data=bars,
            strategy=RuntimeBacktestConflictStrategy,
            symbols="TEST",
            show_progress=False,
            strategy_runtime_config={"error_mode": "continue"},
            runtime_config_override=False,
        )
    except ValueError as exc:
        print(f"scenario2_exception={exc}")


def scenario_warm_start_override_true() -> None:
    """Run warm-start override scenario."""
    checkpoint_path = "runtime_config_demo_snapshot.pkl"
    phase1 = make_bars("2023-03-01", 2)
    phase2 = make_bars("2023-03-03", 2)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    result1 = aq.run_backtest(
        data=phase1,
        strategy=RuntimeWarmConflictStrategy,
        symbols="TEST",
        show_progress=False,
    )
    aq.save_checkpoint(result1.engine, result1.strategy, checkpoint_path)  # type: ignore[arg-type]

    result2 = aq.run_from_checkpoint(
        checkpoint_path=checkpoint_path,
        data=phase2,
        symbols="TEST",
        show_progress=False,
        strategy_runtime_config={"error_mode": "continue"},
    )

    strategy = result2.strategy
    print("scenario3_done")
    if strategy is not None:
        print(f"scenario3_errors={len(strategy.errors)}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def main() -> None:
    """Run all demo scenarios."""
    scenario_backtest_override_true()
    scenario_backtest_override_false()
    scenario_warm_start_override_true()


if __name__ == "__main__":
    main()
