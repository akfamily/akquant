#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Function-style warm start demo."""

import tempfile
from pathlib import Path
from typing import Any

import akquant as aq
import pandas as pd
from akquant import Bar


def make_bars(rows: list[tuple[str, float]]) -> list[Bar]:
    """Build deterministic bar data for a two-phase warm start run."""
    bars: list[Bar] = []
    for dt_str, close in rows:
        ts = pd.Timestamp(dt_str, tz="Asia/Shanghai").value
        bars.append(
            Bar(
                timestamp=int(ts),
                open=close,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=1000.0,
                symbol="FUNC_WARM",
            )
        )
    return bars


def initialize(ctx: Any) -> None:
    """Initialize state that will be persisted into the checkpoint."""
    ctx.events = []
    ctx.processed_closes = []
    ctx.start_count = 0
    ctx.resume_count = 0


def on_start(ctx: Any) -> None:
    """Run on both cold start and warm start."""
    ctx.start_count += 1
    ctx.events.append(f"start:restored={int(bool(ctx.is_restored))}")


def on_resume(ctx: Any) -> None:
    """Run only when the strategy is restored from a checkpoint."""
    ctx.resume_count += 1
    ctx.events.append(
        f"resume:bars={len(ctx.processed_closes)}:starts={ctx.start_count}"
    )


def on_bar(ctx: Any, bar: Bar) -> None:
    """Accumulate simple state so the restored run can prove continuity."""
    ctx.processed_closes.append(float(bar.close))
    ctx.events.append(f"bar:{bar.close:.2f}")


def main() -> None:
    """Run a two-phase function-style warm start demo."""
    checkpoint_path = (
        Path(tempfile.gettempdir()) / "akquant_functional_warm_start_demo.pkl"
    )
    checkpoint_path.unlink(missing_ok=True)

    phase1 = make_bars(
        [
            ("2023-01-03 09:30:00", 10.0),
            ("2023-01-04 09:30:00", 10.4),
        ]
    )
    phase2 = make_bars(
        [
            ("2023-01-05 09:30:00", 10.8),
            ("2023-01-06 09:30:00", 11.2),
        ]
    )

    result1 = aq.run_backtest(
        data=phase1,
        strategy=on_bar,
        initialize=initialize,
        on_start=on_start,
        on_resume=on_resume,
        symbols="FUNC_WARM",
        show_progress=False,
        initial_cash=100000.0,
    )
    if result1.engine is None:
        raise RuntimeError("Engine should not be None")
    aq.save_checkpoint(result1.engine, result1.strategy, str(checkpoint_path))

    result2 = aq.run_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        data=phase2,
        symbols="FUNC_WARM",
        show_progress=False,
    )

    strategy = result2.strategy
    if strategy is None:
        raise RuntimeError("Strategy should not be None")

    print("phase1_events=" + "|".join(result1.strategy.events))  # type: ignore[union-attr]
    print("phase2_events=" + "|".join(strategy.events))
    print(
        "processed_closes="
        + ",".join(f"{price:.2f}" for price in strategy.processed_closes)
    )
    print(f"start_count={strategy.start_count}")
    print(f"resume_count={strategy.resume_count}")
    print("done_functional_warm_start_demo")

    checkpoint_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
