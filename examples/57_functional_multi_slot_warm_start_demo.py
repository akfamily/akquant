#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Function-style multi-slot warm start demo."""

import tempfile
from pathlib import Path
from typing import Any

import akquant as aq
import pandas as pd
from akquant import Bar


def make_bars(rows: list[tuple[str, float]]) -> list[Bar]:
    """Build deterministic bars for a two-phase multi-slot resume demo."""
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
                symbol="FUNC_SLOT_WARM",
            )
        )
    return bars


def initialize(ctx: Any) -> None:
    """Initialize per-slot state that should survive checkpoint restore."""
    ctx.events = []
    ctx.processed_closes = []
    ctx.start_count = 0
    ctx.resume_count = 0


def get_slot_id(ctx: Any, default: str) -> str:
    """Resolve the slot id consistently across lifecycle stages."""
    return str(
        getattr(ctx, "_owner_strategy_id", None)
        or getattr(ctx, "strategy_id", None)
        or default
    )


def on_start(ctx: Any) -> None:
    """Record start events for each slot."""
    ctx.start_count += 1
    slot_id = get_slot_id(ctx, "_default")
    ctx.events.append(f"{slot_id}:start:restored={int(bool(ctx.is_restored))}")


def on_resume(ctx: Any) -> None:
    """Record resume events for each restored slot."""
    ctx.resume_count += 1
    slot_id = get_slot_id(ctx, "_default")
    ctx.events.append(
        f"{slot_id}:resume:bars={len(ctx.processed_closes)}:starts={ctx.start_count}"
    )


def alpha_on_bar(ctx: Any, bar: Bar) -> None:
    """Primary slot callback."""
    slot_id = get_slot_id(ctx, "alpha")
    ctx.processed_closes.append(float(bar.close))
    ctx.events.append(f"{slot_id}:bar:{bar.close:.2f}")


def beta_on_bar(ctx: Any, bar: Bar) -> None:
    """Secondary slot callback."""
    slot_id = get_slot_id(ctx, "beta")
    ctx.processed_closes.append(float(bar.close))
    ctx.events.append(f"{slot_id}:bar:{bar.close:.2f}")


def main() -> None:
    """Run a two-phase multi-slot warm start demo."""
    checkpoint_path = (
        Path(tempfile.gettempdir()) / "akquant_functional_multi_slot_warm_start.pkl"
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
        strategy=alpha_on_bar,
        initialize=initialize,
        on_start=on_start,
        on_resume=on_resume,
        symbols="FUNC_SLOT_WARM",
        strategy_id="alpha",
        strategies_by_slot={"beta": beta_on_bar},
        show_progress=False,
        initial_cash=100000.0,
    )
    if result1.engine is None:
        raise RuntimeError("Engine should not be None")
    aq.save_checkpoint(result1.engine, result1.strategy, str(checkpoint_path))

    result2 = aq.run_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        data=phase2,
        symbols="FUNC_SLOT_WARM",
        show_progress=False,
    )

    alpha_strategy = result2.strategy
    if alpha_strategy is None:
        raise RuntimeError("Primary strategy should not be None")
    slot_strategies = getattr(alpha_strategy, "_slot_strategies", {})
    beta_strategy = slot_strategies.get("beta")
    if beta_strategy is None:
        raise RuntimeError("Beta slot strategy should not be None")

    print("alpha_events=" + "|".join(alpha_strategy.events))
    print("beta_events=" + "|".join(beta_strategy.events))
    print(
        "alpha_processed_closes="
        + ",".join(f"{price:.2f}" for price in alpha_strategy.processed_closes)
    )
    print(
        "beta_processed_closes="
        + ",".join(f"{price:.2f}" for price in beta_strategy.processed_closes)
    )
    print(f"alpha_start_count={alpha_strategy.start_count}")
    print(f"beta_start_count={beta_strategy.start_count}")
    print(f"alpha_resume_count={alpha_strategy.resume_count}")
    print(f"beta_resume_count={beta_strategy.resume_count}")
    print(f"slot_ids={sorted(slot_strategies.keys())}")
    print("done_functional_multi_slot_warm_start_demo")

    checkpoint_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
