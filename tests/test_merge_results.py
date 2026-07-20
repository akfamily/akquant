"""Tests for merge_results / MergedResult (issue #282)."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from akquant import (
    Bar,
    Strategy,
    merge_results,
    run_backtest,
    run_from_checkpoint,
    save_checkpoint,
)
from akquant.backtest import MergedResult


def _make_bars(
    start: str, periods: int, symbol: str = "MRG", start_price: float = 100.0
) -> list[Bar]:
    """Deterministic daily bars."""
    bars: list[Bar] = []
    idx = pd.date_range(start=start, periods=periods, freq="D")
    for i, ts in enumerate(idx):
        price = start_price + float(i)
        bars.append(
            Bar(
                timestamp=ts.value,
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price + 0.5,
                volume=1000.0,
                symbol=symbol,
            )
        )
    return bars


class _AltStrategy(Strategy):
    """Buy on bar 0, sell on bar 2 within each phase; flat by the tail bars.

    Trading only on the first bars leaves no pending order at the checkpoint
    boundary (which pickles the strategy), while still producing fills/trades
    and equity movement for the merge to combine.
    """

    def on_bar(self, bar: Bar) -> None:
        idx = getattr(self, "_mrg_idx", 0)
        setattr(self, "_mrg_idx", idx + 1)
        if idx == 0:
            self.buy(symbol=bar.symbol, quantity=1.0)
        elif idx == 2:
            self.sell(symbol=bar.symbol, quantity=1.0)


def _two_phase_merge(tmp_path: Path) -> tuple[Any, Any]:
    """Run phase1 -> checkpoint -> phase2, return (merged, full) results."""
    checkpoint = tmp_path / "mrg.pkl"
    phase1 = _make_bars("2023-01-01", 5, start_price=100.0)
    phase2 = _make_bars("2023-01-06", 5, start_price=105.0)
    full_bars = _make_bars("2023-01-01", 10, start_price=100.0)

    r1 = run_backtest(
        data=phase1,
        strategy=_AltStrategy,
        symbols="MRG",
        initial_cash=1_000_000.0,
        show_progress=False,
    )
    save_checkpoint(r1.engine, r1.strategy, str(checkpoint))  # type: ignore[arg-type]
    r2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="MRG",
        show_progress=False,
    )
    merged = merge_results(r1, r2)

    full = run_backtest(
        data=full_bars,
        strategy=_AltStrategy,
        symbols="MRG",
        initial_cash=1_000_000.0,
        show_progress=False,
    )
    return merged, full


def test_merge_results_returns_merged_result(tmp_path: Path) -> None:
    """merge_results returns a MergedResult exposing merged curves."""
    merged, _ = _two_phase_merge(tmp_path)
    assert isinstance(merged, MergedResult)
    assert not merged.equity_curve.empty
    # 合并曲线应横跨两段日期（用 UTC 日期比较，规避 tz 展示偏移）
    idx = pd.DatetimeIndex(merged.equity_curve.index)
    dates = idx.tz_convert("UTC").normalize().unique()
    assert pd.Timestamp("2023-01-01", tz="UTC") in dates  # phase1 首日
    assert pd.Timestamp("2023-01-09", tz="UTC") in dates  # phase2 尾段
    # 合并后覆盖的天数应多于任一单段（拼接生效）
    assert len(dates) >= 9


def test_merge_results_curve_is_monotonic_in_time(tmp_path: Path) -> None:
    """Merged equity curve index is sorted and unique."""
    merged, _ = _two_phase_merge(tmp_path)
    idx = merged.equity_curve.index
    assert idx.is_monotonic_increasing
    assert idx.is_unique


def test_merge_results_metrics_align_with_full_backtest(tmp_path: Path) -> None:
    """Core metrics recomputed on merged curve match same-period full backtest."""
    merged, full = _two_phase_merge(tmp_path)
    # 端点权益应与完整回测一致（分段续跑保持账户连续）
    m = merged.metrics
    assert m.end_market_value == pytest.approx(
        float(full.metrics.end_market_value), rel=1e-6
    )
    # max_drawdown 为非负比例
    assert m.max_drawdown >= 0.0
    # total_return_pct 与完整回测同期口径一致
    assert m.total_return_pct == pytest.approx(
        float(full.metrics.total_return_pct), rel=1e-4, abs=1e-6
    )


def test_merge_results_single_result_is_identity(tmp_path: Path) -> None:
    """merge_results(one) reproduces that result's curve views."""
    checkpoint = tmp_path / "single.pkl"
    bars = _make_bars("2023-01-01", 5)
    r = run_backtest(
        data=bars,
        strategy=_AltStrategy,
        symbols="MRG",
        initial_cash=1_000_000.0,
        show_progress=False,
    )
    save_checkpoint(r.engine, r.strategy, str(checkpoint))  # type: ignore[arg-type]
    merged = merge_results(r)
    assert len(merged.equity_curve) == len(r.equity_curve)
    assert merged.equity_curve.iloc[-1] == pytest.approx(float(r.equity_curve.iloc[-1]))


def test_merge_results_rejects_overlapping_segments(tmp_path: Path) -> None:
    """Overlapping time segments raise ValueError."""
    bars_a = _make_bars("2023-01-01", 5)
    bars_b = _make_bars("2023-01-03", 5)  # overlaps a
    ra = run_backtest(
        data=bars_a,
        strategy=_AltStrategy,
        symbols="MRG",
        initial_cash=1_000_000.0,
        show_progress=False,
    )
    rb = run_backtest(
        data=bars_b,
        strategy=_AltStrategy,
        symbols="MRG",
        initial_cash=1_000_000.0,
        show_progress=False,
    )
    with pytest.raises(ValueError, match="overlap"):
        merge_results(ra, rb)


def test_merge_results_requires_at_least_one(tmp_path: Path) -> None:
    """merge_results with no arguments raises ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        merge_results()


def test_merge_results_dedupes_boundary_timestamp(tmp_path: Path) -> None:
    """Shared boundary timestamps collapse to one row when dedupe_boundary=True."""
    merged, _ = _two_phase_merge(tmp_path)
    # equity 曲线索引唯一（边界去重生效）
    assert merged.equity_curve.index.is_unique
    # orders 主键 id 无重复
    if not merged.orders_df.empty and "id" in merged.orders_df.columns:
        assert merged.orders_df["id"].is_unique


def test_merge_results_metrics_df_has_core_fields(tmp_path: Path) -> None:
    """metrics_df exposes the core metric subset."""
    merged, _ = _two_phase_merge(tmp_path)
    df = merged.metrics_df
    for field in [
        "total_return_pct",
        "max_drawdown",
        "sharpe_ratio",
        "end_market_value",
    ]:
        assert field in df.index


def test_merge_results_out_of_scope_metric_raises(tmp_path: Path) -> None:
    """Accessing a non-core metric on merged result raises AttributeError."""
    merged, _ = _two_phase_merge(tmp_path)
    with pytest.raises(AttributeError, match="核心指标子集"):
        _ = merged.metrics.ulcer_index
