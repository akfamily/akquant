"""运行时统计 self.stats / ctx.strategy_stats / engine.strategy_stats 的口径测试."""

from typing import Any

import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest
from akquant.strategy_stats import DEFAULT_ROW_FIELDS, StrategyStats

STATS_KEYS = {
    "total_return_pct",
    "total_pnl",
    "unrealized_pnl",
    "total_commission",
    "max_drawdown",
    "max_drawdown_pct",
    "max_drawdown_value",
    "win_rate",
    "loss_rate",
    "winning_trades",
    "losing_trades",
    "profit_factor",
    "avg_pnl",
    "avg_profit",
    "avg_loss",
    "largest_win",
    "largest_loss",
    "max_wins",
    "max_losses",
    "trade_count",
    "current_drawdown_pct",
    "equity",
}


def _bars(closes: list[float], symbol: str = "AAA") -> list[Bar]:
    return [
        Bar(
            timestamp=pd.Timestamp("2024-01-02 15:00:00", tz="Asia/Shanghai").value
            + i * 86_400_000_000_000,
            open=c,
            high=c + 0.5,
            low=c - 0.5,
            close=c,
            volume=1e6,
            symbol=symbol,
        )
        for i, c in enumerate(closes)
    ]


class _RoundTrips(Strategy):
    """每 2 根 bar 买一次、再过 2 根卖出，制造有赢有亏的平仓交易."""

    def __init__(self) -> None:
        super().__init__()
        self.ctx_snapshots: list[dict[str, Any]] = []
        self.final_engine_stats: dict[str, Any] = {}
        self._n = 0

    def on_bar(self, bar: Bar) -> None:
        self._n += 1
        assert self.ctx is not None
        self.ctx_snapshots.append(dict(self.ctx.strategy_stats()))
        if self._n % 4 == 1:
            self.buy(bar.symbol, 100)
        elif self._n % 4 == 3:
            self.sell(bar.symbol, 100)

    def on_stop(self) -> None:
        # run() 返回后由 run_backtest 挂上的 Engine（Strategy 未声明该属性）
        engine = getattr(self, "_engine")
        self.final_engine_stats = dict(engine.strategy_stats())


# 末尾 3 根让最后一笔买入成交后不再卖出成交，期末留有持仓，
# unrealized_pnl 的比较才不是 0 == 0。
CLOSES: list[float] = [10, 11, 12, 11, 10, 9, 8, 9, 10, 12, 13, 11, 10, 9, 11, 12]
CLOSES += [13, 14, 15]


def _run(strat: _RoundTrips) -> Any:
    return run_backtest(
        data=_bars(CLOSES),
        strategy=strat,
        initial_cash=100_000,
        commission_rate=0.0003,
        show_progress=False,
    )


def test_ctx_strategy_stats_has_all_keys() -> None:
    """回调期间 ctx.strategy_stats() 返回全部字段."""
    strat = _RoundTrips()
    _run(strat)
    assert strat.ctx_snapshots, "on_bar 没被调用"
    assert set(strat.ctx_snapshots[-1]) == STATS_KEYS
    assert strat.ctx_snapshots[-1]["trade_count"] > 0
    assert strat.ctx_snapshots[-1]["max_drawdown"] > 0


def test_engine_stats_in_on_stop_equal_metrics() -> None:
    """on_stop 里 engine.strategy_stats() 与 result.metrics 同名字段逐位相等."""
    strat = _RoundTrips()
    result = _run(strat)
    m = result.metrics_df["value"]
    got = strat.final_engine_stats
    assert got["trade_count"] > 0, "用例没有产生平仓交易，测试无效"
    assert got["max_drawdown"] > 0, "用例没有回撤，测试无效"
    assert got["unrealized_pnl"] != 0, "期末无持仓，unrealized_pnl 比较无效"
    for key in STATS_KEYS - {"trade_count", "current_drawdown_pct", "equity"}:
        assert got[key] == float(m[key]), key
    assert got["equity"] == float(m["end_market_value"])


def test_ctx_max_drawdown_is_monotonic_and_bounded_by_final() -> None:
    """运行中最大回撤单调不减，且不超过最终 metrics."""
    strat = _RoundTrips()
    result = _run(strat)
    series = [s["max_drawdown"] for s in strat.ctx_snapshots]
    assert series == sorted(series), "最大回撤在运行中不能变小"
    assert series[-1] <= float(result.metrics_df["value"]["max_drawdown"])


class _StatsProperty(_RoundTrips):
    def __init__(self) -> None:
        super().__init__()
        self.on_bar_stats: list[StrategyStats] = []
        self.on_stop_stats: StrategyStats | None = None

    def on_bar(self, bar: Bar) -> None:
        self.on_bar_stats.append(self.stats)
        super().on_bar(bar)

    def on_stop(self) -> None:
        self.on_stop_stats = self.stats


def test_stats_property_in_on_stop_equals_metrics() -> None:
    """on_stop 里的 self.stats 与 result.metrics 逐位相等."""
    strat = _StatsProperty()
    result = run_backtest(
        data=_bars(CLOSES),
        strategy=strat,
        initial_cash=100_000,
        commission_rate=0.0003,
        show_progress=False,
    )
    m = result.metrics_df["value"]
    s = strat.on_stop_stats
    assert s is not None
    assert s.trade_count is not None and s.trade_count > 0
    assert s.max_drawdown is not None and s.max_drawdown > 0
    assert s.win_rate == float(m["win_rate"])
    assert s.max_drawdown == float(m["max_drawdown"])
    assert s.total_pnl == float(m["total_pnl"])
    assert s.profit_factor == float(m["profit_factor"])
    # 期末有持仓：ctx 运行期口径此时为 0，只有 on_stop 的引擎口径才等于 metrics
    assert s.unrealized_pnl != 0
    assert s.unrealized_pnl == float(m["unrealized_pnl"])
    assert isinstance(s.trade_count, int)


def test_stats_property_mid_run_uses_ctx() -> None:
    """回调期间 self.stats 取自 ctx，交易数随运行增加."""
    strat = _StatsProperty()
    run_backtest(
        data=_bars(CLOSES),
        strategy=strat,
        initial_cash=100_000,
        commission_rate=0.0003,
        show_progress=False,
    )
    assert strat.on_bar_stats[0].trade_count == 0
    last = strat.on_bar_stats[-1].trade_count
    assert last is not None and last >= 1


def test_as_rows_default_fields_and_formatting() -> None:
    """as_rows 默认字段、中文标签与显示文本."""
    s = StrategyStats.from_mapping(
        {k: 0.0 for k in STATS_KEYS}
        | {
            "total_return_pct": 12.345,
            "win_rate": 50.0,
            "profit_factor": float("inf"),
            "trade_count": 4,
            "winning_trades": 2,
            "losing_trades": 2,
            "max_wins": 1,
            "max_losses": 1,
        }
    )
    rows = s.as_rows()
    assert [r[0]["value"] for r in rows] == [
        {
            "total_return_pct": "累计收益",
            "max_drawdown_pct": "最大回撤",
            "win_rate": "胜率",
            "profit_factor": "盈亏比",
            "trade_count": "交易次数",
            "total_pnl": "总盈亏",
        }[f]
        for f in DEFAULT_ROW_FIELDS
    ]
    by_field = dict(zip(DEFAULT_ROW_FIELDS, rows))
    assert by_field["total_return_pct"][1]["text"] == "12.35%"
    assert by_field["profit_factor"][1]["text"] == "∞"
    assert by_field["trade_count"][1]["text"] == "4"


def test_as_rows_rejects_unknown_field() -> None:
    """未知字段名抛 ValueError."""
    s = StrategyStats.from_mapping({k: 0.0 for k in STATS_KEYS})
    with pytest.raises(ValueError, match="nope"):
        s.as_rows(fields=["nope"])


def test_stats_outside_callback_raises() -> None:
    """不在回调里读取 stats 抛 RuntimeError."""
    with pytest.raises(RuntimeError):
        _ = Strategy().stats
