"""运行时统计 self.stats / ctx.strategy_stats / engine.strategy_stats 的口径测试."""

from typing import Any

import pandas as pd
from akquant import Bar, Strategy, run_backtest

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


def test_engine_stats_in_on_stop_equal_metrics() -> None:
    """on_stop 里 engine.strategy_stats() 与 result.metrics 同名字段逐位相等."""
    strat = _RoundTrips()
    result = _run(strat)
    m = result.metrics_df["value"]
    got = strat.final_engine_stats
    assert got["trade_count"] > 0, "用例没有产生平仓交易，测试无效"
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
