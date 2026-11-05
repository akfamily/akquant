"""self.stats 与 result.metrics 的口径对齐(期货/T+1)与绘图 owner 身份测试.

股票+手续费场景见 test_strategy_stats.py;这里补 spec §11 要求的另外两个场景,
以及"指标脚本与主策略同 drawing_id 互不覆盖"的身份测试。
"""

from typing import Any, Optional

import akquant as aq
import pandas as pd
from akquant import Bar, Indicator, Strategy, run_backtest
from akquant.strategy_stats import StrategyStats

SYMBOL = "AAA"
# 有赢有亏;末尾 3 根让最后一笔买入成交后不再卖出,期末留有持仓。
CLOSES: list[float] = [10, 11, 12, 11, 10, 9, 8, 9, 10, 12, 13, 11, 10, 9, 11, 12]
CLOSES += [13, 14, 15]

# 与 metrics 同名、需逐位相等的字段(trade_count/equity 另行比较)
SHARED = [
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
]


def _bars(closes: list[float], symbol: str = SYMBOL) -> list[Bar]:
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
    """每 4 根 bar 买一次、隔 2 根卖出,期末留一笔多头持仓."""

    def __init__(self) -> None:
        super().__init__()
        self.on_stop_stats: Optional[StrategyStats] = None
        self._n = 0

    def on_bar(self, bar: Bar) -> None:
        self._n += 1
        if self._n % 4 == 1:
            self.buy(bar.symbol, 10)
        elif self._n % 4 == 3:
            self.sell(bar.symbol, 10)

    def on_stop(self) -> None:
        self.on_stop_stats = self.stats


def _assert_parity(strat: _RoundTrips, result: Any) -> None:
    m = result.metrics_df["value"]
    s = strat.on_stop_stats
    assert s is not None
    # 非空转守卫:没有平仓/回撤/持仓浮盈亏时相等比较只是 0 == 0
    assert s.trade_count is not None and s.trade_count > 0
    assert s.max_drawdown is not None and s.max_drawdown > 0
    assert s.unrealized_pnl is not None and s.unrealized_pnl != 0
    assert s.losing_trades is not None and s.losing_trades > 0
    for key in SHARED:
        assert getattr(s, key) == float(m[key]), key
    assert s.equity == float(m["end_market_value"])


def test_futures_stats_equal_metrics() -> None:
    """期货(合约乘数 10)多次往返含亏损、期末有持仓:self.stats == metrics."""
    strat = _RoundTrips()
    config = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=1_000_000.0),
        instruments_config=[
            aq.InstrumentConfig(
                symbol=SYMBOL,
                asset_type="FUTURES",
                multiplier=10.0,
                margin_ratio=0.1,
            )
        ],
    )
    result = run_backtest(
        data=_bars(CLOSES),
        strategy=strat,
        config=config,
        commission_rate=0.0003,
        show_progress=False,
    )
    _assert_parity(strat, result)


def test_stock_t_plus_one_stats_equal_metrics() -> None:
    """股票 T+1:self.stats == metrics."""
    strat = _RoundTrips()
    result = run_backtest(
        data=_bars(CLOSES),
        strategy=strat,
        initial_cash=100_000,
        commission_rate=0.0003,
        t_plus_one=True,
        show_progress=False,
    )
    _assert_parity(strat, result)


class _MainDrawer(Strategy):
    def on_bar(self, bar: Bar) -> None:
        self.draw.label("same", (bar.timestamp, bar.close), "main", symbol=bar.symbol)


class _IndDrawer(Indicator):
    def on_bar(self, bar: Bar) -> None:
        self.draw.label("same", (bar.timestamp, bar.close), "ind", symbol=bar.symbol)


def test_indicator_and_strategy_same_drawing_id_keep_own_owner() -> None:
    """同标的同 drawing_id:指标脚本与主策略各留一个对象,互不覆盖."""
    result = run_backtest(
        data=_bars(CLOSES[:5]),
        strategy=_MainDrawer,
        indicators=[_IndDrawer],
        symbols=[SYMBOL],
        initial_cash=1e5,
        strategy_id="main",
        show_progress=False,
    )
    drawings = [d for d in result.drawings if d["drawing_id"] == "same"]
    assert len(drawings) == 2
    by_owner = {d["owner_strategy_id"]: d for d in drawings}
    assert set(by_owner) == {"main", "_ind_drawer"}
    assert by_owner["main"]["text"] == "main"
    assert by_owner["_ind_drawer"]["text"] == "ind"
