# -*- coding: utf-8 -*-
"""last_prices 时间片快照（last_prices_snapshot_per_timestamp）语义与等价性."""

import akquant as aq
import pandas as pd
from akquant import Strategy

TZ = "Asia/Shanghai"
DAYS = pd.date_range("2024-01-02", periods=4, freq="B", tz=TZ)


def _make_bars(symbol: str, closes: list[float]) -> pd.DataFrame:
    rows = []
    for d, c in zip(DAYS, closes):
        rows.append(
            {
                "date": d,
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "volume": 1_000_000.0,
                "symbol": symbol,
            }
        )
    return pd.DataFrame(rows)


class _PriceProbe(Strategy):
    """每 bar 记录 ctx.last_prices 中本标的价格（缺省 -1.0 表示快照中不存在）."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[float] = []

    def on_bar(self, bar: aq.Bar) -> None:
        self.seen.append(self.ctx.last_prices.get(bar.symbol, -1.0))


def _run_probe(flag: bool) -> list[float]:
    strat = _PriceProbe()
    aq.run_backtest(
        data={"AAA": _make_bars("AAA", [10.0, 11.0, 12.0, 13.0])},
        strategy=strat,
        symbols=["AAA"],
        initial_cash=100_000.0,
        commission_rate=0.0,
        show_progress=False,
        last_prices_snapshot_per_timestamp=flag,
    )
    return strat.seen


def test_snapshot_off_sees_same_bar_price() -> None:
    """默认（关）：context 逐 bar 快照，on_bar 看到当根 bar 价格."""
    assert _run_probe(flag=False) == [10.0, 11.0, 12.0, 13.0]


def test_snapshot_on_sees_previous_timestamp_price() -> None:
    """开启：context 用上一时间片快照；第一时间片快照尚未发布（读不到）."""
    assert _run_probe(flag=True) == [-1.0, 10.0, 11.0, 12.0]


class _DailyRotate(Strategy):
    """cross-section 每日轮换：奇数日满仓 AAA、偶数日满仓 BBB（next-open 成交）."""

    def __init__(self) -> None:
        super().__init__()
        self._day = 0

    def on_cross_section(self, trading_date: object, timestamp: int) -> None:
        _ = trading_date, timestamp
        self._day += 1
        pick = "AAA" if self._day % 2 == 1 else "BBB"
        other = "BBB" if pick == "AAA" else "AAA"
        self.order_target_percent(target_percent=0.0, symbol=other)
        self.order_target_percent(target_percent=0.9, symbol=pick)


def _run_rotation(flag: bool) -> pd.DataFrame:
    data = {
        "AAA": _make_bars("AAA", [10.0, 11.0, 12.0, 13.0]),
        "BBB": _make_bars("BBB", [20.0, 19.0, 21.0, 18.0]),
    }
    result = aq.run_backtest(
        data=data,
        strategy=_DailyRotate(),
        symbols=["AAA", "BBB"],
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        show_progress=False,
        last_prices_snapshot_per_timestamp=flag,
    )
    return result.trades_df.reset_index(drop=True)


def test_snapshot_mode_trades_identical_for_cross_section() -> None:
    """cross-section 负载下开/关快照成交逐笔一致（Timer 事件仍用即时快照）."""
    trades_off = _run_rotation(flag=False)
    trades_on = _run_rotation(flag=True)
    assert len(trades_off) > 0
    pd.testing.assert_frame_equal(trades_off, trades_on)
