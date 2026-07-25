# -*- coding: utf-8 -*-
"""skip_inert_bar_hooks：空 on_bar 策略跳过每 bar Python 回调的语义与校验."""

import akquant as aq
import pandas as pd
import pytest
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


class _DailyRotate(Strategy):
    """cross-section 每日满仓轮换（无 on_bar，属 inert 策略）."""

    def __init__(self) -> None:
        super().__init__()
        self._day = 0
        self.cross_section_calls = 0
        self.trade_events = 0

    def on_cross_section(self, trading_date: object, timestamp: int) -> None:
        _ = trading_date, timestamp
        self._day += 1
        self.cross_section_calls += 1
        pick = "AAA" if self._day % 2 == 1 else "BBB"
        other = "BBB" if pick == "AAA" else "AAA"
        self.order_target_percent(target_percent=0.0, symbol=other)
        self.order_target_percent(target_percent=0.9, symbol=pick)

    def on_trade(self, trade: object) -> None:
        _ = trade
        self.trade_events += 1


def _run_rotation(skip: bool) -> tuple[pd.DataFrame, _DailyRotate]:
    data = {
        "AAA": _make_bars("AAA", [10.0, 11.0, 12.0, 13.0]),
        "BBB": _make_bars("BBB", [20.0, 19.0, 21.0, 18.0]),
    }
    strat = _DailyRotate()
    result = aq.run_backtest(
        data=data,
        strategy=strat,
        symbols=["AAA", "BBB"],
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        show_progress=False,
        skip_inert_bar_hooks=skip,
    )
    return result.trades_df.reset_index(drop=True), strat


def test_skip_inert_hooks_trades_identical_and_cross_section_alive() -> None:
    """inert 策略开启跳过：on_cross_section 与成交回调照常，成交逐笔一致."""
    trades_off, strat_off = _run_rotation(skip=False)
    trades_on, strat_on = _run_rotation(skip=True)
    assert len(trades_off) > 0
    pd.testing.assert_frame_equal(trades_off, trades_on)
    # on_cross_section（timer 驱动）不受影响
    assert strat_on.cross_section_calls == strat_off.cross_section_calls > 0
    # on_trade 仍按 bar 送达（_flush_pending_order_events 保留）
    assert strat_on.trade_events == strat_off.trade_events > 0


class _WithOnBar(Strategy):
    def on_bar(self, bar: aq.Bar) -> None:
        _ = bar


def test_skip_inert_hooks_rejects_overridden_on_bar() -> None:
    """覆写了 on_bar 的策略开启跳过必须报错."""
    with pytest.raises(ValueError, match="on_bar"):
        aq.run_backtest(
            data={"AAA": _make_bars("AAA", [10.0, 11.0, 12.0, 13.0])},
            strategy=_WithOnBar(),
            symbols=["AAA"],
            initial_cash=100_000.0,
            show_progress=False,
            skip_inert_bar_hooks=True,
        )


class _WithWarmup(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.warmup_period = 2


def test_skip_inert_hooks_rejects_warmup() -> None:
    """设置 warmup_period 的策略开启跳过必须报错."""
    with pytest.raises(ValueError, match="warmup_period"):
        aq.run_backtest(
            data={"AAA": _make_bars("AAA", [10.0, 11.0, 12.0, 13.0])},
            strategy=_WithWarmup(),
            symbols=["AAA"],
            initial_cash=100_000.0,
            show_progress=False,
            skip_inert_bar_hooks=True,
        )
