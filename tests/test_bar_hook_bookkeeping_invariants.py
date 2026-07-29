# -*- coding: utf-8 -*-
"""由 _on_bar_event 维护、而 timer 驱动策略仍依赖的状态,必须保持送达.

两项不变量都用「只覆写 on_cross_section、不覆写 on_bar」的策略验证——即
issue #347 判定为"惰性"的那一类. 该 issue 提议对这类策略跳过每 bar 的
Python 回调; 实测这会静默丢失 on_expiry 并让 get_holding_bars 恒返 0.
这两个测试锁住这两条不变量, 使任何此类改动直接变红而非静默产出错数.
"""

from datetime import date, datetime, timezone
from typing import Any, Dict, List

import akquant as aq

SYMBOL = "FUT_EXP_CS"

_DAYS = [
    datetime(2026, 1, 28, 15, 0, tzinfo=timezone.utc),
    datetime(2026, 1, 29, 15, 0, tzinfo=timezone.utc),
    datetime(2026, 1, 30, 15, 0, tzinfo=timezone.utc),
    datetime(2026, 1, 31, 15, 0, tzinfo=timezone.utc),
    datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc),
]
_PRICES = [100.0, 101.0, 102.0, 110.0, 109.0]


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)


def _bars() -> List[aq.Bar]:
    return [aq.Bar(_ns(d), p, p, p, p, 1000.0, SYMBOL) for d, p in zip(_DAYS, _PRICES)]


def _config() -> aq.BacktestConfig:
    return aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(),
        instruments_config=[
            aq.InstrumentConfig(
                symbol=SYMBOL,
                asset_type="FUTURES",
                multiplier=10.0,
                margin_ratio=0.1,
                tick_size=0.2,
                expiry_date=date(2026, 1, 31),
                settlement_type="settlement_price",
                settlement_price=108.0,
            )
        ],
    )


class _CrossSectionOnly(aq.Strategy):
    """只覆写 on_cross_section 与 on_expiry, 不覆写 on_bar."""

    def __init__(self) -> None:
        super().__init__()
        self.expiry_events: List[Dict[str, Any]] = []
        self.holding_bars_seen: List[int] = []
        self.cs_calls = 0

    def on_cross_section(self, trading_date: object, timestamp: int) -> None:
        self.cs_calls += 1
        if self.cs_calls == 1:
            self.buy(symbol=SYMBOL, quantity=1)
        self.holding_bars_seen.append(self.get_holding_bars(SYMBOL))

    def on_expiry(self, event: Dict[str, Any]) -> None:
        self.expiry_events.append(dict(event))


def _run() -> _CrossSectionOnly:
    strategy = _CrossSectionOnly()
    _ = aq.run_backtest(
        data=_bars(),
        strategy=strategy,
        symbols=[SYMBOL],
        initial_cash=1_000_000.0,
        config=_config(),
        show_progress=False,
    )
    return strategy


def test_on_expiry_delivered_to_timer_driven_strategy() -> None:
    """on_expiry 的送达不得依赖策略是否覆写 on_bar.

    check_expiry_events 全仓只有两个调用点(strategy_events.py 的 bar 与 tick
    路径), on_timer_event 不调用它. 因此任何"跳过 bar 级回调"的改动都会让
    on_expiry 彻底失联.
    """
    strategy = _run()

    assert strategy.cs_calls == 5
    assert len(strategy.expiry_events) == 1

    event = strategy.expiry_events[0]
    assert event["symbol"] == SYMBOL
    assert float(event["quantity_before"]) == 1.0
    assert float(event["quantity_closed"]) == 1.0
    assert float(event["cash_flow"]) == 1080.0
    assert event["reason"] == "expiry"


def test_get_holding_bars_advances_for_timer_driven_strategy() -> None:
    """get_holding_bars() 必须递进.

    其唯一数据源是 _hold_bars, 仅由 strategy_events.on_bar_event 维护.
    跳过 bar 级回调会让它恒返 0——不抛异常, 静默错数.
    """
    strategy = _run()
    assert strategy.holding_bars_seen == [0, 1, 2, 0, 0]
