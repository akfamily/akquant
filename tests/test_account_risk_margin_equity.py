# -*- coding: utf-8 -*-
"""账户级动态风控在保证金账户里按真实权益判定, 触发后放行减仓单.

修复前这些规则用“现金 + 持仓名义市值”当权益。保证金账户开期货仓不扣名义额, 于是:

- 多头: 名义额被加进权益, 真实亏损 30% 也触发不了回撤/日亏损规则;
- 空头: 开仓瞬间权益被减掉名义额, 零盈亏就被判成亏损;
- 平仓后: “权益”回落被当成回撤, 之后的新单全部被拒。

另外规则触发后不分开仓减仓一律拒单, 仓位平不掉。现在只拒会增加风险的订单。
"""

from __future__ import annotations

from typing import Any, List, Tuple

import pandas as pd
import pytest
from akquant import (
    BacktestConfig,
    Bar,
    ChinaFuturesConfig,
    ChinaFuturesInstrumentTemplateConfig,
    CurrentClose,
    Strategy,
    StrategyConfig,
    run_backtest,
)
from akquant.config import RiskConfig

SYMBOL = "RB2305"
Step = Tuple[str, float]


def _bars(prices: List[float]) -> List[Bar]:
    start = pd.Timestamp("2023-05-16 21:00:00", tz="Asia/Shanghai")
    return [
        Bar(
            timestamp=(start + pd.Timedelta(minutes=10 * i)).value,
            open=p,
            high=p,
            low=p,
            close=p,
            volume=10000.0,
            symbol=SYMBOL,
        )
        for i, p in enumerate(prices)
    ]


def _run(
    prices: List[float], script: List[Step], mode: str = "margin", **risk: Any
) -> Tuple[List[Tuple[str, str]], List[Tuple[float, float]]]:
    """按脚本逐根下单; 返回 (各委托的状态与拒单原因, 每根 bar 的可用保证金与买入力)."""
    snapshots: List[Tuple[float, float]] = []

    class _Scripted(Strategy):
        def __init__(self) -> None:
            self.step = 0

        def on_bar(self, bar: Bar) -> None:
            account = self.get_account()
            snapshots.append((float(account["free_margin"]), float(self.buying_power)))
            if self.step < len(script):
                action, quantity = script[self.step]
                if action == "buy":
                    self.buy(bar.symbol, quantity)
                else:
                    self.sell(bar.symbol, quantity)
            self.step += 1

    result = run_backtest(
        data=_bars(prices),
        strategy=_Scripted,
        symbols=SYMBOL,
        show_progress=False,
        sessions={"RB": "ALWAYS_OPEN"},
        fill_policy=CurrentClose(),
        config=BacktestConfig(
            strategy_config=StrategyConfig(
                initial_cash=500000.0,
                commission_rate=0.0,
                min_commission=0.0,
                stamp_tax_rate=0.0,
                transfer_fee_rate=0.0,
                risk=RiskConfig(
                    account_mode=mode,
                    check_cash=True,
                    initial_margin_ratio=0.1,
                    maintenance_margin_ratio=0.3,
                    enable_short_sell=True,
                    **risk,
                ),
            ),
            china_futures=ChinaFuturesConfig(
                instrument_templates_by_symbol_prefix=[
                    ChinaFuturesInstrumentTemplateConfig(
                        symbol_prefix="RB",
                        multiplier=10.0,
                        margin_ratio=0.1,
                        commission_rate=0.0,
                        tick_size=1.0,
                        lot_size=1.0,
                    )
                ],
            ),
        ),
    )
    orders = result.orders_df.sort_values("created_at")
    statuses = [
        (str(row["status"]).lower(), str(row["reject_reason"] or ""))
        for _, row in orders.iterrows()
    ]
    return statuses, snapshots


def _status(orders: List[Tuple[str, str]]) -> List[str]:
    return [status for status, _ in orders]


def test_long_loss_triggers_daily_loss_rule() -> None:
    """多 5 手 4000 → 3000, 真实亏损 5 万即 10%: 超过日亏损 3%, 加仓单被拒."""
    orders, _ = _run([4000, 3000, 3000], [("buy", 5), ("buy", 1)], max_daily_loss=0.03)
    assert _status(orders) == ["filled", "rejected"]
    assert "Daily loss 10.00%" in orders[1][1]


def test_long_loss_triggers_drawdown_rule() -> None:
    """多 5 手 4000 → 1000, 真实亏损 30%: 超过回撤 20%, 加仓单被拒."""
    orders, _ = _run(
        [4000, 1000, 1000], [("buy", 5), ("buy", 1)], max_account_drawdown=0.2
    )
    assert _status(orders) == ["filled", "rejected"]
    assert "Max drawdown 30.00%" in orders[1][1]


@pytest.mark.parametrize(
    "risk",
    [{"max_daily_loss": 0.03}, {"stop_loss_threshold": 0.95}],
    ids=["daily_loss", "stop_loss"],
)
def test_short_with_flat_price_is_not_a_loss(risk: Any) -> None:
    """空 1 手、价格不动, 盈亏为 0: 平仓单与之后的新单都正常成交."""
    orders, _ = _run(
        [4000, 4000, 4000, 4000], [("sell", 1), ("buy", 1), ("sell", 1)], **risk
    )
    assert _status(orders) == ["filled", "filled", "filled"]


def test_flat_round_trip_is_not_a_drawdown() -> None:
    """多 5 手开平一个来回、盈亏为 0: 之后的新单不被当成回撤拒掉."""
    orders, _ = _run(
        [4000, 4000, 4000, 4000],
        [("buy", 5), ("sell", 5), ("buy", 1)],
        max_account_drawdown=0.2,
    )
    assert _status(orders) == ["filled", "filled", "filled"]


@pytest.mark.parametrize("mode", ["margin", "cash"])
@pytest.mark.parametrize(
    "risk",
    [
        {"max_daily_loss": 0.03},
        {"max_account_drawdown": 0.05},
        {"stop_loss_threshold": 0.95},
    ],
    ids=["daily_loss", "drawdown", "stop_loss"],
)
def test_reducing_orders_pass_after_rule_triggers(mode: str, risk: Any) -> None:
    """规则触发后: 加仓单被拒, 减仓单(部分减仓、全部平仓)放行."""
    orders, _ = _run(
        [4000, 3000, 3000, 3000, 3000],
        [("buy", 5), ("buy", 1), ("sell", 2), ("sell", 3)],
        mode=mode,
        **risk,
    )
    assert _status(orders) == ["filled", "rejected", "filled", "filled"]


def test_order_that_flips_position_only_closes() -> None:
    """卖出量超过持仓: 回测把它拆成平仓腿与反手开仓腿, 平仓腿放行, 开仓腿被拒."""
    orders, _ = _run([4000, 3000, 3000], [("buy", 5), ("sell", 6)], max_daily_loss=0.03)
    assert _status(orders) == ["filled", "filled", "rejected"]
    assert "Daily loss" in orders[2][1]


def test_buying_power_tracks_free_margin_for_long_and_short() -> None:
    """持有期货多头或空头时, buying_power 与可用保证金的比例一致.

    修复前多头的 buying_power 把名义额算进去(高于可用保证金), 空头则偏低。
    """
    _, flat = _run([4000, 4000], [])
    _, long_side = _run([4000, 4000], [("buy", 5)])
    _, short_side = _run([4000, 4000], [("sell", 5)])
    base = flat[1][1] / flat[1][0]
    assert long_side[1][0] == pytest.approx(480000.0)
    assert long_side[1][1] / long_side[1][0] == pytest.approx(base)
    assert short_side[1][1] / short_side[1][0] == pytest.approx(base)
