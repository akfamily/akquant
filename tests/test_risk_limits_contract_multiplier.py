# -*- coding: utf-8 -*-
"""按金额设的风控限额在期货上要乘合约乘数.

修复前这些限额一律按 ``价格 × 数量`` 计, 漏了合约乘数: 螺纹钢(乘数 10)一手
4000 的委托名义额是 40000, 却被当成 4000; 浮亏 1000 被当成 100。限额等于被放宽
“乘数”倍, 期权(乘数 10000)上基本失效。股票乘数为 1, 不受影响。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List

import akquant
from akquant import (
    BacktestConfig,
    ChinaFuturesConfig,
    ChinaFuturesInstrumentTemplateConfig,
    CurrentClose,
    StrategyConfig,
)
from akquant.config import RiskConfig

SYMBOL = "RB2305"
MULTIPLIER = 10.0


def _bars() -> List[akquant.Bar]:
    days = [datetime(2023, 1, k, 2, 0, tzinfo=timezone.utc) for k in (3, 4, 5)]
    prices = [4000.0, 3900.0, 3900.0]
    return [
        akquant.Bar(int(t.timestamp() * 1e9), p, p, p, p, 10000.0, SYMBOL)
        for t, p in zip(days, prices)
    ]


class _BuyOneLotTwice(akquant.Strategy):
    """前两根 bar 各买 1 手: 第 2 根时持仓浮亏 (4000-3900)×10 = 1000."""

    def __init__(self) -> None:
        super().__init__()
        self._step = 0

    def on_bar(self, bar: akquant.Bar) -> None:
        if self._step < 2:
            self.buy(symbol=bar.symbol, quantity=1)
        self._step += 1


def _run(risk: RiskConfig | None = None, **kwargs: Any) -> Any:
    result = akquant.run_backtest(
        data=_bars(),
        strategy=_BuyOneLotTwice,
        symbols=SYMBOL,
        show_progress=False,
        sessions={"RB": "ALWAYS_OPEN"},
        fill_policy=CurrentClose(),
        strategy_id="alpha",
        config=BacktestConfig(
            strategy_config=StrategyConfig(
                initial_cash=500000.0,
                commission_rate=0.0,
                min_commission=0.0,
                stamp_tax_rate=0.0,
                transfer_fee_rate=0.0,
                risk=risk or RiskConfig(account_mode="margin"),
            ),
            china_futures=ChinaFuturesConfig(
                instrument_templates_by_symbol_prefix=[
                    ChinaFuturesInstrumentTemplateConfig(
                        symbol_prefix="RB",
                        multiplier=MULTIPLIER,
                        margin_ratio=0.1,
                        commission_rate=0.0,
                        tick_size=1.0,
                        lot_size=1.0,
                    )
                ],
            ),
        ),
        **kwargs,
    )
    orders = result.orders_df.sort_values("created_at")
    return [
        (str(row["status"]).lower(), str(row["reject_reason"] or ""))
        for _, row in orders.iterrows()
    ]


def test_baseline_without_limits_fills_both_orders() -> None:
    """不设限额时两笔都成交(保证下面的拒单确实来自限额)."""
    assert [status for status, _ in _run()] == ["filled", "filled"]


def test_strategy_daily_loss_counts_contract_multiplier() -> None:
    """浮亏 1000 超过策略日亏损上限 500: 第二笔被拒, 报文里是 1000 而不是 100."""
    orders = _run(strategy_max_daily_loss={"alpha": 500.0})
    assert orders[0][0] == "filled"
    assert orders[1][0] == "rejected"
    assert "1000" in orders[1][1]


def test_strategy_daily_loss_not_triggered_below_real_loss() -> None:
    """上限 1500 高于真实浮亏 1000: 不拒单."""
    orders = _run(strategy_max_daily_loss={"alpha": 1500.0})
    assert [status for status, _ in orders] == ["filled", "filled"]


def test_strategy_max_order_value_counts_contract_multiplier() -> None:
    """一手名义额 40000 超过策略单笔上限 10000: 第一笔就被拒."""
    orders = _run(strategy_max_order_value={"alpha": 10000.0})
    assert orders[0][0] == "rejected"
    assert "40000" in orders[0][1]


def test_strategy_max_order_value_allows_order_within_real_notional() -> None:
    """上限 45000 高于名义额 40000: 放行."""
    orders = _run(strategy_max_order_value={"alpha": 45000.0})
    assert orders[0][0] == "filled"


def test_global_max_order_value_counts_contract_multiplier() -> None:
    """账户级 max_order_value 同口径."""
    orders = _run(risk=RiskConfig(account_mode="margin", max_order_value=10000.0))
    assert orders[0][0] == "rejected"
    assert "40000" in orders[0][1]


def test_strategy_risk_budget_counts_contract_multiplier() -> None:
    """风险预算按名义额累计: 预算 50000 只够第一手(40000), 第二手(39000)被拒."""
    orders = _run(strategy_risk_budget={"alpha": 50000.0})
    assert [status for status, _ in orders] == ["filled", "rejected"]
