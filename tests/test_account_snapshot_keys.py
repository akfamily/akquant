"""get_account() 的键与口径: 回测与实盘共用 AccountSnapshot 一份定义.

账户满足恒等式 ``equity == margin + frozen_cash + available_cash``:
总权益 = 持仓占用 + 挂单冻结 + 可用资金。
"""

from __future__ import annotations

from typing import Any, List

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
from akquant.execution.base import AccountSnapshot
from akquant.gateway.broker_models import UnifiedAccount
from akquant.gateway.broker_strategy_api import _account_to_dict

KEYS = set(AccountSnapshot.__required_keys__)


class _Scripted(Strategy):
    """每根 bar 先记一次账户, 再执行脚本里的一步."""

    steps: List[Any] = []

    def __init__(self) -> None:
        self.accounts: List[dict[str, Any]] = []

    def on_bar(self, bar: Bar) -> None:
        self.accounts.append(dict(self.get_account()))
        index = len(self.accounts) - 1
        if index < len(self.steps) and self.steps[index] is not None:
            side, quantity, price = self.steps[index]
            order = self.buy if side == "buy" else self.sell
            order(quantity=quantity, price=price)


def _run_stock(steps: List[Any]) -> List[dict[str, Any]]:
    index = pd.date_range("2026-01-05", periods=len(steps) + 1, freq="B")
    data = pd.DataFrame(
        {
            "date": index,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e7,
            "symbol": "X",
        }
    )
    strategy = _Scripted()
    strategy.steps = steps
    run_backtest(
        data=data,
        strategy=strategy,
        symbols="X",
        initial_cash=100000.0,
        commission_rate=0.0,
        show_progress=False,
    )
    return strategy.accounts


def _run_futures(steps: List[Any]) -> List[dict[str, Any]]:
    start = pd.Timestamp("2023-05-16 21:00:00", tz="Asia/Shanghai")
    bars = [
        Bar(
            timestamp=(start + pd.Timedelta(minutes=10 * i)).value,
            open=3000.0,
            high=3000.0,
            low=3000.0,
            close=3000.0,
            volume=1e4,
            symbol="RB2305",
        )
        for i in range(len(steps) + 1)
    ]
    strategy = _Scripted()
    strategy.steps = steps
    run_backtest(
        data=bars,
        strategy=strategy,
        symbols="RB2305",
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
                    account_mode="margin",
                    check_cash=True,
                    initial_margin_ratio=0.1,
                    maintenance_margin_ratio=0.3,
                    enable_short_sell=True,
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
                ]
            ),
        ),
    )
    return strategy.accounts


def test_snapshot_has_one_free_figure_and_no_duplicates() -> None:
    """可用资金只有 available_cash 一项; used_margin、free_margin 已删."""
    assert not AccountSnapshot.__optional_keys__
    assert {"equity", "margin", "frozen_cash", "available_cash", "cash"} <= KEYS
    assert not {"free_margin", "used_margin"} & KEYS


def test_backtest_and_broker_accounts_have_the_same_keys() -> None:
    """回测与实盘的 get_account() 键完全一致."""
    backtest = _run_stock([None])[0]
    broker = _account_to_dict(UnifiedAccount("a", 1.0, 1.0, 1.0))
    assert set(backtest) == KEYS
    assert set(broker) == KEYS


def test_stock_available_cash_is_cash_minus_pending_buys() -> None:
    """股票现金账户: 可用资金 = 现金 - 买单按委托价冻结的资金; 挂卖单不改变它."""
    accounts = _run_stock(
        [("buy", 1000, None), ("buy", 2000, 9.0), ("sell", 500, 12.0), None]
    )
    held, with_buy, with_sell = accounts[1], accounts[2], accounts[3]

    assert (held["cash"], held["frozen_cash"], held["available_cash"]) == (
        90000.0,
        0.0,
        90000.0,
    )
    assert (with_buy["frozen_cash"], with_buy["available_cash"]) == (18000.0, 72000.0)
    assert (with_sell["frozen_cash"], with_sell["available_cash"]) == (18000.0, 72000.0)


def test_futures_available_cash_is_equity_minus_margin_and_frozen() -> None:
    """期货保证金账户: 可用资金 = 权益 - 持仓占用保证金 - 挂单按委托价冻结的保证金."""
    accounts = _run_futures([("buy", 5, None), ("buy", 10, 2900.0), None])
    held, with_order = accounts[1], accounts[2]

    assert (held["equity"], held["margin"], held["available_cash"]) == (
        500000.0,
        15000.0,
        485000.0,
    )
    # 10 手 * 2900 * 乘数 10 * 保证金率 0.1
    assert (with_order["frozen_cash"], with_order["available_cash"]) == (
        29000.0,
        456000.0,
    )


@pytest.mark.parametrize("runner", [_run_stock, _run_futures])
def test_equity_is_margin_plus_frozen_plus_available(runner: Any) -> None:
    """恒等式: 总权益 = 持仓占用 + 挂单冻结 + 可用资金."""
    price = 9.0 if runner is _run_stock else 2900.0
    for account in runner([("buy", 5, None), ("buy", 10, price), None]):
        assert account["equity"] == pytest.approx(
            account["margin"] + account["frozen_cash"] + account["available_cash"]
        )
