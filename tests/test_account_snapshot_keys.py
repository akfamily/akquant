"""get_account() 的键: 回测与实盘两个执行后端共用 AccountSnapshot 一份定义."""

from __future__ import annotations

from typing import Any

import pandas as pd
from akquant import Bar, Strategy, run_backtest
from akquant.execution.base import AccountSnapshot
from akquant.gateway.broker_models import UnifiedAccount
from akquant.gateway.broker_strategy_api import _account_to_dict

REQUIRED = set(AccountSnapshot.__required_keys__)
OPTIONAL = set(AccountSnapshot.__optional_keys__)


class _ReadAccount(Strategy):
    def __init__(self) -> None:
        self.account: dict[str, Any] = {}

    def on_bar(self, bar: Bar) -> None:
        self.account = self.get_account()


def test_snapshot_declares_live_only_keys_as_optional() -> None:
    """只有实盘才有的键单列为可选, 其余都是必有键."""
    assert OPTIONAL == {"available_cash"}
    assert {"cash", "equity", "market_value", "free_margin"} <= REQUIRED


def test_backtest_account_has_exactly_the_required_keys() -> None:
    """回测 get_account() 的键恰好是必有键."""
    index = pd.date_range("2026-01-05", periods=2, freq="B")
    data = pd.DataFrame(
        {
            "date": index,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e6,
            "symbol": "X",
        }
    )
    strategy = _ReadAccount()
    run_backtest(data=data, strategy=strategy, symbols="X", show_progress=False)
    assert set(strategy.account) == REQUIRED


def test_broker_account_has_required_keys_plus_live_only_ones() -> None:
    """实盘 get_account() 的键是必有键加实盘专有键."""
    account = _account_to_dict(UnifiedAccount("a", 1.0, 1.0, 1.0))
    assert set(account) == REQUIRED | OPTIONAL
