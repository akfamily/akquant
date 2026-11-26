"""get_account() 的 account_id: 实盘取柜台账户回报里的账号, 回测与模拟盘为空串."""

from __future__ import annotations

from typing import Any, List, Optional, get_type_hints

import akquant as aq
import pandas as pd
from akquant import AssetType, Bar, Instrument, Strategy, run_backtest, run_live
from akquant.execution.base import AccountSnapshot
from akquant.gateway import register_broker, unregister_broker
from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.broker_models import BrokerCapability, UnifiedAccount
from akquant.gateway.broker_state_cache import BrokerStateCache
from akquant.gateway.broker_strategy_api import _account_to_dict


class _AccountLog(Strategy):
    """在 on_start 与每根 bar 上记下 get_account()["account_id"]."""

    def __init__(self) -> None:
        self.ids: List[Any] = []

    def on_start(self) -> None:
        self.ids.append(("start", self.get_account()["account_id"]))

    def on_bar(self, bar: Bar) -> None:
        self.ids.append(("bar", self.get_account()["account_id"]))


def test_account_id_is_a_required_snapshot_key() -> None:
    """AccountSnapshot 带 account_id(str), 仍然没有可选键."""
    assert "account_id" in AccountSnapshot.__required_keys__
    assert get_type_hints(AccountSnapshot)["account_id"] is str
    assert not AccountSnapshot.__optional_keys__


def test_backtest_account_id_is_empty_string() -> None:
    """回测没有柜台: on_start 与 on_bar 里都是空串."""
    index = pd.date_range("2026-01-05", periods=2, freq="B")
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
    result = run_backtest(
        data=data, strategy=_AccountLog, symbols="X", show_progress=False
    )
    assert result.strategy is not None
    assert result.strategy.ids == [("start", ""), ("bar", ""), ("bar", "")]


def _futures(symbol: str) -> Instrument:
    return Instrument(
        symbol=symbol,
        asset_type=AssetType.Futures,
        multiplier=10.0,
        margin_ratio=0.1,
        tick_size=1.0,
        lot_size=1,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _bars(symbol: str) -> List[Bar]:
    index = pd.date_range(
        "2024-01-02 09:31:00", periods=2, freq="1min", tz="Asia/Shanghai"
    )
    return [Bar(int(ts.value), 10.0, 10.0, 10.0, 10.0, 1e6, symbol) for ts in index]


def test_paper_account_id_is_empty_string() -> None:
    """模拟盘(paper)由 AKQuant 撮合, 没有柜台账号: 空串."""
    strategy = _AccountLog()
    aq.run_live(
        strategy_cls=strategy,
        instruments=[_futures("RB2310")],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": _bars("RB2310"), "freq": "1min"},
        show_progress=False,
        duration="60s",
    )
    assert strategy.ids == [("start", ""), ("bar", ""), ("bar", "")]


def test_broker_account_maps_reported_account_id() -> None:
    """柜台账户回报(UnifiedAccount)里的账号原样进入快照; 没有回报时为空串."""
    assert _account_to_dict(UnifiedAccount("FA-001", 1.0, 1.0, 1.0))["account_id"] == (
        "FA-001"
    )
    assert _account_to_dict(None)["account_id"] == ""
    assert _account_to_dict(UnifiedAccount("", 1.0, 1.0, 1.0))["account_id"] == ""


class _AccountGateway:
    """只有交易侧的柜台桩; query_account 按 reports 依次返回(None 表示还没回报)."""

    def __init__(self, reports: List[Optional[UnifiedAccount]]) -> None:
        self.reports = list(reports)

    def connect(self) -> None:
        """no-op."""

    def disconnect(self) -> None:
        """no-op."""

    def start(self) -> None:
        """no-op."""

    def place_order(self, req: Any) -> str:
        """不会被调用."""
        raise AssertionError("no orders expected")

    def cancel_order(self, broker_order_id: str) -> None:
        """no-op."""

    def get_capabilities(self) -> BrokerCapability:
        """最小能力集."""
        return BrokerCapability(
            broker_name="account-fake",
            broker_live=True,
            client_order_id=True,
            order_type=True,
            time_in_force_str=True,
        )

    def query_order(self, broker_order_id: str) -> None:
        """无委托."""
        return None

    def query_trades(self, since: Optional[int] = None) -> List[Any]:
        """无成交."""
        return []

    def query_account(self) -> Optional[UnifiedAccount]:
        """按顺序返回账户回报, 用完后保持最后一个."""
        if len(self.reports) > 1:
            return self.reports.pop(0)
        return self.reports[0]

    def query_positions(self) -> List[Any]:
        """无持仓."""
        return []

    def on_order(self, callback: Any) -> None:
        """不推送."""

    def on_trade(self, callback: Any) -> None:
        """不推送."""

    def on_execution_report(self, callback: Any) -> None:
        """不推送."""

    def sync_open_orders(self) -> List[Any]:
        """无挂单."""
        return []

    def sync_today_trades(self) -> List[Any]:
        """无当日成交."""
        return []

    def heartbeat(self) -> bool:
        """始终就绪."""
        return True


def _run_broker_live(gateway: _AccountGateway) -> _AccountLog:
    from akquant.gateway.protocols import GatewayBundle

    strategy = _AccountLog()
    register_broker(
        "account-fake",
        lambda **_: GatewayBundle(
            market_gateway=None,
            trader_gateway=gateway,
            trader_capabilities=gateway.get_capabilities(),
            metadata={"broker": "account-fake"},
        ),
    )
    try:
        run_live(
            strategy_cls=strategy,
            instruments=[_futures("ACCT_A")],
            market_broker="replay",
            trader_broker="account-fake",
            trading_mode="broker_live",
            gateway_options={"bars": _bars("ACCT_A")},
            show_progress=False,
            duration="60s",
        )
    finally:
        unregister_broker("account-fake")
    return strategy


def test_broker_live_account_id_comes_from_broker_report() -> None:
    """broker_live: get_account()["account_id"] 是柜台回报里的账号."""
    report = UnifiedAccount("FA-001", equity=1e6, cash=1e6, available_cash=1e6)
    strategy = _run_broker_live(_AccountGateway([report]))
    assert strategy.ids
    assert all(account_id == "FA-001" for _, account_id in strategy.ids)


def test_broker_live_account_id_empty_before_first_report() -> None:
    """柜台还没回报账户时为空串, 回报之后才有账号."""
    report = UnifiedAccount("FA-002", equity=1e6, cash=1e6, available_cash=1e6)
    gateway = _AccountGateway([None, report])
    cache = BrokerStateCache(gateway)
    execution = BrokerExecution(object(), gateway, cache, None)
    assert execution.get_account()["account_id"] == ""
    cache.invalidate_account()
    assert execution.get_account()["account_id"] == "FA-002"
