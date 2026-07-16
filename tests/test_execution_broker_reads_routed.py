"""回归护栏：get_account/get_portfolio_value/get_open_orders 经公共函数路由到 execution.

修复前这三个公共只读函数直接读 strategy.ctx；broker_live 下 ctx 恒为 None，
导致 get_account 抛 RuntimeError、get_portfolio_value 返回 0.0、
get_open_orders 返回 []，而不是柜台侧真实数据（BrokerExecution 已正确实现，
只是从未被这三个公共函数调用到）。本文件用公共自由函数
`akquant.strategy_trading_api.get_account/get_portfolio_value/get_open_orders`
断言它们确实路由到了 strategy.execution（此处绑定为 BrokerExecution）。
"""

from typing import Any

from akquant import strategy_trading_api as api
from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.broker_models import (
    UnifiedAccount,
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedPosition,
)
from akquant.gateway.broker_state_cache import BrokerStateCache


class _Gw:
    """假柜台网关：持仓/资金/未完成委托均来自柜台侧真相."""

    def query_positions(self) -> list[UnifiedPosition]:
        return [
            UnifiedPosition(symbol="600000.SH", quantity=1000, available_quantity=800)
        ]

    def query_account(self) -> UnifiedAccount:
        return UnifiedAccount(
            account_id="a", equity=1500.0, cash=1000.0, available_cash=850.0
        )

    def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
        return [
            UnifiedOrderSnapshot(
                client_order_id="c1",
                broker_order_id="b1",
                symbol="600000.SH",
                status=UnifiedOrderStatus.NEW,
            ),
            UnifiedOrderSnapshot(
                client_order_id="c2",
                broker_order_id="b2",
                symbol="000001.SZ",
                status=UnifiedOrderStatus.NEW,
            ),
        ]

    def cancel_order(self, bid: str) -> None:
        return None


class _Submitter:
    def submit_order(self, **kwargs: Any) -> str:
        return "BID-1"

    def _get_execution_capabilities(self) -> dict[str, bool]:
        return {"broker_live": True, "client_order_id": True}


class _BrokerLiveStrategy:
    """最小 broker_live 策略替身：ctx 恒为 None，读全部经 execution."""

    ctx = None
    current_bar = None
    current_tick = None

    def __init__(self) -> None:
        gw = _Gw()
        self.execution = BrokerExecution(self, gw, BrokerStateCache(gw), _Submitter())


def test_get_account_routes_to_broker_execution_not_raise() -> None:
    """broker_live 下 get_account 应返回柜台账户字典，而非抛 RuntimeError."""
    strategy = _BrokerLiveStrategy()
    account = api.get_account(strategy)
    assert account["cash"] == 1000.0
    assert account["equity"] == 1500.0


def test_get_portfolio_value_routes_to_broker_execution_not_zero() -> None:
    """broker_live 下 get_portfolio_value 应返回柜台权益，而非 0.0."""
    strategy = _BrokerLiveStrategy()
    assert api.get_portfolio_value(strategy) == 1500.0


def test_get_open_orders_routes_to_broker_execution_not_empty() -> None:
    """broker_live 下 get_open_orders 应返回柜台未完成委托，而非 []."""
    strategy = _BrokerLiveStrategy()
    orders = api.get_open_orders(strategy)
    assert len(orders) == 2
    symbols = {o.symbol for o in orders}
    assert symbols == {"600000.SH", "000001.SZ"}


def test_get_open_orders_symbol_filter_routes_to_broker_execution() -> None:
    """broker_live 下 get_open_orders(symbol=...) 应按 symbol 过滤柜台委托."""
    strategy = _BrokerLiveStrategy()
    orders = api.get_open_orders(strategy, symbol="600000.SH")
    assert len(orders) == 1
    assert orders[0].symbol == "600000.SH"


def test_backtest_reads_still_go_through_sim_via_ctx() -> None:
    """回测(SimExecution)下这三个函数仍应经 ctx 读取，行为与重构前一致."""

    class _Ctx:
        account_equity = 999.0
        cash = 500.0
        active_orders: list[Any] = []
        canceled_order_ids: list[str] = []
        account_market_value = 0.0
        account_notional_value = 0.0
        account_used_margin = 0.0
        account_unrealized_pnl = 0.0
        account_maintenance_ratio = 0.0
        account_frozen_cash = 0.0
        account_short_market_value = 0.0
        margin_accrued_interest = 0.0
        margin_daily_interest = 0.0
        risk_config = None

        def get_position(self, symbol: str) -> float:
            return 0.0

    class _SimStrategy:
        current_bar = None
        current_tick = None
        equity = 999.0

        def __init__(self) -> None:
            self.ctx = _Ctx()
            from akquant.execution.sim import SimExecution

            self.execution = SimExecution(self)

    strategy = _SimStrategy()
    assert api.get_portfolio_value(strategy) == 999.0
    assert api.get_account(strategy)["cash"] == 500.0
    assert api.get_open_orders(strategy) == []
