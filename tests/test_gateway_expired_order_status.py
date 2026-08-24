"""``Expired`` 终态在实盘链路上的贯通性.

回测侧 Rust ``OrderStatus`` 早有 ``Expired``(``strategy_order_events``
的终态集也含它), 实盘侧 ``UnifiedOrderStatus`` 却没有 ⇒ 柜台/中间件报来的
``expired`` 只能落到非终态兜底(``SUBMITTED``), 该单会永远留在挂单表里被
``cancel_all_orders`` 反复撤(柜台回 ``251020 委托状态错误不能撤单``)。
"""

from typing import Any

from akquant.akquant import OrderStatus
from akquant.gateway.broker_event_adapter import map_order_snapshot
from akquant.gateway.broker_models import UnifiedOrderSnapshot, UnifiedOrderStatus
from akquant.gateway.ctp_adapter import CTPTraderAdapter
from akquant.gateway.mapper import create_default_mapper
from akquant.gateway.miniqmt import MiniQMTTraderGateway
from akquant.gateway.ptrade import PTradeTraderGateway
from akquant.live._payload_utils import is_terminal_status


def _snapshot(status: UnifiedOrderStatus) -> UnifiedOrderSnapshot:
    return UnifiedOrderSnapshot(
        client_order_id="cli-1",
        broker_order_id="b-1",
        symbol="600000.SH",
        status=status,
    )


def test_unified_order_status_exposes_expired() -> None:
    """``UnifiedOrderStatus`` 要有 ``EXPIRED``, 取值与 Rust 侧同名."""
    assert UnifiedOrderStatus("Expired") is UnifiedOrderStatus.EXPIRED


def test_map_order_snapshot_keeps_expired() -> None:
    """``Expired`` 不能在 Unified -> StrategyOrder 适配里退化成 ``New``.

    ``_to_status`` 的兜底是 ``OrderStatus.New``(非终态), 漏一条映射就等于把
    终态单重新变成活单。
    """
    order = map_order_snapshot(_snapshot(UnifiedOrderStatus.EXPIRED))
    assert order.status == OrderStatus.Expired


def test_is_terminal_status_accepts_expired() -> None:
    """实盘 runner 的终态集要认 ``Expired``(回测侧终态集早已包含它)."""
    assert is_terminal_status(UnifiedOrderStatus.EXPIRED) is True
    assert is_terminal_status("Expired") is True


def test_default_status_map_recognizes_expired_text() -> None:
    """柜台原始状态文本 ``expired`` 要映射到 ``EXPIRED``, 不能落非终态兜底."""
    assert create_default_mapper().map_order_status("expired") == (
        UnifiedOrderStatus.EXPIRED
    )


def test_builtin_brokers_treat_expired_as_terminal() -> None:
    """三个内置 broker 的终态判定要与核心口径一致."""
    for cls in (CTPTraderAdapter, MiniQMTTraderGateway, PTradeTraderGateway):
        adapter: Any = cls.__new__(cls)
        assert adapter._is_terminal_status(UnifiedOrderStatus.EXPIRED) is True
