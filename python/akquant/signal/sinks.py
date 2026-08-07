"""两个下单出口: paper 走引擎注入, broker_live 走柜台报单.

两者的差异必须在此收敛, 因为它们的能力边界不同(见 signal-ingestion-rfc.md 4.1/4.2):

- **paper**: 经 ``SignalPort`` 发 ``Event::OrderRequest``, 由 ``ChannelProcessor``
  执行**完整**风控(含 daily_loss / drawdown / risk_budget), 再由模拟撮合器成交。
- **broker_live**: 经 ``BrokerOrderSubmitter`` 直连柜台, 只有**策略级三项**限额
  前置生效(order_value / order_size / position_size)。引擎的实盘执行器不报柜台,
  故这条路不能走引擎注入。
"""

from __future__ import annotations

from typing import Any

from ..log import build_log_extra, get_logger
from .models import Signal

logger = get_logger("signal.sink")


class PaperOrderSink:
    """paper 模式出口: 把信号注入引擎事件通道."""

    def __init__(self, port: Any) -> None:
        """持有 ``SignalPort``(其 submit 线程安全)."""
        self._port = port

    @property
    def mode(self) -> str:
        """出口模式标识."""
        return "paper"

    def submit(self, signal: Signal) -> str:
        """注入委托, 返回引擎侧本地订单 id.

        ``tag`` 里写入 ``signal_id``, 使异步拒单/成交能反查回原始信号。
        """
        return str(
            self._port.submit(
                symbol=signal.symbol,
                side=signal.side,
                quantity=signal.quantity,
                price=signal.price,
                order_type=signal.order_type,
                tag=signal.signal_id,
            )
        )


class BrokerOrderSink:
    """broker_live 模式出口: 经 submitter 报到柜台."""

    def __init__(self, submitter: Any) -> None:
        """持有 ``BrokerOrderSubmitter``."""
        self._submitter = submitter

    @property
    def mode(self) -> str:
        """出口模式标识."""
        return "broker_live"

    def submit(self, signal: Signal) -> str:
        """报单并返回主腿 id; 被前置风控拒绝时返回空串.

        ``BrokerOrderSubmitter`` 在被风控拦下时返回空回执
        (``order_ids=()``), 这是同步可知的拒单信号。
        """
        receipt = self._submitter.submit_order(
            symbol=signal.symbol,
            side=signal.side,
            quantity=signal.quantity,
            price=signal.price,
            order_type=signal.order_type or "Market",
            tag=signal.signal_id,
        )
        order_ids = tuple(getattr(receipt, "order_ids", ()) or ())
        if not order_ids:
            logger.warning(
                "信号 %s 报单返回空回执(通常是被前置风控拦下)",
                signal.signal_id,
                extra=build_log_extra(phase="signal"),
            )
            return ""
        return str(getattr(receipt, "primary", order_ids[0]))
