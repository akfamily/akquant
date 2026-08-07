"""外部量化信号平台接入.

把平台推来的交易指令变成委托, **不经过策略回调** —— 信号已经是指令, 不需要策略
再决策一次(设计依据见 docs/zh/meta/signal-ingestion-rfc.md)。

最小用法::

    from akquant import run_live
    from akquant.signal import QueueSignalSource, Signal

    source = QueueSignalSource()

    # 另一个线程/进程按需投信号
    source.put(Signal(signal_id="s-1", symbol="000001.SZ", action="buy",
                      quantity=100, price=10.5))

    run_live(
        instruments=[...],
        broker="ctp",
        trading_mode="paper",       # 或 "broker_live"
        signal_source=source,
    )

两种模式的风控覆盖面**不同**, 这是引擎架构决定的, 不是配置问题:

- ``paper``: 信号经引擎事件通道注入, 走**完整**风控(含 daily_loss / drawdown /
  risk_budget), 由模拟撮合器成交。
- ``broker_live``: 信号经柜台通道报单, 只有**策略级三项**限额前置生效
  (order_value / order_size / position_size)。引擎的实盘执行器不向柜台报单,
  故这条路不能走引擎注入。
"""

from .dedup import SignalDedup
from .dispatcher import SignalDispatcher
from .models import Signal, SignalAction, SignalResult, SignalStatus
from .protocols import OrderSink, SignalSource, SignalSourceBase
from .sinks import BrokerOrderSink, PaperOrderSink
from .sources import QueueSignalSource

__all__ = [
    "Signal",
    "SignalAction",
    "SignalResult",
    "SignalStatus",
    "SignalSource",
    "SignalSourceBase",
    "OrderSink",
    "PaperOrderSink",
    "BrokerOrderSink",
    "SignalDedup",
    "SignalDispatcher",
    "QueueSignalSource",
]
