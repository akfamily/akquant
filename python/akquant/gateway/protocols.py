from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence

from .broker_models import (
    BrokerCapability,
    UnifiedAccount,
    UnifiedExecutionReport,
    UnifiedOrderRequest,
    UnifiedOrderSnapshot,
    UnifiedPosition,
    UnifiedTrade,
)

if TYPE_CHECKING:
    from ..akquant import Bar

#: 行情源提供的历史 bar 拉取函数, ``(symbols, count) -> list[Bar]``: 给 symbols
#: 各拉最近 count 根**已收盘** bar, 按 (timestamp, symbol) 升序返回; 周期由行情源
#: 自身决定(与它推送的实时 bar 同周期)。
HistoryProvider = Callable[[Sequence[str], int], list["Bar"]]


class MarketGateway(Protocol):
    """Market data gateway protocol.

    可选方法 ``configure_session_clamp(resolver, grace_ns) -> None``:
        自己用 ``BarAggregator`` 由 tick 合成 bar 的行情网关可实现它。实盘 runner
        在启动网关线程之前调用一次: ``resolver`` 是与引擎同配置的
        ``SessionResolver``(同一 ``sessions`` / ``trading_calendar`` / 时区, 已
        ``set_active`` 全部合约), ``grace_ns`` 为 ``day_close_delay`` − 3 秒(不足
        3 秒时不调用)。实现转给 ``BarAggregator.set_session_clamp``, 使收盘后宽限
        期内的 tick 计入当日收盘 bar。不实现则不钳制; 抛异常时 runner 只告警。
        bar 来自数据或服务端的网关(如 replay)不必实现。
    """

    def connect(self) -> None:
        """Connect market data channel."""

    def disconnect(self) -> None:
        """Disconnect market data channel."""

    def subscribe(self, symbols: Sequence[str]) -> None:
        """Subscribe symbols."""

    def unsubscribe(self, symbols: Sequence[str]) -> None:
        """Unsubscribe symbols."""

    def on_tick(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register tick callback."""

    def on_bar(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register bar callback."""

    def start(self) -> None:
        """Start gateway event loop."""


class TraderGateway(Protocol):
    """Trader gateway protocol.

    可选方法 ``classify_order_error(exc) -> UnifiedErrorType``:
        下单/撤单调用抛异常时, 核心用它区分「柜台明确回绝」(``RISK_REJECTED`` /
        ``NON_RETRYABLE`` → 回吐 Rejected 事件)与「订单状态不可知」(``RETRYABLE``
        → 只发 on_error, 不谎报拒单)。**不实现则一律按状态未知处理**, 这是安全
        缺省: 超时往往发生在报文已发出之后, 谎报拒单会诱导策略重下单。
        见 ``gateway/order_errors.py``。
    """

    def connect(self) -> None:
        """Connect trader channel."""

    def disconnect(self) -> None:
        """Disconnect trader channel."""

    def place_order(self, req: UnifiedOrderRequest) -> str:
        """Place order."""

    def get_capabilities(self) -> BrokerCapability:
        """Return trader capability matrix."""

    def cancel_order(self, broker_order_id: str) -> None:
        """Cancel order."""

    def query_order(self, broker_order_id: str) -> UnifiedOrderSnapshot | None:
        """Query order."""

    def query_trades(self, since: int | None = None) -> list[UnifiedTrade]:
        """Query trades."""

    def query_account(self) -> UnifiedAccount | None:
        """Query account."""

    def query_positions(self) -> list[UnifiedPosition]:
        """Query positions."""

    def on_order(self, callback: Callable[[UnifiedOrderSnapshot], None]) -> None:
        """Register order callback."""

    def on_trade(self, callback: Callable[[UnifiedTrade], None]) -> None:
        """Register trade callback."""

    def on_execution_report(
        self, callback: Callable[[UnifiedExecutionReport], None]
    ) -> None:
        """Register execution report callback."""

    def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
        """Sync open orders — **only** ones still working at the broker.

        Implementations must filter to ``NEW`` / ``SUBMITTED`` /
        ``PARTIALLY_FILLED``. Many venues answer "query orders" with the day's
        *full* order book (cancelled and filled rows included); returning that
        as-is breaks both consumers: ``BrokerExecution.cancel_all_orders``
        cancels every entry one by one (re-cancelling a cancelled order is a
        broker-side reject), and ``BrokerRecovery`` replays the whole history as
        if it were still working.
        """

    def sync_today_trades(self) -> list[UnifiedTrade]:
        """Sync today's trades."""

    def heartbeat(self) -> bool:
        """Heartbeat check."""

    def start(self) -> None:
        """Start gateway event loop."""


@dataclass
class GatewayBundle:
    """Gateway instances and optional metadata.

    ``history_provider`` 是行情源可选提供的历史拉取函数(签名见
    ``HistoryProvider``)。提供时 ``run_live`` 默认在会话启动前按策略所需回看
    深度自动预加载历史(由 ``preload_bars`` 控制); 不提供则保持原行为。实现应
    按标的逐个拉取, 单个标的失败只跳过它并告警, 而不是整体抛异常。
    """

    market_gateway: MarketGateway | None = None
    trader_gateway: TraderGateway | None = None
    trader_capabilities: BrokerCapability | None = None
    metadata: dict[str, Any] | None = None
    history_provider: HistoryProvider | None = None
