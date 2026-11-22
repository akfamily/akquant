"""实盘会话中刷新合约表: 新合约登记进引擎、订阅行情、进入策略快照, 并可下单成交.

会话跑在 paper 模式, 行情由按节奏推送的测试网关提供: 网关只推已订阅标的的 bar,
未订阅的标的等待订阅(至多 10 秒), 因此新合约的 bar 能推到即说明订阅已下发。
合约来源在首根 bar 之后返回新合约; 策略在第二根 bar 等新合约出现在快照里, 收到新合约
首根 bar 时下单。
"""

import datetime as dt
import threading
from typing import Any, Callable, List, Optional, Sequence

import pandas as pd
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar
from akquant.gateway import register_broker, unregister_broker
from akquant.gateway.protocols import GatewayBundle

OLD = "REFRESH_A"
NEW = "REFRESH_B"
BROKER = "instrument-refresh-paced"
_START = pd.Timestamp("2024-01-02 10:00", tz="Asia/Shanghai")
_CALENDAR = ["2023-12-29", "2024-01-02", "2024-01-03"]


def _instrument(symbol: str) -> Instrument:
    return Instrument(symbol=symbol, asset_type=AssetType.Stock, lot_size=1)


def _bar(symbol: str, second: int) -> Bar:
    return Bar(
        timestamp=int((_START + pd.Timedelta(seconds=second)).value),
        open=10.0,
        high=10.5,
        low=9.5,
        close=10.0,
        volume=100_000.0,
        symbol=symbol,
    )


BARS = sorted(
    [_bar(OLD, s) for s in range(6)] + [_bar(NEW, s) for s in range(3, 6)],
    key=lambda b: b.timestamp,
)


class _Gateway:
    """逐根推 bar; 标的未订阅时等它被订阅(至多 10 秒)再推."""

    def __init__(self, feed: Any, symbols: Sequence[str]) -> None:
        self._feed = feed
        self.symbols = [str(s) for s in symbols]
        self.stop = threading.Event()
        self.finished = threading.Event()

    def connect(self) -> None:
        """无外部连接."""

    def disconnect(self) -> None:
        """无外部连接."""

    def subscribe(self, symbols: Sequence[str]) -> None:
        self.symbols = [str(s) for s in symbols]

    def unsubscribe(self, symbols: Sequence[str]) -> None:
        self.symbols = [s for s in self.symbols if s not in set(symbols)]

    def on_tick(self, callback: Callable[[dict], None]) -> None:
        """不出 tick."""

    def on_bar(self, callback: Callable[[dict], None]) -> None:
        """Bar 直接进 feed."""

    def start(self) -> None:
        try:
            for bar in BARS:
                waited = 0.0
                while bar.symbol not in self.symbols and waited < 10.0:
                    if self.stop.wait(0.01):
                        return
                    waited += 0.01
                if bar.symbol not in self.symbols:
                    return
                self._feed.add_bar(bar)
                if self.stop.wait(0.05):
                    return
        finally:
            self.finished.set()


class _Source:
    """首次刷新等策略放行后返回新合约; 之后的刷新时刻在一天以后."""

    def __init__(self, gate: threading.Event) -> None:
        self.gate = gate
        self.calls = 0

    def next_refresh(self) -> dt.datetime:
        now = dt.datetime.now(dt.timezone.utc)
        return now if self.calls == 0 else now + dt.timedelta(days=1)

    def refresh(self, now: Optional[dt.datetime] = None) -> List[Instrument]:
        self.calls += 1
        self.gate.wait(timeout=10.0)
        return [_instrument(OLD), _instrument(NEW)]


class _Strategy(Strategy):
    def __init__(self) -> None:
        self.gate = threading.Event()
        self.old_bars = 0
        self.new_bars = 0
        self.seen_new: Optional[str] = None
        self.trades: List[Any] = []
        self.rejected: List[Any] = []

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol == NEW:
            self.new_bars += 1
            if self.new_bars == 1:  # 有行情的时刻下单(无行情时刻撮合的单按规则拒绝)
                self.buy(NEW, 100)
            return
        self.old_bars += 1
        if self.old_bars == 1:
            self.gate.set()
        elif self.old_bars == 2:
            for _ in range(1000):
                if NEW in self._instrument_snapshots:
                    break
                threading.Event().wait(0.01)
            self.seen_new = self.get_instrument(NEW).asset_type

    def on_trade(self, trade: Any) -> None:
        self.trades.append(trade)

    def on_reject(self, order: Any) -> None:
        self.rejected.append(order)


def test_new_instrument_registered_subscribed_and_tradable_mid_session() -> None:
    """运行中刷新出的新合约: 策略能看到、收到行情、下单成交, 不被当作未登记拒单."""
    strategy = _Strategy()
    source = _Source(strategy.gate)
    gateways: List[_Gateway] = []

    def build(**kwargs: Any) -> GatewayBundle:
        gateway = _Gateway(kwargs["feed"], kwargs["symbols"])
        gateways.append(gateway)
        return GatewayBundle(
            market_gateway=gateway,
            trader_gateway=None,
            trader_capabilities=None,
            metadata={"broker": BROKER, "bounded_event_total": len(BARS)},
        )

    register_broker(BROKER, build)
    try:
        run_live(
            strategy_cls=strategy,
            instruments=[_instrument(OLD)],
            broker=BROKER,
            trading_mode="paper",
            show_progress=False,
            duration="20s",
            trading_calendar=_CALENDAR,
            instrument_source=source,
        )
    finally:
        unregister_broker(BROKER)
        for gateway in gateways:
            gateway.stop.set()
            gateway.finished.wait(timeout=2.0)

    assert strategy.seen_new == "STOCK"
    assert strategy.new_bars == 3
    assert strategy.rejected == []
    assert [(t.symbol, float(t.quantity)) for t in strategy.trades] == [(NEW, 100.0)]
    assert source.calls == 1
