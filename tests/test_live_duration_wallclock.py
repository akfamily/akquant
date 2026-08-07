"""duration 必须是真正的墙钟兜底: 行情停摆也要到点结束.

修复前的缺陷(见 docs/zh/meta/signal-ingestion-rfc.md 4.6): ``_apply_time_limit``
只 patch ``on_bar``/``on_tick``, 墙钟检查因此只在**有行情事件时**执行。行情一停
(自定义网关推完即退出、或盘后无推送), 引擎永远阻塞在 ``wait_peek`` 上,
``on_bar`` 不再被调用, ``duration`` 便永不触发 —— 会话挂死。

这与 ``_runner.py`` 中"duration 仍作安全网……墙钟兜底避免挂死"的注释所承诺的
行为不符。

修法: 把截止时刻下沉到引擎的等待循环(``Engine.set_session_deadline_ns`` +
``DataProcessor`` 在 feed 等待前检查), 不再依赖事件到达。

**本测试必须自带超时**: 若缺陷复现, 会话会永久挂住; 用子线程 + join(timeout)
把它变成断言失败而非测试套件卡死。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, List, Sequence

import pandas as pd
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar
from akquant.gateway import register_broker, unregister_broker
from akquant.gateway.protocols import GatewayBundle

SYMBOL = "DURTEST"
BROKER = "duration-quiet"


def _instrument() -> Instrument:
    """构造股票标的."""
    return Instrument(
        symbol=SYMBOL,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=1,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


class _QuietAfterFirstGateway:
    """推一根 bar 就退出的行情网关 —— 之后行情彻底停摆.

    刻意**不**声明 ``bounded_event_total``: 那个机制同样挂在 ``on_bar`` 上,
    声明了就会掩盖本测试要验证的缺陷。真实场景里任何"推完即退"的自定义网关
    都是这个形状。
    """

    def __init__(self, feed: Any, symbols: Sequence[str], bar: Bar) -> None:
        """记录 feed 与待推的唯一一根 bar."""
        self._feed = feed
        self._symbols = [str(s) for s in symbols]
        self._bar = bar

    def connect(self) -> None:
        """no-op."""

    def disconnect(self) -> None:
        """no-op."""

    def subscribe(self, symbols: Sequence[str]) -> None:
        """替换订阅集."""
        self._symbols = [str(s) for s in symbols]

    def unsubscribe(self, symbols: Sequence[str]) -> None:
        """移除订阅."""

    def on_tick(self, callback: Callable[[dict], None]) -> None:
        """仅存引用."""

    def on_bar(self, callback: Callable[[dict], None]) -> None:
        """仅存引用."""

    def start(self) -> None:
        """推一根就返回, 线程随即结束."""
        if str(self._bar.symbol) in set(self._symbols):
            self._feed.add_bar(self._bar)


class _CountingStrategy(Strategy):
    """只计数, 不下单."""

    def __init__(self) -> None:
        """初始化计数器."""
        self.bars_seen = 0

    def on_bar(self, bar: Bar) -> None:
        """计数."""
        self.bars_seen += 1


def test_duration_ends_session_when_market_data_stops() -> None:
    """行情停摆后, duration 仍须让会话在时限附近结束(而非挂死)."""
    strategy = _CountingStrategy()
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    bar = Bar(
        timestamp=int(now.value),
        open=10.0,
        high=10.5,
        low=9.5,
        close=10.0,
        volume=100_000.0,
        symbol=SYMBOL,
    )
    register_broker(
        BROKER,
        lambda **kw: GatewayBundle(
            market_gateway=_QuietAfterFirstGateway(kw["feed"], kw["symbols"], bar),
            trader_gateway=None,
            trader_capabilities=None,
            metadata={"broker": BROKER},  # 刻意不声明 bounded_event_total
        ),
    )

    finished = threading.Event()
    elapsed: List[float] = []
    errors: List[BaseException] = []

    def session() -> None:
        started = time.monotonic()
        try:
            run_live(
                strategy_cls=strategy,
                instruments=[_instrument()],
                broker=BROKER,
                trading_mode="paper",
                cash=1_000_000.0,
                show_progress=False,
                duration="3s",
            )
        except BaseException as exc:  # noqa: BLE001 — 带回主线程断言
            errors.append(exc)
        finally:
            elapsed.append(time.monotonic() - started)
            finished.set()

    thread = threading.Thread(target=session, daemon=True)
    thread.start()
    try:
        # 时限 3s, 给足余量; 超时即说明缺陷复现(会话挂死)。
        completed = finished.wait(timeout=25.0)
    finally:
        unregister_broker(BROKER)

    assert completed, "会话未在时限后结束 —— duration 没能在行情停摆时兜底, 缺陷复现"
    assert not errors, f"会话异常退出: {errors}"
    assert strategy.bars_seen == 1, f"应只收到 1 根 bar, 实际 {strategy.bars_seen}"
    # 结束时刻应贴近时限: 太早说明提前退出, 太晚说明兜底不及时。
    assert 2.5 <= elapsed[0] <= 12.0, f"结束耗时异常: {elapsed[0]:.2f}s"


def test_session_deadline_setter_is_exposed() -> None:
    """引擎须暴露 set_session_deadline_ns, 且忽略非正值."""
    from akquant import Engine

    engine = Engine()
    assert hasattr(engine, "set_session_deadline_ns")
    engine.set_session_deadline_ns(0)  # 非正 → 视为不限, 不应抛
    engine.set_session_deadline_ns(None)
    engine.set_session_deadline_ns(int(time.time() * 1_000_000_000) + 10**9)
