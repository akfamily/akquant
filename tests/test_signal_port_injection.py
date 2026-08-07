"""外部线程经 SignalPort 注入指令, 引擎须接受并撮合(paper 模式).

设计依据: docs/zh/meta/signal-ingestion-rfc.md 第 4.1 节。

这条通路的意义: 让外部量化信号平台的指令进入引擎, 而**不经过策略回调** ——
策略回调内无法触达 Engine(`run(&mut self)` 独占借用, 见 RFC 4.2.1), 故注入必须
发生在引擎线程之外, 靠 crossbeam sender 跨线程送 `Event::OrderRequest`。

**测试设计要点(两条都是踩过的坑)**:

1. **必须与策略回调握手, 不能靠 sleep 定时**。`broker='replay'` 会一次性把 bar
   全塞进 feed, 引擎瞬间消费完、会话随即因 `bounded_event_total` 终止 —— 整个
   过程可能只有几十毫秒, 任何 `time.sleep` 都赶不上。这里用
   `threading.Event` 让注入线程等 `on_bar` 放行, 与会话快慢无关。

2. **bar 用当前墙钟时间戳, 不用历史日期**。live 引擎按墙钟判定 timer 是否到期,
   若 bar 落在历史(如 2024-01), 框架 timer 全部"早已到期", 时序语义错乱
   (`replay` 网关文档亦声明其 timer 行为不作保证)。

已实测确认**不是**问题的方向(留档以免重复排查): daemon 线程在 live 会话期间
不存在 GIL 饥饿 —— 探针线程以 0.05s 周期运行 2 秒, 最大间隔 0.063s。
"""

import threading
import time
from typing import Any, Callable, List, Sequence

import pandas as pd
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar
from akquant.gateway import register_broker, unregister_broker
from akquant.gateway.protocols import GatewayBundle

SYMBOL = "SIGPORT_A"
BAR_CLOSE = 10.0
BROKER = "sigport-paced"


def _instrument(symbol: str) -> Instrument:
    """构造股票标的."""
    return Instrument(
        symbol=symbol,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=1,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _bar(ts: int, close: float = BAR_CLOSE) -> Bar:
    """构造一根 bar(成交量足够, 保证可撮合)."""
    return Bar(
        timestamp=ts,
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=100_000.0,
        symbol=SYMBOL,
    )


def _bars(count: int = 6) -> List[Bar]:
    """当前墙钟时间戳起的连续 bar(见模块 docstring 第 2 条)."""
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    return [_bar(int((now + pd.Timedelta(seconds=i)).value)) for i in range(count)]


class _PacedMarketGateway:
    """按固定间隔逐根推 bar 的行情网关.

    ``replay`` 一次性推完全部 bar, 引擎瞬间跑完, 观察不到"会话进行中"的状态。
    本网关在两根之间 sleep, 让注入确实发生在会话中途。
    """

    def __init__(self, feed: Any, symbols: Sequence[str], bars: List[Bar]) -> None:
        """记录 feed、订阅集与待推事件."""
        self._feed = feed
        self._symbols = [str(s) for s in symbols]
        self._bars = bars
        self.interval = 0.25
        # 会话结束时置位, 让推送线程立刻退出。不做这件事会跨测试污染:
        # 会话由 bounded_event_total 在最后一根 bar 处终止, 而此刻推送线程仍在
        # sleep, 于是它活到下一个测试里继续占用 GIL/时序, 使结果随执行顺序漂移。
        self.stop = threading.Event()
        self.finished = threading.Event()

    def connect(self) -> None:
        """no-op: 无外部连接."""

    def disconnect(self) -> None:
        """no-op: 无外部连接."""

    def subscribe(self, symbols: Sequence[str]) -> None:
        """替换订阅集."""
        self._symbols = [str(s) for s in symbols]

    def unsubscribe(self, symbols: Sequence[str]) -> None:
        """从订阅集移除给定标的."""
        removed = {str(s) for s in symbols}
        self._symbols = [s for s in self._symbols if s not in removed]

    def on_tick(self, callback: Callable[[dict], None]) -> None:
        """仅存引用(与 CTPMarketAdapter 一致)."""

    def on_bar(self, callback: Callable[[dict], None]) -> None:
        """仅存引用(与 CTPMarketAdapter 一致)."""

    def start(self) -> None:
        """在 runner 的 daemon 线程上按间隔推送, 可被 ``stop`` 提前中断."""
        try:
            allowed = set(self._symbols)
            for bar in self._bars:
                if self.stop.is_set():
                    return
                if str(bar.symbol) in allowed:
                    self._feed.add_bar(bar)
                    # 用 wait 而非 sleep: 会话一结束就能立刻醒来退出。
                    if self.stop.wait(timeout=self.interval):
                        return
        finally:
            self.finished.set()


def _register(bars: List[Bar], created: List[_PacedMarketGateway]) -> None:
    """注册按节奏推送的行情 broker(paper 模式不需要 trader 侧).

    把建出的网关实例塞进 ``created``, 供调用方在会话结束后回收其推送线程。

    必须声明 ``bounded_event_total``, 否则会话挂死: ``duration`` 是靠 patch
    ``on_bar``/``on_tick`` 实现的(见 ``_runner.py`` 的 ``_apply_time_limit``),
    只在**有行情事件时**才检查墙钟。本网关推完最后一根即退出, 引擎随后永远阻塞
    在 ``wait_peek`` 上, ``on_bar`` 不再被调用, ``duration`` 便永不触发。
    """

    def _build(**kwargs: Any) -> GatewayBundle:
        gateway = _PacedMarketGateway(
            feed=kwargs["feed"], symbols=kwargs["symbols"], bars=bars
        )
        created.append(gateway)
        return GatewayBundle(
            market_gateway=gateway,
            trader_gateway=None,
            trader_capabilities=None,
            metadata={"broker": BROKER, "bounded_event_total": len(bars)},
        )

    register_broker(BROKER, _build)


class _SignalStrategy(Strategy):
    """自身不下单; 与注入线程双向握手, 消除时序竞态.

    订单只可能来自外部注入 —— 这是本测试的核心不变量。

    **为何要双向握手**: 注入必须落在**最后一根 bar 之前**才有撮合机会
    (``bounded_event_total`` 会在最后一根处终止会话)。单向放行会退化成"赌那个
    时间窗", 受 GIL 调度影响而随机失败。这里首根 bar 放行注入, 之后的 bar 反过来
    等注入完成才继续 —— 会话无法越过注入点, 结果与调度快慢无关。
    """

    def __init__(self) -> None:
        """初始化事件记录容器与双向握手闸门."""
        self.orders: List[Any] = []
        self.trades: List[Any] = []
        self.rejected: List[Any] = []
        self.bars_seen = 0
        self.gate = threading.Event()
        self.injected = threading.Event()

    def on_bar(self, bar: Bar) -> None:
        """首根 bar 放行注入; 第二根起等注入落地后再继续(不下单)."""
        self.bars_seen += 1
        if self.bars_seen == 1:
            self.gate.set()
            return
        if self.bars_seen == 2:
            # 阻塞在这里时 GIL 被本回调持有? 不: Event.wait 会释放 GIL,
            # 注入线程因此得以运行。超时兜底避免测试挂死。
            self.injected.wait(timeout=10.0)

    def on_order(self, order: Any) -> None:
        """记录委托事件."""
        self.orders.append(order)

    def on_trade(self, trade: Any) -> None:
        """记录成交."""
        self.trades.append(trade)

    def on_reject(self, order: Any) -> None:
        """记录拒单."""
        self.rejected.append(order)


def _run(
    strategy: _SignalStrategy,
    bind: Callable[[Any], None],
    *,
    max_order_value: float | None = None,
    bar_count: int = 6,
) -> None:
    """跑一轮 paper 会话, 把 SignalPort 交给 ``bind``.

    会话结束后必须停掉推送线程并等它退出 —— 否则它活进下一个测试, 结果会随执行
    顺序漂移(实测过: 同一份测试单跑 PASS、合跑 FAIL, 且失败集合每轮不同)。
    """
    gateways: List[_PacedMarketGateway] = []
    _register(_bars(bar_count), gateways)
    try:
        run_live(
            strategy_cls=strategy,
            instruments=[_instrument(SYMBOL)],
            broker=BROKER,
            trading_mode="paper",
            cash=1_000_000.0,
            show_progress=False,
            duration="15s",
            signal_port_ready=bind,
            strategy_max_order_value=(
                None if max_order_value is None else {"_default": max_order_value}
            ),
        )
    finally:
        unregister_broker(BROKER)
        for gateway in gateways:
            gateway.stop.set()
            gateway.finished.wait(timeout=2.0)


def _injector(
    strategy: _SignalStrategy,
    log: List[str],
    **submit_kwargs: Any,
) -> Callable[[Any], None]:
    """造一个 ``signal_port_ready`` 回调: 等 on_bar 放行后在独立线程注入一笔单."""

    def bind(port: Any) -> None:
        running = threading.Event()

        def worker() -> None:
            running.set()
            if not strategy.gate.wait(timeout=10.0):
                log.append("gate-timeout")
                return
            try:
                order_id = port.submit(**submit_kwargs)
                log.append(f"ok:{order_id}")
            except BaseException as exc:  # noqa: BLE001 — 须带回主线程断言
                log.append(f"fail:{type(exc).__name__}:{exc}")
            finally:
                strategy.injected.set()  # 放行会话继续(见 _SignalStrategy 说明)

        threading.Thread(target=worker, daemon=True).start()
        # 必须确认线程已真正开始执行才能返回。`bind` 由 runner 在 engine.run()
        # 之前同步调用, 一返回主线程就进 Rust 主循环并长期持有 GIL —— 若此刻新
        # 线程尚未被调度, 它可能整场会话都拿不到执行机会(实测: 不等的话结果随
        # 机失败, 且失败集合每轮不同)。这是 signal_port_ready 使用者的通用约束,
        # 已写进 run_live 文档。
        running.wait(timeout=5.0)

    return bind


def test_external_thread_injection_is_accepted_and_filled() -> None:
    """外部线程注入的委托必须被引擎接受并成交(全程不经策略回调)."""
    strategy = _SignalStrategy()
    log: List[str] = []
    _run(
        strategy,
        _injector(
            strategy,
            log,
            symbol=SYMBOL,
            side="Buy",
            quantity=100.0,
            price=BAR_CLOSE,
            tag="signal-001",
        ),
    )

    assert log and log[0].startswith("ok:"), f"注入未成功: {log}"
    order_id = log[0][3:]

    assert strategy.rejected == [], f"注入单被风控拒绝: {strategy.rejected}"
    assert strategy.orders, "引擎未产生委托事件 —— 注入未被接受"
    assert any(getattr(o, "id", "") == order_id for o in strategy.orders), (
        f"委托事件里找不到注入的订单 {order_id}"
    )
    assert strategy.trades, "注入单未成交"
    filled = sum(float(getattr(t, "quantity", 0.0)) for t in strategy.trades)
    assert filled == 100.0, f"成交量应为 100, 实际 {filled}"


def test_injected_order_still_passes_risk() -> None:
    """注入的委托同样受策略级风控约束, 不因来自外部而绕过."""
    strategy = _SignalStrategy()
    log: List[str] = []
    # 名义 100 * 10 = 1000 > 上限 500
    _run(
        strategy,
        _injector(
            strategy, log, symbol=SYMBOL, side="Buy", quantity=100.0, price=BAR_CLOSE
        ),
        max_order_value=500.0,
    )

    assert log and log[0].startswith("ok:"), f"注入未成功: {log}"
    assert strategy.rejected, "超限的注入单应被风控拒绝"
    assert strategy.trades == [], "被拒的注入单不应成交"
    reason = str(getattr(strategy.rejected[0], "reject_reason", ""))
    assert "order value" in reason, f"拒单原因应指明名义超限, 实际: {reason!r}"


def test_submit_returns_immediately() -> None:
    """注入只是 channel send, 不等引擎确认, 必须即时返回."""
    strategy = _SignalStrategy()
    log: List[str] = []
    elapsed: List[float] = []

    def bind(port: Any) -> None:
        running = threading.Event()

        def worker() -> None:
            running.set()
            if not strategy.gate.wait(timeout=10.0):
                log.append("gate-timeout")
                return
            started = time.monotonic()
            try:
                port.submit(symbol=SYMBOL, side="Buy", quantity=100.0, price=BAR_CLOSE)
                log.append("ok")
            except BaseException as exc:  # noqa: BLE001
                log.append(f"fail:{type(exc).__name__}")
            elapsed.append(time.monotonic() - started)
            strategy.injected.set()

        threading.Thread(target=worker, daemon=True).start()
        running.wait(timeout=5.0)  # 见 _injector 里的同款说明

    _run(strategy, bind)

    assert log == ["ok"], f"注入未成功: {log}"
    assert elapsed[0] < 0.05, f"submit 阻塞过久: {elapsed[0]:.3f}s"


def test_submit_rejects_invalid_input() -> None:
    """端口自身的入参校验: 空 symbol / 非正数量 / 未知枚举立即报错."""
    strategy = _SignalStrategy()
    accepted: List[str] = []

    def bind(port: Any) -> None:
        # 同步执行即可: 只验证参数校验, 不需要落到引擎。
        # 立刻放行会话: 本测试不注入有效单, 否则第二根 bar 会一直等下去。
        strategy.injected.set()
        for kwargs, label in (
            ({"symbol": "", "side": "Buy", "quantity": 1.0}, "空 symbol"),
            ({"symbol": "   ", "side": "Buy", "quantity": 1.0}, "空白 symbol"),
            ({"symbol": SYMBOL, "side": "Buy", "quantity": 0.0}, "零数量"),
            ({"symbol": SYMBOL, "side": "Buy", "quantity": -5.0}, "负数量"),
            ({"symbol": SYMBOL, "side": "Hold", "quantity": 1.0}, "未知 side"),
            (
                {
                    "symbol": SYMBOL,
                    "side": "Buy",
                    "quantity": 1.0,
                    "order_type": "Iceberg",
                },
                "未知 order_type",
            ),
        ):
            try:
                port.submit(**kwargs)
            except ValueError:
                continue
            accepted.append(label)

    _run(strategy, bind, bar_count=2)

    assert accepted == [], f"以下非法入参未被拒绝: {accepted}"
    assert strategy.orders == [], "非法注入不应产生委托"


def test_market_order_injection_fills_at_bar_price() -> None:
    """省略 price 即市价单, 应按行情价成交."""
    strategy = _SignalStrategy()
    log: List[str] = []
    _run(
        strategy,
        _injector(strategy, log, symbol=SYMBOL, side="Buy", quantity=50.0),
    )

    assert log and log[0].startswith("ok:"), f"注入未成功: {log}"
    assert strategy.trades, "市价注入单未成交"
    filled = sum(float(getattr(t, "quantity", 0.0)) for t in strategy.trades)
    assert filled == 50.0, f"成交量应为 50, 实际 {filled}"
