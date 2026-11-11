"""on_bars(batch): 同一时间片的多标的 bar 批量回调 (#441)."""

from types import SimpleNamespace
from typing import Any

import akquant as aq
import pandas as pd
import pytest
from akquant import BarBatch, Strategy
from akquant import bar_batch as bb

TS = list(pd.date_range("2024-01-02 09:31", periods=6, freq="1min", tz="Asia/Shanghai"))
TS_NS = [int(ts.value) for ts in TS]
PX = {
    "A": [10, 11, 12, 13, 14, 15],
    "B": [20, 21, 22, 23, 24, 25],
    "C": [30, 31, 32, 33, 34, 35],
}


def _df(symbol: str, skip: tuple[int, ...] = ()) -> pd.DataFrame:
    rows = [
        {
            "date": ts,
            "open": px - 0.5,
            "high": px + 1,
            "low": px - 1,
            "close": px,
            "volume": 1e6,
            "symbol": symbol,
        }
        for i, (ts, px) in enumerate(zip(TS, PX[symbol]))
        if i not in skip
    ]
    return pd.DataFrame(rows)


def _data() -> dict[str, pd.DataFrame]:
    """C 缺第 3 分钟(09:33)的 bar."""
    return {"A": _df("A"), "B": _df("B"), "C": _df("C", skip=(2,))}


def _run(strategy: Any, **kwargs: Any) -> Any:
    result = aq.run_backtest(
        data=_data(),
        strategy=strategy,
        symbols=list("ABC"),
        initial_cash=1e6,
        show_progress=False,
        **kwargs,
    )
    return result.strategy


class _Recorder(Strategy):
    """记下回调序列与每个批次."""

    def on_start(self) -> None:
        self.warmup_period = 1
        self.set_history_depth(5)
        for symbol in "ABC":
            self.subscribe(symbol)
        self.calls: list[tuple[str, int, str]] = []
        self.batches: list[BarBatch] = []
        self.visible: list[dict[str, float]] = []
        self.trades: list[tuple[str, float, int]] = []

    def on_bar(self, bar: Any) -> None:
        self.calls.append(("bar", int(bar.timestamp), bar.symbol))

    def on_bars(self, batch: BarBatch) -> None:
        self.calls.append(("bars", batch.timestamp, ""))
        self.batches.append(batch)
        self.visible.append(
            {
                s: float(self.get_history(count=1, symbol=s, field="close")[-1])
                for s in "ABC"
            }
        )
        assert self.current_bar is None

    def on_trade(self, trade: Any) -> None:
        self.trades.append((trade.symbol, float(trade.price), int(trade.timestamp)))


def test_backtest_fires_once_per_slice_including_incomplete_and_last() -> None:
    """每个时间片恰好一次, 缺 bar 的片与最后一片都触发."""
    s = _run(_Recorder)
    assert [b.timestamp for b in s.batches] == TS_NS
    assert all(b.flush_reason == "boundary" for b in s.batches)
    assert [b.symbols for b in s.batches] == [
        ("A", "B", "C"),
        ("A", "B", "C"),
        ("A", "B"),
        ("A", "B", "C"),
        ("A", "B", "C"),
        ("A", "B", "C"),
    ]
    assert [b.missing for b in s.batches] == [(), (), ("C",), (), (), ()]


def test_backtest_batch_follows_all_on_bar_of_the_slice_without_lookahead() -> None:
    """先逐根 on_bar 再 on_bars; 批次派发时看不到下一片的数据."""
    s = _run(_Recorder)
    for i, ts in enumerate(TS_NS):
        in_slice = [c for c in s.calls if c[1] == ts]
        assert in_slice[-1] == ("bars", ts, "")
        assert {c[0] for c in in_slice[:-1]} == {"bar"}
        # 批次派发在下一片任何回调之前
        pos = s.calls.index(("bars", ts, ""))
        assert all(c[1] <= ts for c in s.calls[:pos])
        for bar in s.batches[i]:
            assert s.visible[i][bar.symbol] == float(bar.close)
    # 缺 bar 的片: C 看到的仍是上一片的收盘价
    assert s.visible[2]["C"] == 31.0


def test_batch_accessors() -> None:
    """BarBatch 的取值接口."""
    batch = _run(_Recorder).batches[2]
    assert len(batch) == 2
    assert "A" in batch and "C" not in batch
    assert float(batch["B"].close) == 22.0
    assert batch.get("C") is None
    assert [bar.symbol for bar in batch] == ["A", "B"]
    with pytest.raises(KeyError):
        batch["C"]


class _OrderInBars(_Recorder):
    def on_bars(self, batch: BarBatch) -> None:
        super().on_bars(batch)
        if batch.timestamp == TS_NS[1]:
            self.buy("B", 100)


class _OnBarOnly(Strategy):
    """只写 on_bar: 在 09:32 片的最后一根 bar 上下单."""

    def on_start(self) -> None:
        self.warmup_period = 1
        for symbol in "ABC":
            self.subscribe(symbol)
        self.trades: list[tuple[str, float, int]] = []

    def on_bar(self, bar: Any) -> None:
        if int(bar.timestamp) == TS_NS[1] and bar.symbol == "C":
            self.buy("B", 100)

    def on_trade(self, trade: Any) -> None:
        self.trades.append((trade.symbol, float(trade.price), int(trade.timestamp)))


@pytest.mark.parametrize(
    ("current_close", "expected"),
    [
        (False, ("B", 21.5, TS_NS[2])),  # 缺省: 下一根开盘
        (True, ("B", 21.0, TS_NS[1])),  # 本片收盘
    ],
)
def test_orders_in_on_bars_fill_like_orders_in_last_on_bar(
    current_close: bool, expected: tuple[str, float, int]
) -> None:
    """on_bars 里下的单与在本片最后一根 on_bar 里下单同一口径."""
    kwargs: dict[str, Any] = {"fill_policy": aq.CurrentClose()} if current_close else {}
    assert _run(_OnBarOnly, **kwargs).trades == [expected]
    assert _run(_OrderInBars, **kwargs).trades == [expected]


def test_strategy_without_on_bars_registers_nothing() -> None:
    """只写 on_bar 的策略: 不收集、不登记批次定时器."""
    s = _run(_OnBarOnly)
    assert "_bar_slice_state" not in s.__dict__
    assert s.__dict__.get("_framework_wants_bar_batches") is False


def test_warmup_bars_are_excluded_and_all_warmup_slices_do_not_fire() -> None:
    """预热期内的 bar 不进批次(列在 missing); 全在预热期内的时间片不触发."""

    class Warm(_Recorder):
        def on_start(self) -> None:
            super().on_start()
            self.warmup_period = 4

    s = _run(Warm)
    # A/B 的第 4 根在 09:34; C 缺 09:33, 第 4 根在 09:35
    assert [(b.timestamp, b.symbols, b.missing) for b in s.batches] == [
        (TS_NS[3], ("A", "B"), ("C",)),
        (TS_NS[4], ("A", "B", "C"), ()),
        (TS_NS[5], ("A", "B", "C"), ()),
    ]


def test_functional_on_bars() -> None:
    """函数式入口 run_backtest(on_bars=...)."""
    seen: list[tuple[int, tuple[str, ...]]] = []

    def initialize(ctx: Any) -> None:
        ctx.warmup_period = 1

    def on_bar(ctx: Any, bar: Any) -> None:
        pass

    def on_bars(ctx: Any, batch: BarBatch) -> None:
        seen.append((batch.timestamp, batch.symbols))

    _run(on_bar, initialize=initialize, on_bars=on_bars)
    assert [ts for ts, _ in seen] == TS_NS
    assert seen[2][1] == ("A", "B")


# ---- 实盘派发规则: 直接驱动收集函数, 墙钟与定时器登记用替身 ----


class _Live(Strategy):
    got: list[Any]

    def on_bars(self, batch: BarBatch) -> None:
        self.got.append(
            (batch.timestamp, batch.symbols, batch.missing, batch.flush_reason)
        )


def _live(monkeypatch: pytest.MonkeyPatch, now: int = 0) -> tuple[Any, list[Any]]:
    strategy: Any = _Live()
    strategy.got = []
    timers: list[tuple[int, str]] = []
    strategy.ctx = SimpleNamespace(
        schedule=lambda ts, payload: timers.append((ts, payload)), current_time=0
    )
    strategy._set_live_market_data_owner()
    monkeypatch.setattr(bb, "_wall_clock_ns", lambda: now)
    return strategy, timers


def _feed(strategy: Any, symbol: str, ts: int) -> None:
    """按 on_bar_event 里的顺序驱动收集."""
    bar = SimpleNamespace(symbol=symbol, timestamp=ts, close=1.0)
    if bb.collect_bar(strategy, bar):
        bb.mark_bar_ready(strategy, bar)
        bb.flush_if_complete(strategy)


S = 1_000_000_000
E1, E2, E3 = 60 * S, 120 * S, 180 * S


def test_live_first_slice_waits_for_deadline_then_complete_fires_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首片没有可比的标的集合只能等截止; 之后到齐即发."""
    strategy, timers = _live(monkeypatch)
    for symbol in "ABC":
        _feed(strategy, symbol, E1)
    assert strategy.got == []
    assert timers == [(E1 + 5 * S, f"{bb.BAR_SLICE_TIMER_PREFIX}{E1}")]
    assert bb.dispatch_bar_slice_timer(strategy, timers[0][1])
    assert strategy.got == [(E1, ("A", "B", "C"), (), "deadline")]

    for symbol in "CAB":
        _feed(strategy, symbol, E2)
    assert strategy.got[-1] == (E2, ("A", "B", "C"), (), "complete")
    # 该片的截止定时器随后到点: 不重复派发
    assert bb.dispatch_bar_slice_timer(strategy, timers[1][1])
    assert len(strategy.got) == 2


def test_live_deadline_reports_missing_and_late_bar_is_not_added(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """截止时未到齐: 报 missing; 之后才到的 bar 不补进批次、只计数."""
    strategy, timers = _live(monkeypatch)
    for symbol in "ABC":
        _feed(strategy, symbol, E1)
    bb.dispatch_bar_slice_timer(strategy, timers[0][1])
    _feed(strategy, "A", E2)
    _feed(strategy, "B", E2)
    bb.dispatch_bar_slice_timer(strategy, timers[1][1])
    assert strategy.got[-1] == (E2, ("A", "B"), ("C",), "deadline")

    _feed(strategy, "C", E2)  # 迟到
    assert len(strategy.got) == 2
    assert bb._state(strategy).late_count == 1
    assert len(timers) == 2  # 迟到的 bar 不开新片


def test_live_watermark_flushes_previous_slice_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """截止前先见到更晚时间片的 bar: 上一片先派发."""
    strategy, timers = _live(monkeypatch)
    for symbol in "ABC":
        _feed(strategy, symbol, E1)
    bb.dispatch_bar_slice_timer(strategy, timers[0][1])
    _feed(strategy, "A", E2)
    _feed(strategy, "A", E3)
    assert strategy.got[-1] == (E2, ("A",), ("B", "C"), "watermark")
    # E2 的截止定时器到点时已派发过
    bb.dispatch_bar_slice_timer(strategy, timers[1][1])
    assert len(strategy.got) == 2


def test_live_deadline_counts_from_wall_clock_when_slice_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """时间片时刻早于墙钟(回放/延迟): 截止时间从当前时刻起算."""
    strategy, timers = _live(monkeypatch, now=E3)
    strategy.bars_deadline_seconds = 2.0
    _feed(strategy, "A", E1)
    assert timers == [(E3 + 2 * S, f"{bb.BAR_SLICE_TIMER_PREFIX}{E1}")]


# ---- 实盘会话(真引擎、真墙钟): 自定义行情网关按时间推 bar ----

_LIVE_BROKER = "on-bars-live"


class _TimedGateway:
    """按墙钟分三片推 bar: 第 2 片缺 C(随后迟到), 第 3 片到齐."""

    def __init__(self, feed: Any, pushed: list[int]) -> None:
        self._feed = feed
        self._pushed = pushed

    def connect(self) -> None:
        """no-op."""

    def disconnect(self) -> None:
        """no-op."""

    def subscribe(self, symbols: Any) -> None:
        """no-op."""

    def unsubscribe(self, symbols: Any) -> None:
        """no-op."""

    def on_tick(self, callback: Any) -> None:
        """no-op."""

    def on_bar(self, callback: Any) -> None:
        """no-op."""

    def _push(self, ts: int, symbols: str) -> None:
        for symbol in symbols:
            self._feed.add_bar(
                aq.Bar(
                    timestamp=ts,
                    open=10.0,
                    high=10.0,
                    low=10.0,
                    close=10.0,
                    volume=1e6,
                    symbol=symbol,
                )
            )

    def start(self) -> None:
        """推三片后返回."""
        import time

        first = time.time_ns()
        self._pushed.append(first)
        self._push(first, "ABC")
        time.sleep(1.0)
        second = time.time_ns()
        self._pushed.append(second)
        self._push(second, "AB")
        time.sleep(1.0)
        self._push(second, "C")  # 第 2 片的 C 在截止之后才到
        third = time.time_ns()
        self._pushed.append(third)
        self._push(third, "ABC")


class _LiveSession(Strategy):
    bars_deadline_seconds = 0.4

    def __init__(self) -> None:
        self.got: list[tuple[int, tuple[str, ...], tuple[str, ...], str, int]] = []
        self.bar_calls = 0

    def on_bar(self, bar: Any) -> None:
        self.bar_calls += 1

    def on_bars(self, batch: BarBatch) -> None:
        import time

        self.got.append(
            (
                batch.timestamp,
                batch.symbols,
                batch.missing,
                batch.flush_reason,
                time.time_ns(),
            )
        )


def test_live_session_deadline_complete_and_late_bar_through_real_engine() -> None:
    """真实 run_live 会话: 截止时间由引擎的墙钟定时器触发, 到齐提前派发."""
    import threading

    from akquant import AssetType, Instrument, run_live
    from akquant.gateway import register_broker, unregister_broker
    from akquant.gateway.protocols import GatewayBundle

    pushed: list[int] = []
    register_broker(
        _LIVE_BROKER,
        lambda **kw: GatewayBundle(
            market_gateway=_TimedGateway(kw["feed"], pushed),
            trader_gateway=None,
            trader_capabilities=None,
            metadata={"broker": _LIVE_BROKER},
        ),
    )
    strategy = _LiveSession()
    errors: list[BaseException] = []
    finished = threading.Event()

    def session() -> None:
        try:
            run_live(
                strategy_cls=strategy,
                instruments=[
                    Instrument(
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
                    for symbol in "ABC"
                ],
                broker=_LIVE_BROKER,
                trading_mode="paper",
                cash=1_000_000.0,
                show_progress=False,
                duration="4s",
            )
        except BaseException as exc:  # noqa: BLE001 带回主线程断言
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=session, daemon=True)
    thread.start()
    try:
        assert finished.wait(timeout=30.0), "会话未结束"
    finally:
        unregister_broker(_LIVE_BROKER)
    assert not errors, errors

    first, second, third = pushed
    assert [(ts, syms, missing, why) for ts, syms, missing, why, _ in strategy.got] == [
        (first, ("A", "B", "C"), (), "deadline"),
        (second, ("A", "B"), ("C",), "deadline"),
        (third, ("A", "B", "C"), (), "complete"),
    ]
    # 截止批次在时间片时刻 + 0.4 秒之后、下一片推送之前派发(引擎与 Python 的时钟
    # 读数有亚毫秒级偏差, 留 50 毫秒容差)
    for (ts, _, _, _, fired), next_push in zip(strategy.got[:2], pushed[1:]):
        assert ts + 350_000_000 <= fired < next_push
    assert strategy.bar_calls == 9  # 迟到的 C 仍触发 on_bar
    assert bb._state(strategy).late_count == 1


def test_replay_session_matches_backtest_boundaries_including_last_slice() -> None:
    """Replay 走实盘入口但按时间片边界派发, 末片在会话收尾前派发."""
    from akquant import AssetType, Instrument, run_live

    bars = [
        aq.Bar(
            timestamp=ts,
            open=px,
            high=px,
            low=px,
            close=px,
            volume=1e9,
            symbol=symbol,
        )
        for i, ts in enumerate(TS_NS[:3])
        for symbol, px in (("A", 10.0), ("B", 20.0), ("C", 30.0))
        if not (symbol == "C" and i == 2)
    ]
    strategy: Any = _Live()
    strategy.got = []
    run_live(
        strategy_cls=strategy,
        instruments=[
            Instrument(
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
            for symbol in "ABC"
        ],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars},
        cash=1_000_000.0,
        show_progress=False,
        duration="30s",
    )
    assert strategy.got == [
        (TS_NS[0], ("A", "B", "C"), (), "boundary"),
        (TS_NS[1], ("A", "B", "C"), (), "boundary"),
        (TS_NS[2], ("A", "B"), ("C",), "boundary"),
    ]


# ---- 实盘 on_cross_section: 由时间片批次驱动 ----


class _LiveCross(Strategy):
    crossed: list[Any]

    def on_cross_section(self, trading_date: Any, timestamp: int) -> None:
        self.crossed.append((trading_date, timestamp))


def _live_cross(
    monkeypatch: pytest.MonkeyPatch, sessions: dict[str, str]
) -> tuple[Any, list[Any], Any]:
    """实盘替身: ``sessions`` 给出各标的此刻的时段名(缺省 Continuous)."""
    import datetime as dt

    strategy: Any = _LiveCross()
    strategy.crossed = []
    timers: list[tuple[int, str]] = []
    ctx = SimpleNamespace(
        schedule=lambda ts, payload: timers.append((ts, payload)),
        current_time=0,
        trading_day=dt.date(2024, 1, 2),
        session_of=lambda symbol: (
            f"TradingSession.{sessions.get(symbol, 'Continuous')}"
        ),
    )
    strategy.ctx = ctx
    strategy._set_live_market_data_owner()
    monkeypatch.setattr(bb, "_wall_clock_ns", lambda: 0)
    return strategy, timers, ctx


def test_live_cross_section_fires_once_per_day_without_on_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """只重写 on_cross_section: 当日首个派发的完整时间片后触发, 每日一次."""
    import datetime as dt

    strategy, timers, ctx = _live_cross(monkeypatch, {})
    for symbol in "ABC":
        _feed(strategy, symbol, E1)
    assert strategy.crossed == []  # 首片等截止时间
    bb.dispatch_bar_slice_timer(strategy, timers[0][1])
    assert strategy.crossed == [(dt.date(2024, 1, 2), E1)]
    for symbol in "ABC":
        _feed(strategy, symbol, E2)
    assert len(strategy.crossed) == 1  # 当日不再触发

    ctx.trading_day = dt.date(2024, 1, 3)
    for symbol in "ABC":
        _feed(strategy, symbol, E3)  # 到齐即触发
    assert strategy.crossed[-1] == (dt.date(2024, 1, 3), E3)
    assert len(strategy.crossed) == 2


def test_live_cross_section_waits_for_unopened_symbol_but_not_suspended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """当日尚未出现的标的: 未开盘则等, 在交易时段却没有 bar(停牌)则不等."""
    import datetime as dt

    sessions: dict[str, str] = {}
    strategy, timers, ctx = _live_cross(monkeypatch, sessions)
    for symbol in "ABC":
        _feed(strategy, symbol, E1)
    bb.dispatch_bar_slice_timer(strategy, timers[0][1])

    # 次日: C 未开盘(如没有夜盘的品种), A/B 的时间片到截止也不触发
    ctx.trading_day = dt.date(2024, 1, 3)
    sessions["C"] = "Closed"
    _feed(strategy, "A", E2)
    _feed(strategy, "B", E2)
    bb.dispatch_bar_slice_timer(strategy, timers[1][1])
    assert len(strategy.crossed) == 1
    # C 开盘后仍无 bar(停牌): 下一片截止时触发
    sessions["C"] = "Continuous"
    _feed(strategy, "A", E3)
    _feed(strategy, "B", E3)
    bb.dispatch_bar_slice_timer(strategy, timers[2][1])
    assert strategy.crossed[-1] == (dt.date(2024, 1, 3), E3)


def test_live_cross_section_skips_watermark_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """先见到更晚时间片而派发的批次不触发(历史里已有更晚的 bar)."""
    strategy, timers, _ = _live_cross(monkeypatch, {})
    _feed(strategy, "A", E1)
    _feed(strategy, "A", E2)  # E1 因水位线派发: 不触发; E2 到齐: 触发
    assert [ts for _, ts in strategy.crossed] == [E2]
    assert len(timers) == 2


def test_replay_session_fires_on_cross_section_each_day() -> None:
    """replay(实盘入口)里 on_cross_section 每个交易日触发一次, 此前从不触发."""
    from akquant import AssetType, Instrument, run_live

    stamps = [
        "2023-01-03 09:31:00",
        "2023-01-03 09:32:00",
        "2023-01-04 09:31:00",
        "2023-01-04 09:32:00",
    ]
    ts_ns = [int(pd.Timestamp(s, tz="Asia/Shanghai").value) for s in stamps]
    bars = [
        aq.Bar(
            timestamp=ts,
            open=10.0,
            high=10.0,
            low=10.0,
            close=10.0,
            volume=1e9,
            symbol=symbol,
        )
        for ts in ts_ns
        for symbol in "AB"
    ]
    strategy: Any = _LiveCross()
    strategy.crossed = []
    run_live(
        strategy_cls=strategy,
        instruments=[
            Instrument(
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
            for symbol in "AB"
        ],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars},
        cash=1_000_000.0,
        show_progress=False,
        duration="30s",
    )
    assert [(str(day), ts) for day, ts in strategy.crossed] == [
        ("2023-01-03", ts_ns[0]),
        ("2023-01-04", ts_ns[2]),
    ]
