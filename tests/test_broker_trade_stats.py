"""broker_live 会话级交易统计：LiveTradeStats 与 BrokerTradeStats（spec 5.5）."""

import logging
import math
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest
from akquant.akquant import AssetType, Instrument, LiveTradeStats
from akquant.gateway.broker_stats import BrokerTradeStats
from akquant.gateway.broker_strategy_api import wrap_state_invalidation
from akquant.gateway.models import (
    UnifiedAccount,
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedPosition,
    UnifiedTrade,
)
from akquant.live._runner import LiveRunner


def test_live_trade_stats_smoke() -> None:
    """LiveTradeStats 冒烟：一笔亏损平仓与回撤."""
    s = LiveTradeStats(100_000.0)
    s.on_equity(1, 100_000.0)
    s.on_fill("AAA", "buy", 100.0, 10.0, 1)
    s.on_fill("AAA", "sell", 100.0, 9.0, 2)
    snap = s.snapshot(3, 99_900.0, {})
    assert snap["losing_trades"] == 1.0 and snap["win_rate"] == 0.0
    assert snap["max_drawdown_value"] == 100.0


# ---------------------------------------------------------------------------
# BrokerTradeStats 单元测试
# ---------------------------------------------------------------------------


@dataclass
class T:
    """柜台成交替身（字段与 UnifiedTrade 一致）."""

    trade_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp_ns: int = 1
    client_order_id: str = "mw-tag-1"
    position_effect: str = "auto"


@dataclass
class P:
    """柜台持仓替身（字段与 UnifiedPosition 一致）.

    direction 的真实取值：middleware 不填（""）；CTP 为 "Buy"/"Sell"，
    且空头的 quantity 为负。
    """

    symbol: str
    quantity: float
    avg_price: float
    direction: str = ""


def _stats(
    own: bool = True, mult: float | None = 1.0, asset: str = "stock"
) -> BrokerTradeStats:
    return BrokerTradeStats(
        is_own_trade=lambda p: own,
        multiplier_of=lambda s: mult,
        asset_type_of=lambda s: asset,
    )


def _position(st: BrokerTradeStats, symbol: str) -> float:
    inner = st._stats
    assert inner is not None
    return inner.position(symbol)


def test_replayed_trade_id_counted_once() -> None:
    """恢复循环重放同一 trade_id 只计一次."""
    st = _stats()
    st.initialize(equity=100_000.0, positions=[], timestamp_ns=0)
    wrapped = wrap_state_invalidation(
        lambda n, p: None, lambda: [], on_new_fill=st.on_trade
    )
    for _ in range(3):  # 恢复循环重放
        wrapped("trade", T("t1", "AAA", "buy", 100, 10.0))
    wrapped("trade", T("t2", "AAA", "sell", 100, 11.0))
    snap = st.snapshot(3, 100_100.0, {})
    assert snap is not None
    assert snap["trade_count"] == 1.0
    assert snap["largest_win"] == 100.0
    # 重放若被重复计入，会剩 200 股的多头库存（trade_count 看不出来）
    assert _position(st, "AAA") == 0.0


def test_foreign_trade_not_counted() -> None:
    """不属于本会话的成交不计入."""
    st = _stats(own=False)
    st.initialize(equity=1.0, positions=[], timestamp_ns=0)
    st.on_trade(T("t1", "AAA", "buy", 100, 10.0))
    st.on_trade(T("t2", "AAA", "sell", 100, 11.0))
    snap = st.snapshot(3, 1.0, {})
    assert snap is not None
    assert snap["trade_count"] == 0.0
    assert _position(st, "AAA") == 0.0


def test_seeded_position_closed_gives_pnl() -> None:
    """会话前持仓（middleware 形态：direction 为空）播种后平仓计入盈亏."""
    st = _stats()
    st.initialize(equity=100_000.0, positions=[P("AAA", 100, 10.0)], timestamp_ns=0)
    st.on_trade(T("t1", "AAA", "sell", 100, 12.0))
    snap = st.snapshot(2, 100_200.0, {})
    assert snap is not None
    assert snap["winning_trades"] == 1.0 and snap["largest_win"] == 200.0


def test_seeded_ctp_short_position() -> None:
    """CTP 空头持仓（direction="Sell"、quantity 为负）播种为空头."""
    st = _stats(asset="futures", mult=10.0)
    st.initialize(
        equity=1e6, positions=[P("rb2601", -2, 3500.0, "Sell")], timestamp_ns=0
    )
    assert _position(st, "rb2601") == -2.0
    st.on_trade(T("t1", "rb2601", "buy", 2, 3400.0, position_effect="close_today"))
    snap = st.snapshot(2, 1e6 + 2000.0, {})
    assert snap is not None
    assert snap["largest_win"] == 2000.0


def test_seed_without_avg_price_skipped(caplog: pytest.LogCaptureFixture) -> None:
    """持仓没有均价（middleware 目前不填 avg_price）时不播种并告警."""
    st = _stats()
    with caplog.at_level(logging.WARNING):
        st.initialize(equity=1e5, positions=[P("AAA", 100, 0.0)], timestamp_ns=0)
    assert _position(st, "AAA") == 0.0
    assert any("均价" in r.getMessage() for r in caplog.records)


def test_seed_hedged_position_nets_without_fake_close() -> None:
    """同一合约多空两行（锁仓）按净持仓播种，不在启动时制造一笔假平仓."""
    st = _stats(asset="futures", mult=10.0)
    st.initialize(
        equity=1e6,
        positions=[
            P("rb2601", 3, 3500.0, "Buy"),
            P("rb2601", -1, 3600.0, "Sell"),
        ],
        timestamp_ns=0,
    )
    assert _position(st, "rb2601") == 2.0
    snap = st.snapshot(1, 1e6, {})
    assert snap is not None and snap["trade_count"] == 0.0


def test_excess_close_truncated_not_reversed() -> None:
    """股票卖出超过持仓的部分截掉，不当成反向开仓."""
    st = _stats()
    st.initialize(equity=100_000.0, positions=[], timestamp_ns=0)
    st.on_trade(T("t1", "AAA", "buy", 100, 10.0))
    st.on_trade(T("t2", "AAA", "sell", 300, 11.0))  # 股票卖出一律视为平仓：截断为 100
    snap = st.snapshot(3, 100_100.0, {})
    assert snap is not None
    assert snap["trade_count"] == 1.0
    # 没有开出 200 股空头
    assert _position(st, "AAA") == 0.0


def test_stock_sell_without_position_ignored() -> None:
    """股票无持仓时的卖出（会话外买入的仓）整笔忽略."""
    st = _stats(asset="fund")
    st.initialize(equity=100_000.0, positions=[], timestamp_ns=0)
    st.on_trade(T("t1", "510300.SH", "sell", 100, 4.0))
    assert _position(st, "510300.SH") == 0.0


def test_futures_open_short_allowed() -> None:
    """期货 position_effect=open 的卖出是开空."""
    st = _stats(asset="futures", mult=300.0)
    st.initialize(equity=1e6, positions=[], timestamp_ns=0)
    st.on_trade(T("t1", "IF", "sell", 1, 3800.0, position_effect="open"))
    assert _position(st, "IF") == -1.0


def test_unknown_asset_honours_explicit_position_effect() -> None:
    """未配置资产类型的标的：明确的 open 卖出是开空，明确的 close 照常截断."""
    st = _stats(asset="", mult=300.0)
    st.initialize(equity=1e6, positions=[], timestamp_ns=0)
    st.on_trade(T("t1", "IF", "sell", 1, 3800.0, position_effect="open"))
    assert _position(st, "IF") == -1.0
    st.on_trade(T("t2", "IF", "buy", 3, 3790.0, position_effect="close"))
    assert _position(st, "IF") == 0.0


def test_unknown_asset_auto_sell_is_close() -> None:
    """未配置资产类型且 position_effect=auto（middleware 股票成交）：卖出视为平仓."""
    st = _stats(asset="")
    st.initialize(equity=1e5, positions=[], timestamp_ns=0)
    st.on_trade(T("t1", "600000.SH", "sell", 100, 10.0))
    assert _position(st, "600000.SH") == 0.0
    st.on_trade(T("t2", "600000.SH", "buy", 100, 10.0))
    st.on_trade(T("t3", "600000.SH", "sell", 300, 11.0))
    assert _position(st, "600000.SH") == 0.0


def test_futures_close_today_truncated() -> None:
    """期货 close_today 超过持仓的部分截掉."""
    st = _stats(asset="futures", mult=300.0)
    st.initialize(equity=1e6, positions=[], timestamp_ns=0)
    st.on_trade(T("t1", "IF", "buy", 1, 3800.0, position_effect="open"))
    st.on_trade(T("t2", "IF", "sell", 2, 3810.0, position_effect="close_today"))
    assert _position(st, "IF") == 0.0
    snap = st.snapshot(3, 1e6 + 3000.0, {})
    assert snap is not None and snap["largest_win"] == 3000.0


def test_missing_multiplier_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    """未配置合约乘数：按 1 计算，同一标的只告警一次."""
    st = _stats(mult=None, asset="futures")
    st.initialize(equity=1e6, positions=[], timestamp_ns=0)
    with caplog.at_level(logging.WARNING):
        st.on_trade(T("t1", "IF", "buy", 1, 3800.0, position_effect="open"))
        st.on_trade(T("t2", "IF", "buy", 1, 3801.0, position_effect="open"))
    assert sum("乘数" in r.getMessage() for r in caplog.records) == 1
    assert _position(st, "IF") == 2.0


def test_trade_without_id_skipped(caplog: pytest.LogCaptureFixture) -> None:
    """没有 trade_id 的成交不计入，告警一次."""
    st = _stats()
    st.initialize(equity=1.0, positions=[], timestamp_ns=0)
    wrapped = wrap_state_invalidation(
        lambda n, p: None, lambda: [], on_new_fill=st.on_trade
    )
    with caplog.at_level(logging.WARNING):
        wrapped("trade", T("", "AAA", "buy", 1, 1.0))
        wrapped("trade", T("", "AAA", "buy", 1, 1.0))
    assert _position(st, "AAA") == 0.0
    assert sum("trade_id" in r.getMessage() for r in caplog.records) == 1


def test_stats_error_does_not_break_state_sync(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """统计器抛异常时只告警，柜台持仓同步照常."""

    class _Cache:
        def __init__(self) -> None:
            self.fills: list[tuple[str, float]] = []

        def apply_fill(self, symbol: str, qty: float) -> None:
            self.fills.append((symbol, qty))

        def invalidate_available(self) -> None:
            pass

        def invalidate_account(self) -> None:
            pass

        def invalidate_open_orders(self) -> None:
            pass

    def _boom(_payload: Any) -> None:
        raise RuntimeError("boom")

    cache = _Cache()
    wrapped = wrap_state_invalidation(
        lambda n, p: None, lambda: [cache], on_new_fill=_boom
    )
    with caplog.at_level(logging.WARNING):
        wrapped("trade", T("t1", "AAA", "buy", 5, 1.0, client_order_id=""))
    assert cache.fills == [("AAA", 5.0)]
    assert any("统计" in r.getMessage() for r in caplog.records)


def test_bad_broker_values_skipped(caplog: pytest.LogCaptureFixture) -> None:
    """非有限价格、未知方向等脏数据跳过，不抛异常."""
    st = _stats()
    st.initialize(equity=1e5, positions=[], timestamp_ns=0)
    st.on_trade(T("t1", "AAA", "buy", 100, math.nan))
    st.on_trade(T("t2", "AAA", "hold", 100, 10.0))
    st.on_trade(T("t3", "AAA", "buy", 0, 10.0))
    assert _position(st, "AAA") == 0.0


def test_uninitialized_snapshot_is_none() -> None:
    """未初始化（柜台查询失败）时 snapshot 返回 None，on_trade/on_equity 无副作用."""
    st = _stats()
    st.on_trade(T("t1", "AAA", "buy", 100, 10.0))
    st.on_equity(1, 100.0)
    assert st.snapshot(2, 100.0, {}) is None
    assert not st.initialized


def test_non_positive_equity_rejected(caplog: pytest.LogCaptureFixture) -> None:
    """柜台权益缺失（0）时不初始化；运行中的 0 权益点被跳过，不制造假回撤."""
    st = _stats()
    with caplog.at_level(logging.WARNING):
        st.initialize(equity=0.0, positions=[], timestamp_ns=0)
    assert not st.initialized
    st.initialize(equity=100_000.0, positions=[], timestamp_ns=0)
    st.on_equity(1, 0.0)
    st.on_equity(2, 100_000.0)
    snap = st.snapshot(3, 100_000.0, {})
    assert snap is not None and snap["max_drawdown_value"] == 0.0


def test_snapshot_uses_normalized_last_prices() -> None:
    """成交与最新价的标的后缀大小写不一致时仍能算浮盈."""
    st = _stats()
    st.initialize(equity=100_000.0, positions=[], timestamp_ns=0)
    st.on_trade(T("t1", "600000.sh", "buy", 100, 10.0))
    snap = st.snapshot(2, 100_100.0, {"600000.SH": 11.0})
    assert snap is not None and snap["unrealized_pnl"] == 100.0


# ---------------------------------------------------------------------------
# 与回测同口径
# ---------------------------------------------------------------------------


def _weekday_bars(closes: list[float], symbol: str = "AAA") -> list[Bar]:
    days = pd.bdate_range("2024-01-02", periods=len(closes), tz="Asia/Shanghai")
    return [
        Bar(
            timestamp=(d + pd.Timedelta(hours=15)).value,
            open=c,
            high=c + 0.5,
            low=c - 0.5,
            close=c,
            volume=1e6,
            symbol=symbol,
        )
        for d, c in zip(days, closes)
    ]


class _FourFills(Strategy):
    """买 100 → 卖 100 → 买 200 → 卖 200，每单间隔两根 bar（避开 T+1）."""

    def __init__(self) -> None:
        super().__init__()
        self._n = 0

    def on_bar(self, bar: Bar) -> None:
        self._n += 1
        plan = {1: ("buy", 100), 3: ("sell", 100), 5: ("buy", 200), 7: ("sell", 200)}
        action = plan.get(self._n)
        if action is None:
            return
        if action[0] == "buy":
            self.buy(bar.symbol, action[1])
        else:
            self.sell(bar.symbol, action[1])


def test_broker_stats_match_backtest_metrics() -> None:
    """同一组成交：BrokerTradeStats 的交易汇总与回测 result.metrics 完全相等."""
    result = run_backtest(
        data=_weekday_bars([10, 10, 11, 11, 12, 12, 11, 11, 11]),
        strategy=_FourFills(),
        initial_cash=100_000,
        commission_rate=0.0,
        show_progress=False,
    )
    trades = result.trades_df.sort_values("entry_time")
    assert len(trades) == 2, "用例应产生两笔平仓交易"
    m = result.metrics_df["value"]
    assert float(m["total_commission"]) == 0.0
    assert float(m["winning_trades"]) >= 1 and float(m["losing_trades"]) >= 1

    st = _stats()
    st.initialize(equity=100_000.0, positions=[], timestamp_ns=0)
    seq = 0
    for row in cast(list[Any], list(trades.itertuples())):
        for side, price, ts in (
            ("Buy", float(row.entry_price), row.entry_time),
            ("Sell", float(row.exit_price), row.exit_time),
        ):
            seq += 1
            st.on_trade(
                UnifiedTrade(
                    trade_id=f"t{seq}",
                    broker_order_id=f"b{seq}",
                    client_order_id=f"c{seq}",
                    symbol="AAA",
                    side=side,
                    quantity=float(row.quantity),
                    price=price,
                    timestamp_ns=pd.Timestamp(ts).value,
                )
            )
    snap = st.snapshot(seq + 1, float(m["end_market_value"]), {})
    assert snap is not None
    for key in (
        "winning_trades",
        "losing_trades",
        "profit_factor",
        "largest_win",
        "largest_loss",
        "avg_pnl",
    ):
        assert snap[key] == float(m[key]), key


# ---------------------------------------------------------------------------
# LiveRunner 接线
# ---------------------------------------------------------------------------


class _Gw:
    """最小交易网关替身：权益/持仓查询与下单."""

    def __init__(self, positions: list[UnifiedPosition] | None = None) -> None:
        self.positions = list(positions or [])
        self.account_queries = 0
        self.requests: list[Any] = []

    def query_account(self) -> UnifiedAccount:
        self.account_queries += 1
        return UnifiedAccount(
            account_id="a", equity=100_000.0, cash=100_000.0, available_cash=1e5
        )

    def query_positions(self) -> list[UnifiedPosition]:
        return list(self.positions)

    def sync_today_trades(self) -> list[Any]:
        return []

    def place_order(self, req: Any) -> str:
        self.requests.append(req)
        return f"b{len(self.requests)}"


class _Slot:
    """broker_live 策略替身（只需要 on_error）."""

    def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
        pass


def _runner(gw: _Gw, n_slots: int = 1) -> tuple[LiveRunner, list[Any]]:
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "middleware"
    runner.instruments = [
        Instrument("600000.SH", AssetType.Stock),
        Instrument("IF2612", AssetType.Futures, multiplier=300.0),
    ]
    runner._init_broker_bridge_state()
    slots: list[Any] = [_Slot() for _ in range(n_slots)]
    for slot in slots:
        runner._install_broker_order_submitter(cast(Any, gw), cast(Any, slot))
    runner._baseline_broker_state(gw)
    return runner, slots


def test_runner_shares_one_stats_and_initializes_once() -> None:
    """多 slot 共用一个会话级统计器，且只初始化一次（会话启动时用柜台持仓播种）."""
    gw = _Gw([UnifiedPosition("600000.SH", 100, 100, avg_price=10.0)])
    runner, slots = _runner(gw, n_slots=2)
    stats = slots[0].execution.trade_stats
    assert stats is not None and stats.initialized
    assert slots[1].execution.trade_stats is stats
    # 启动激活时预载各 slot 的账户缓存，供每 bar 权益采样只读
    for slot in slots:
        assert slot.execution.state_cache.account_if_loaded() is not None
    queries = gw.account_queries
    runner._baseline_broker_state(gw)
    assert gw.account_queries == queries
    assert stats._stats.position("600000.SH") == 100.0


def test_runner_multiplier_and_asset_type_from_instruments() -> None:
    """合约乘数与资产类型取自 run_live 的 instruments（后缀大小写不敏感）."""
    runner, _ = _runner(_Gw())
    assert runner._instrument_multiplier("IF2612") == 300.0
    assert runner._instrument_asset_type("IF2612") == "futures"
    assert runner._instrument_asset_type("600000.sh") == "stock"
    assert runner._instrument_multiplier("unknown") is None
    assert runner._instrument_asset_type("unknown") == ""


def test_runner_counts_own_trades_only() -> None:
    """只统计本会话的单：外来单即使已进 id 映射表也不计入."""
    gw = _Gw([UnifiedPosition("600000.SH", 100, 100, avg_price=10.0)])
    runner, slots = _runner(gw)
    slot = slots[0]
    stats = slot.execution.trade_stats
    rt = runner._broker_runtime

    # 前端手动单：委托回报先到，会把 id 写进 runner 的映射表
    rt.queue_event(
        "order",
        UnifiedOrderSnapshot(
            client_order_id="manual-1",
            broker_order_id="m1",
            symbol="600000.SH",
            status=UnifiedOrderStatus.FILLED,
            filled_quantity=100,
            avg_fill_price=13.0,
        ),
    )
    rt.drain_events(slot)
    rt.queue_event(
        "trade",
        UnifiedTrade("tm1", "m1", "manual-1", "600000.SH", "Sell", 100, 13.0, 1),
    )
    rt.drain_events(slot)
    snap = stats.snapshot(2, 100_000.0, {})
    assert snap["trade_count"] == 0.0
    assert stats._stats.position("600000.SH") == 100.0

    # 本会话策略下的单：成交回报只带 broker_order_id，按下单时绑定的归属认领
    receipt = slot.execution.submit_order(symbol="600000.SH", side="Sell", quantity=100)
    rt.queue_event(
        "trade",
        UnifiedTrade("t1", receipt.primary, "", "600000.SH", "Sell", 100, 12.0, 3),
    )
    rt.drain_events(slot)
    snap = stats.snapshot(4, 100_200.0, {})
    assert snap["trade_count"] == 1.0 and snap["largest_win"] == 200.0


def test_runner_counts_session_prefixed_trade() -> None:
    """client_order_id 带本会话前缀的成交计入."""
    runner, slots = _runner(_Gw())
    stats = slots[0].execution.trade_stats
    prefix = f"middleware-{runner._broker_session_tag}-"
    rt = runner._broker_runtime
    rt.queue_event(
        "trade",
        UnifiedTrade("t1", "x1", prefix + "zz1", "600000.SH", "Buy", 100, 10.0, 1),
    )
    rt.drain_events(slots[0])
    assert stats._stats.position("600000.SH") == 100.0


def test_broker_live_self_stats_reports_trade_fields() -> None:
    """broker_live 下 self.stats 的交易类字段来自统计器，不再只有 equity."""
    from akquant.strategy_stats import _broker_live_stats

    gw = _Gw([UnifiedPosition("600000.SH", 100, 100, avg_price=10.0)])
    runner, slots = _runner(gw)
    slot = slots[0]
    slot._last_prices = {"600000.SH": 11.0}
    stats = _broker_live_stats(slot)
    assert stats is not None
    assert stats.equity == 100_000.0
    assert stats.trade_count == 0
    assert stats.unrealized_pnl == 100.0
    assert stats.total_commission == 0.0


def test_per_bar_equity_fed_before_user_on_bar() -> None:
    """每根 bar 在用户 on_bar 之前用柜台权益调用 on_equity."""
    log: list[tuple[str, int]] = []

    class _RecStats:
        initialized = True

        def on_equity(self, timestamp_ns: int, equity: float) -> None:
            log.append(("equity", timestamp_ns))
            assert equity == 123.0

    class _Cache:
        def account_if_loaded(self) -> Any:
            return UnifiedAccount("a", equity=123.0, cash=0.0, available_cash=0.0)

    class _Exec:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.trade_stats = _RecStats()
            self.state_cache = _Cache()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    class _S(Strategy):
        def on_start(self) -> None:
            self.execution = cast(Any, _Exec(self.execution))

        def on_bar(self, bar: Bar) -> None:
            log.append(("on_bar", int(bar.timestamp)))

    bars = _weekday_bars([10, 11, 12])
    run_backtest(data=bars, strategy=_S(), initial_cash=100_000, show_progress=False)
    expected: list[tuple[str, int]] = []
    for b in bars:
        expected += [("equity", int(b.timestamp)), ("on_bar", int(b.timestamp))]
    assert log == expected


def test_per_bar_equity_never_queries_broker(caplog: pytest.LogCaptureFixture) -> None:
    """每 bar 权益采样只读已缓存的账户：柜台查询一直失败时不查询、不打异常日志."""
    from akquant.gateway.broker_execution import BrokerExecution
    from akquant.gateway.broker_state_cache import BrokerStateCache
    from akquant.strategy_events import _feed_broker_stats_equity

    class _DownGw:
        def __init__(self) -> None:
            self.queries = 0

        def query_account(self) -> Any:
            self.queries += 1
            raise ConnectionError("broker down")

    class _Holder:
        pass

    gw = _DownGw()
    cache = BrokerStateCache(gw)
    holder = _Holder()
    execution = BrokerExecution(holder, gw, cache, submitter=None)
    st = _stats()
    st.initialize(equity=100_000.0, positions=[], timestamp_ns=0)
    execution.trade_stats = st
    cast(Any, holder).execution = execution
    with caplog.at_level(logging.DEBUG):
        for ts in range(1, 11):
            _feed_broker_stats_equity(holder, ts)
    assert gw.queries == 0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    # 有缓存值（即使已被成交标记为过期）时照用它，同样不查询
    cache._account = UnifiedAccount("a", 90_000.0, 0.0, 0.0)
    cache.invalidate_account()
    _feed_broker_stats_equity(holder, 20)
    assert gw.queries == 0
    snap = st.snapshot(21, 100_000.0, {})
    assert snap is not None and snap["max_drawdown_value"] == 10_000.0
