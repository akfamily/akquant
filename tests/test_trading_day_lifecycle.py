"""交易时段重构阶段 2：交易日生命周期与日终结算的验收测试."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

import akquant as aq
import pandas as pd
import pytest
from akquant import (
    Bar,
    Strategy,
    run_backtest,
    run_from_checkpoint,
    save_checkpoint,
)


def _daily_frame(dates: List[str], symbol: str = "600000.SH") -> pd.DataFrame:
    """只有日期（本地 0 点）的日线."""
    idx = pd.to_datetime(dates).tz_localize("Asia/Shanghai")
    n = len(idx)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": symbol,
            "open": [10.0 + i for i in range(n)],
            "high": [10.5 + i for i in range(n)],
            "low": [9.5 + i for i in range(n)],
            "close": [10.2 + i for i in range(n)],
            "volume": [1000.0] * n,
        }
    )


def _seen(result: Any) -> Any:
    """取回测结果中策略记录的 seen."""
    strategy = result.strategy
    assert strategy is not None
    return strategy.seen


class _Recorder(Strategy):
    """记录每根 bar 的本地时间."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: List[pd.Timestamp] = []

    def on_bar(self, bar: Bar) -> None:
        self.seen.append(
            pd.Timestamp(bar.timestamp, tz="UTC").tz_convert("Asia/Shanghai")
        )


def test_date_only_bars_rewritten_to_close() -> None:
    """只有日期的日线改写为当日 15:00（A 股收盘）."""
    result = run_backtest(
        strategy=_Recorder,
        data=_daily_frame(["2024-01-02", "2024-01-03", "2024-01-04"]),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    probe = result.strategy
    assert probe is not None
    assert [t.strftime("%Y-%m-%d %H:%M") for t in probe.seen] == [
        "2024-01-02 15:00",
        "2024-01-03 15:00",
        "2024-01-04 15:00",
    ]


def test_intraday_bars_are_not_rewritten() -> None:
    """含非 0 点时刻的数据原样保留."""
    df = _daily_frame(["2024-01-02", "2024-01-03"])
    df["timestamp"] = df["timestamp"] + pd.Timedelta(hours=10)
    result = run_backtest(
        strategy=_Recorder,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert [t.strftime("%H:%M") for t in _seen(result)] == ["10:00", "10:00"]


def test_date_only_bars_non_china_timezone_keep_local_date() -> None:
    """非中国时区（规则 B → ALWAYS_OPEN）：改写到当日 23:59:59.999999999."""
    idx = pd.to_datetime(["2024-01-05", "2024-01-08"]).tz_localize("America/New_York")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "AAPL",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )

    class Probe(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.seen: List[pd.Timestamp] = []

        def on_bar(self, bar: Bar) -> None:
            self.seen.append(
                pd.Timestamp(bar.timestamp, tz="UTC").tz_convert("America/New_York")
            )

    result = run_backtest(
        strategy=Probe,
        data=df,
        symbols=["AAPL"],
        initial_cash=1e5,
        show_progress=False,
        timezone="America/New_York",
    )
    got = _seen(result)
    assert [t.date() for t in got] == [dt.date(2024, 1, 5), dt.date(2024, 1, 8)]
    assert all(t.hour == 23 and t.minute == 59 for t in got)


def test_sessions_and_trading_calendar_parameters_validate() -> None:
    """未知模板名、空日历报 ValueError."""
    df = _daily_frame(["2024-01-02"])
    with pytest.raises(ValueError, match="未知时段模板"):
        run_backtest(
            strategy=_Recorder,
            data=df,
            symbols=["600000.SH"],
            initial_cash=1e5,
            show_progress=False,
            sessions={"default": "NOPE"},
        )
    with pytest.raises(ValueError, match="trading_calendar"):
        run_backtest(
            strategy=_Recorder,
            data=df,
            symbols=["600000.SH"],
            initial_cash=1e5,
            show_progress=False,
            trading_calendar=[],
        )


def test_trading_calendar_accepts_strings_and_timestamps() -> None:
    """日历元素可以是 date、字符串或 Timestamp."""
    df = _daily_frame(["2024-01-02", "2024-01-03"])
    result = run_backtest(
        strategy=_Recorder,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        trading_calendar=[
            "2023-12-29",
            dt.date(2024, 1, 2),
            pd.Timestamp("2024-01-03"),
        ],
    )
    assert len(_seen(result)) == 2


def test_multi_symbol_frame_rewrites_per_symbol() -> None:
    """多标的 DataFrame 按标的判断：只有日期的标的改写，含时刻的标的保留."""
    daily = _daily_frame(["2024-01-02", "2024-01-03"], symbol="600000.SH")
    intraday = _daily_frame(["2024-01-02", "2024-01-03"], symbol="000001.SZ")
    intraday["timestamp"] = intraday["timestamp"] + pd.Timedelta(hours=10)
    df = pd.concat([daily, intraday], ignore_index=True)

    class Probe(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.seen: dict[str, List[str]] = {}

        def on_bar(self, bar: Bar) -> None:
            local = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert("Asia/Shanghai")
            self.seen.setdefault(bar.symbol, []).append(local.strftime("%H:%M"))

    result = run_backtest(
        strategy=Probe,
        data=df,
        symbols=["600000.SH", "000001.SZ"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert _seen(result) == {
        "600000.SH": ["15:00", "15:00"],
        "000001.SZ": ["10:00", "10:00"],
    }


def test_bar_list_rewritten_without_mutating_input() -> None:
    """Bar 列表输入同样改写，但不修改调用方传入的 Bar 对象."""
    days = ["2024-01-02", "2024-01-03"]
    bars = [
        Bar(
            timestamp=pd.Timestamp(day, tz="Asia/Shanghai").value,
            open=10.0,
            high=10.0,
            low=10.0,
            close=10.0,
            volume=1.0,
            symbol="600000.SH",
        )
        for day in days
    ]
    original = [b.timestamp for b in bars]
    result = run_backtest(
        strategy=_Recorder,
        data=bars,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert [t.strftime("%H:%M") for t in _seen(result)] == ["15:00", "15:00"]
    assert [b.timestamp for b in bars] == original


def test_run_from_checkpoint_rewrites_and_validates(tmp_path: Path) -> None:
    """run_from_checkpoint 同样改写只有日期的数据，并校验 sessions 参数."""
    first = run_backtest(
        strategy=_Recorder,
        data=_daily_frame(["2024-01-02", "2024-01-03"]),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    ckpt = tmp_path / "c.pkl"
    save_checkpoint(first.engine, first.strategy, str(ckpt))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="未知时段模板"):
        run_from_checkpoint(
            checkpoint_path=str(ckpt),
            data=_daily_frame(["2024-01-04"]),
            show_progress=False,
            sessions={"default": "NOPE"},
        )

    resumed = run_from_checkpoint(
        checkpoint_path=str(ckpt),
        data=_daily_frame(["2024-01-04", "2024-01-05"]),
        show_progress=False,
        trading_calendar=["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    )
    seen = _seen(resumed)
    assert [t.strftime("%Y-%m-%d %H:%M") for t in seen[-2:]] == [
        "2024-01-04 15:00",
        "2024-01-05 15:00",
    ]


class _DailyBuyAt0925(Strategy):
    """每日 09:25 定时买 1 股，记录所有事件时刻."""

    def __init__(self) -> None:
        super().__init__()
        self.events: List[tuple[str, int]] = []

    def on_start(self) -> None:
        self.schedule_daily("09:25:00", "buy")

    def on_bar(self, bar: Bar) -> None:
        self.events.append(("bar", int(bar.timestamp)))

    def on_timer(self, payload: str) -> None:
        assert self.ctx is not None
        self.events.append(("timer", int(self.ctx.current_time)))
        if payload == "buy":
            self.buy(symbol="600000.SH", quantity=100, tag="timer-buy")


def test_first_day_intraday_timer_not_backfilled_before_first_bar() -> None:
    """首根 bar 改写到 15:00 后，当日 09:25 定时器不再补触发（无前视、时间不倒退）."""
    result = run_backtest(
        strategy=_DailyBuyAt0925,
        data=_daily_frame(["2024-04-01", "2024-04-02", "2024-04-03"]),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        fill_policy=aq.CurrentClose(),
    )
    strategy = result.strategy
    assert strategy is not None
    times = [ts for _, ts in strategy.events]
    assert times == sorted(times), strategy.events

    fills = result.executions_df
    local = pd.to_datetime(fills["timestamp"]).dt.tz_convert("Asia/Shanghai")
    assert not (local.dt.date == dt.date(2024, 4, 1)).any()
    first = fills.iloc[local.argsort().iloc[0]]
    assert pd.Timestamp(first["timestamp"]).tz_convert("Asia/Shanghai") == (
        pd.Timestamp("2024-04-02 09:25", tz="Asia/Shanghai")
    )
    assert float(first["price"]) == pytest.approx(10.2)  # 04-01 收盘


def test_engine_configure_sessions_validates_directly() -> None:
    """Rust 侧 Engine.configure_sessions 自身校验模板名与空日历."""
    engine = aq.Engine()
    engine.configure_sessions({"default": "CN_STOCK"}, [dt.date(2024, 1, 2)])
    engine.configure_sessions()
    with pytest.raises(ValueError, match="未知时段模板"):
        engine.configure_sessions({"IF": "NOPE"})
    with pytest.raises(ValueError, match="trading_calendar"):
        engine.configure_sessions(None, [])


def test_unparsable_asset_type_falls_back_to_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """资产类型字符串无法识别时回退默认资产类型并只告警一次（不落到全天连续）."""
    from akquant.backtest.engine import _build_session_asset_type_of

    class _Conf:
        asset_type = "warrant"

    asset_type_of = _build_session_asset_type_of(
        {}, {"A": _Conf(), "B": _Conf()}, aq.AssetType.Futures, []
    )
    with caplog.at_level("WARNING", logger="akquant"):
        assert asset_type_of("A") == aq.AssetType.Futures
        assert asset_type_of("B") == aq.AssetType.Futures
    warnings = [r for r in caplog.records if "无法识别" in r.getMessage()]
    assert len(warnings) == 1


def _snapshot_times(result: Any) -> List[pd.Timestamp]:
    """日终快照时间（本地）.

    取 ``result.positions`` 的索引（逐条对应引擎 ``record_snapshot``，含空仓快照）；
    ``equity_curve`` 是逐事件的净值曲线，每根 bar 都有一个点，不能用来观察日终结算。
    """
    return list(result.positions.index.tz_convert("Asia/Shanghai"))


class _BuyOnce(Strategy):
    """第一根 bar 下一张 NextOpen 的 Day 限价单（价格够不着），观察过期时机."""

    def __init__(self) -> None:
        super().__init__()
        self.done = False

    def on_bar(self, bar: Bar) -> None:
        if not self.done:
            self.done = True
            self.buy(bar.symbol, 100, price=1.0, time_in_force=aq.TimeInForce.Day)


def test_day_order_expires_at_its_trading_day_close() -> None:
    """日线：首根 bar 下的 NextOpen Day 单在下一交易日撮合一次，次日收盘过期."""
    result = run_backtest(
        strategy=_BuyOnce,
        data=_daily_frame(["2024-01-02", "2024-01-03", "2024-01-04"]),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    orders = result.orders_df
    assert len(orders) == 1
    row = orders.iloc[0]
    assert str(row["status"]).lower().endswith("expired")
    # 01-02 收盘下单 → 豁免 → 01-03 收盘过期
    updated = pd.Timestamp(row["updated_at"]).tz_convert("Asia/Shanghai")
    assert updated.strftime("%Y-%m-%d %H:%M") == "2024-01-03 15:00"


def test_last_trading_day_is_settled_at_end() -> None:
    """回测最后一个交易日也做日终结算：期末未成交 Day 单为 Expired."""
    result = run_backtest(
        strategy=_BuyOnce,
        data=_daily_frame(["2024-01-02", "2024-01-03"]),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert str(result.orders_df.iloc[0]["status"]).lower().endswith("expired")


def test_night_session_settles_once_per_trading_day_not_at_midnight() -> None:
    """夜盘跨 0 点：周一 21:00 起属周二，周一 15:00 结算一次，不在 0 点再结算."""
    ts = pd.to_datetime(
        [
            "2024-01-08 14:00",
            "2024-01-08 21:30",
            "2024-01-08 23:30",
            "2024-01-09 00:30",
            "2024-01-09 10:00",
            "2024-01-09 14:30",
        ]
    ).tz_localize("Asia/Shanghai")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "au2406",
            "open": 500.0,
            "high": 501.0,
            "low": 499.0,
            "close": 500.0,
            "volume": 10.0,
        }
    )
    result = run_backtest(
        strategy=_Recorder,
        data=df,
        symbols=["au2406"],
        initial_cash=1e6,
        show_progress=False,
        # run_backtest 的 asset_type 关键字会被当成策略参数报错，期货标的改用
        # instruments_config 声明。
        config=aq.BacktestConfig(
            strategy_config=aq.StrategyConfig(),
            instruments_config=[
                aq.InstrumentConfig(
                    symbol="au2406",
                    asset_type="FUTURES",
                    multiplier=1000.0,
                    margin_ratio=0.1,
                )
            ],
        ),
        sessions={"au": "CN_FUTURES_NIGHT_0230"},
    )
    snaps = [t.strftime("%Y-%m-%d %H:%M") for t in _snapshot_times(result)]
    # 每个交易日一个收盘快照：01-08 15:00 与 01-09 15:00
    assert "2024-01-08 15:00" in snaps and "2024-01-09 15:00" in snaps
    assert not any(t.endswith("00:00") for t in snaps)


class _BuyAtOpenProbe(Strategy):
    """当日 09:25 定时器买入，记录同日收盘 bar 上的可用数量."""

    def __init__(self) -> None:
        super().__init__()
        self.available_on_bar: List[tuple] = []

    def on_start(self) -> None:
        self.schedule_daily("09:25", "buy")

    def on_timer(self, payload: str) -> None:
        if payload == "buy":
            self.buy("600000.SH", 100)

    def on_bar(self, bar: Bar) -> None:
        day = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert("Asia/Shanghai").date()
        self.available_on_bar.append(
            (day, float(self.get_available_position(bar.symbol)))
        )


def test_t_plus_one_intraday_buy_not_sellable_on_same_day_close_bar() -> None:
    """日线改到 15:00 后：当日 09:25 买入，当日 15:00 bar 的 on_bar 里可用必须为 0."""
    result = run_backtest(
        strategy=_BuyAtOpenProbe,
        data=_daily_frame(["2024-01-02", "2024-01-03", "2024-01-04"]),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        t_plus_one=True,
    )
    strategy = result.strategy
    assert strategy is not None
    seen = dict(strategy.available_on_bar)
    assert seen[dt.date(2024, 1, 3)] == 0.0  # 01-03 09:25 买入，当日不可卖
    assert seen[dt.date(2024, 1, 4)] == 100.0  # 01-03 收盘结算后解冻


class _BuyOnceAt0925SellAt1530(Strategy):
    """01-03 09:25 按当前价买入一次；每日 15:30 记录可用并卖出全部可用."""

    def __init__(self) -> None:
        super().__init__()
        self.bought = False
        self.available_on_bar: List[tuple] = []
        self.available_at_sell_timer: List[tuple] = []

    def on_start(self) -> None:
        self.schedule_daily("09:25", "buy")
        self.schedule_daily("15:30", "sell")

    def _local_now(self) -> pd.Timestamp:
        assert self.ctx is not None
        return pd.Timestamp(int(self.ctx.current_time), tz="UTC").tz_convert(
            "Asia/Shanghai"
        )

    def on_timer(self, payload: str) -> None:
        if payload == "buy" and not self.bought:
            self.bought = True
            self.buy("600000.SH", 100, fill_mode=aq.CurrentClose())
        elif payload == "sell":
            available = float(self.get_available_position("600000.SH"))
            self.available_at_sell_timer.append((self._local_now().date(), available))
            if available > 0:
                self.sell("600000.SH", available, fill_mode=aq.CurrentClose())

    def on_bar(self, bar: Bar) -> None:
        day = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert("Asia/Shanghai").date()
        self.available_on_bar.append(
            (day, float(self.get_available_position(bar.symbol)))
        )


def test_t_plus_one_buy_then_after_close_timer_is_next_trading_day() -> None:
    """09:25 立即成交的买入：当日 15:00 bar 不可卖；15:30 已属下一交易日.

    当日收盘结算已解冻 T+1，15:30 下的卖单在下一交易日成交.
    """
    result = run_backtest(
        strategy=_BuyOnceAt0925SellAt1530,
        data=_daily_frame(["2024-01-02", "2024-01-03", "2024-01-04"]),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        t_plus_one=True,
    )
    strategy = result.strategy
    assert strategy is not None
    # 首个 09:25 是 01-03 的（01-02 的已早于首根 bar），按当前价立即成交
    on_bar = dict(strategy.available_on_bar)
    assert on_bar[dt.date(2024, 1, 3)] == 0.0  # 当日收盘 bar 仍冻结
    at_timer = dict(strategy.available_at_sell_timer)
    # 01-02 15:30 尚无持仓；01-03 15:30 在 01-03 收盘结算之后，属于 01-04 交易日
    assert at_timer[dt.date(2024, 1, 2)] == 0.0
    assert at_timer[dt.date(2024, 1, 3)] == 100.0

    fills = result.executions_df
    local = pd.to_datetime(fills["timestamp"]).dt.tz_convert("Asia/Shanghai")
    sides = fills["side"].astype(str).str.lower()
    buy_times = list(local[sides == "buy"])
    sell_times = list(local[sides == "sell"])
    assert [t.strftime("%Y-%m-%d %H:%M") for t in buy_times] == ["2024-01-03 09:25"]
    # 卖出发生在买入所属交易日的收盘结算之后（不在买入当日盘中）
    assert sell_times
    assert all(
        t > pd.Timestamp("2024-01-03 15:00", tz="Asia/Shanghai") for t in sell_times
    )


class _BuyFirstBar(Strategy):
    """首根 bar 按当前价买入 150 股."""

    def __init__(self) -> None:
        super().__init__()
        self.done = False

    def on_bar(self, bar: Bar) -> None:
        if not self.done:
            self.done = True
            self.buy(bar.symbol, 150)


def test_close_snapshot_reflects_same_day_forced_liquidation() -> None:
    """收盘快照记在结算之后：最后交易日收盘强平后，最后一行快照为空仓."""
    ts = pd.to_datetime(["2024-01-03 10:00", "2024-01-03 14:00"]).tz_localize(
        "Asia/Shanghai"
    )
    bars = [
        Bar(
            timestamp=t.value,
            open=p,
            high=p,
            low=p,
            close=p,
            volume=10000.0,
            symbol="LIQ",
        )
        for t, p in zip(ts, [100.0, 20.0])
    ]
    result = run_backtest(
        strategy=_BuyFirstBar,
        data=bars,
        symbols=["LIQ"],
        initial_cash=10000.0,
        show_progress=False,
        fill_policy=aq.CurrentClose(),
        lot_size=1,
        risk_config=aq.config.RiskConfig(
            account_mode="margin",
            initial_margin_ratio=0.5,
            maintenance_margin_ratio=0.5,
            financing_rate_annual=0.0,
            borrow_rate_annual=0.0,
            allow_force_liquidation=True,
        ),
    )
    assert not result.liquidation_audit_df.empty
    positions = result.positions
    # 结果里另有 get_results 在最后事件时刻(14:00)补的期末点; 最后一行是收盘快照
    assert positions.index[-1].tz_convert("Asia/Shanghai") == pd.Timestamp(
        "2024-01-03 15:00", tz="Asia/Shanghai"
    )
    assert float(positions.iloc[-1].get("LIQ", 0.0)) == 0.0


def test_intraday_data_beyond_trading_calendar_raises_chinese_value_error() -> None:
    """分钟数据超出显式 trading_calendar：引擎开日时报中文 ValueError."""
    ts = pd.to_datetime(["2024-01-02 10:00", "2024-01-05 10:00"]).tz_localize(
        "Asia/Shanghai"
    )
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "600000.SH",
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 100.0,
        }
    )
    with pytest.raises(ValueError, match="超出 trading_calendar 覆盖范围"):
        run_backtest(
            strategy=_Recorder,
            data=df,
            symbols=["600000.SH"],
            initial_cash=1e5,
            show_progress=False,
            trading_calendar=["2023-12-29", "2024-01-02", "2024-01-03"],
        )


def test_user_timer_payload_with_reserved_day_close_prefix_is_rejected() -> None:
    """用户定时器不能占用引擎内部日终结算前缀（Python 入口与 Rust 入口都拒绝）."""
    reserved = "__akq_day_close__|2024-01-02"

    class _Bad(Strategy):
        def on_start(self) -> None:
            self.schedule_daily("09:30", reserved)

    with pytest.raises(ValueError, match="日终结算专用前缀"):
        run_backtest(
            strategy=_Bad,
            data=_daily_frame(["2024-01-02", "2024-01-03"]),
            symbols=["600000.SH"],
            initial_cash=1e5,
            show_progress=False,
        )

    engine = aq.Engine()
    with pytest.raises(ValueError, match="日终结算专用前缀"):
        engine.add_timer(1, reserved)  # type: ignore[attr-defined]
    engine.add_timer(1, "user")  # type: ignore[attr-defined]


def test_settlement_stream_events_carry_close_time() -> None:
    """日终结算产生的流事件，信封 ts 为交易日收盘时刻（不是最后一根 bar）."""
    ts = pd.to_datetime(["2024-01-03 10:00", "2024-01-04 10:00"]).tz_localize(
        "Asia/Shanghai"
    )
    bars = [
        Bar(
            timestamp=t.value,
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1e4,
            symbol="INT",
        )
        for t in ts
    ]
    events: List[Any] = []
    run_backtest(
        strategy=_BuyFirstBar,
        data=bars,
        symbols=["INT"],
        initial_cash=10000.0,
        show_progress=False,
        fill_policy=aq.CurrentClose(),
        lot_size=1,
        risk_config=aq.config.RiskConfig(
            account_mode="margin",
            initial_margin_ratio=0.5,
            financing_rate_annual=36.5,
            borrow_rate_annual=0.0,
            allow_force_liquidation=False,
        ),
        on_event=events.append,
    )
    settlement = {
        e["payload"]["date"]: pd.Timestamp(int(e["ts"]), tz="UTC").tz_convert(
            "Asia/Shanghai"
        )
        for e in events
        if e.get("event_type") == "settlement"
    }
    assert settlement == {
        "2024-01-03": pd.Timestamp("2024-01-03 15:00", tz="Asia/Shanghai"),
        "2024-01-04": pd.Timestamp("2024-01-04 15:00", tz="Asia/Shanghai"),
    }


class _AlwaysBuy(Strategy):
    """每根 bar 市价买 30 股（写法同 tests/test_account_risk_rules.py）."""

    def on_bar(self, bar: Bar) -> None:
        """每根 bar 买入."""
        self.buy(bar.symbol, 30)


def _bars_at(times: List[str], prices: List[float], symbol: str) -> List[Bar]:
    return [
        Bar(
            timestamp=pd.Timestamp(t, tz="Asia/Shanghai").value,
            open=p,
            high=p,
            low=p,
            close=p,
            volume=10000.0,
            symbol=symbol,
        )
        for t, p in zip(times, prices)
    ]


def test_daily_loss_rule_does_not_reset_at_midnight_within_night_session() -> None:
    """夜盘：周一 23:30 触发日亏损后，周二 00:30 仍属同一交易日，继续拒单.

    旧实现按本地自然日在 0 点重置起始权益，00:30 的单会被放行。
    """
    bars = _bars_at(
        ["2024-01-08 21:30", "2024-01-08 23:30", "2024-01-09 00:30"],
        [100.0, 50.0, 50.0],
        "RISK",
    )
    result = run_backtest(
        data=bars,
        strategy=_AlwaysBuy,
        symbols="RISK",
        initial_cash=100000.0,
        show_progress=False,
        fill_policy=aq.CurrentClose(),
        lot_size=1,
        risk_config=aq.config.RiskConfig(max_daily_loss=0.01),
        sessions={"RISK": "CN_FUTURES_NIGHT_0230"},
    )
    orders = result.orders_df
    reasons = orders["reject_reason"].fillna("").astype(str)
    rejected = orders[reasons.str.contains("Daily loss")]
    assert len(rejected) == 2, orders[["created_at", "status", "reject_reason"]]


_NIGHT_TIMES = ["2024-01-08 21:30", "2024-01-08 23:30", "2024-01-09 00:30"]


def _run_night(**kwargs: Any) -> pd.DataFrame:
    """夜盘三根 bar 跑 _AlwaysBuy，返回订单表."""
    result = run_backtest(
        data=_bars_at(_NIGHT_TIMES, [100.0, 50.0, 50.0], "RISK"),
        strategy=_AlwaysBuy,
        symbols="RISK",
        initial_cash=100000.0,
        show_progress=False,
        fill_policy=aq.CurrentClose(),
        lot_size=1,
        sessions={"RISK": "CN_FUTURES_NIGHT_0230"},
        **kwargs,
    )
    return result.orders_df


def test_strategy_daily_loss_does_not_reset_at_midnight_within_night_session() -> None:
    """策略级日亏损：0 点不重置基线，00:30 的单仍被拒."""
    orders = _run_night(strategy_id="alpha", strategy_max_daily_loss={"alpha": 100.0})
    reasons = orders["reject_reason"].fillna("").astype(str)
    rejected = orders[reasons.str.contains("daily loss")]
    assert len(rejected) == 2, orders[["created_at", "status", "reject_reason"]]


def test_risk_budget_does_not_reset_at_midnight_within_night_session() -> None:
    """风险预算按交易日重置：0 点不重置，00:30 的单仍因预算耗尽被拒."""
    orders = _run_night(portfolio_risk_budget=4500.0, risk_budget_reset_daily=True)
    reasons = orders["reject_reason"].fillna("").astype(str)
    rejected = orders[reasons.str.contains("risk budget")]
    assert len(rejected) == 1, orders[["created_at", "status", "reject_reason"]]
    assert pd.Timestamp(rejected.iloc[0]["created_at"]).hour == 0


class _HoldOnce(Strategy):
    """首根 bar 买入 100 股，之后不动。模块顶层定义以便 pickle."""

    def __init__(self) -> None:
        """初始化."""
        super().__init__()
        self.ordered = False

    def on_bar(self, bar: Bar) -> None:
        """首根 bar 买入."""
        if not self.ordered:
            self.ordered = True
            self.buy(bar.symbol, 100)


class _Idle(Strategy):
    """不交易的策略（存档时 ``_known_orders`` 为空，策略实例可 pickle）."""

    def on_bar(self, bar: Bar) -> None:
        """什么也不做."""


def _intraday_frame(stamps: List[str]) -> pd.DataFrame:
    """给定本地时间戳的 600000.SH 分钟线."""
    idx = pd.to_datetime(stamps).tz_localize("Asia/Shanghai")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "600000.SH",
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e6,
        }
    )


def _ten_oclock_frame(dates: List[str]) -> pd.DataFrame:
    """每个日期 10:00 一根 bar."""
    return _intraday_frame([f"{d} 10:00" for d in dates])


def test_checkpoint_roundtrips_trading_day_progress() -> None:
    """存档保存当前交易日与已结算交易日."""
    from akquant.akquant import Engine

    r = run_backtest(
        data=_ten_oclock_frame(["2024-01-04", "2024-01-05"]),
        strategy=_HoldOnce,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r.engine is not None
    assert r.engine.current_trading_day == dt.date(2024, 1, 5)
    assert r.engine.last_settled_trading_day == dt.date(2024, 1, 5)
    restored = Engine()
    assert restored.current_trading_day is None
    restored.load_state_bytes(r.engine.get_state_bytes())
    assert restored.current_trading_day == dt.date(2024, 1, 5)
    assert restored.last_settled_trading_day == dt.date(2024, 1, 5)


def test_resume_after_weekend_settles_each_day_once(tmp_path: Any) -> None:
    """周五存档、周一续跑：续跑段不重复结算周五."""
    part1 = run_backtest(
        data=_ten_oclock_frame(["2024-01-03", "2024-01-04", "2024-01-05"]),
        strategy=_HoldOnce,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    ckpt = tmp_path / "c.pkl"
    assert part1.engine is not None
    save_checkpoint(part1.engine, part1.strategy, str(ckpt))
    part2 = run_from_checkpoint(
        checkpoint_path=str(ckpt),
        data=_ten_oclock_frame(["2024-01-08", "2024-01-09"]),
        symbols=["600000.SH"],
        show_progress=False,
    )
    # 周一续跑时持仓（周三买入的 100 股）跨周末仍在。
    # 续跑结果的 positions 只含续跑段的快照：周一、周二各一条日终快照（若重复
    # 结算周五会多出一条）。最后交易日（周二）已在收盘结算，get_results 不再在
    # 末根 bar（周二 10:00）补一条期末快照（类别 (b)）。
    times = [t.strftime("%Y-%m-%d %H:%M") for t in _snapshot_times(part2)]
    assert times == ["2024-01-08 15:00", "2024-01-09 15:00"]
    assert (part2.positions["600000.SH"] == 100.0).all()


def test_settled_last_day_has_no_terminal_snapshot_before_close() -> None:
    """最后交易日已在收盘结算：期末不在末根 bar 时刻补快照，快照时间严格递增."""
    r = run_backtest(
        data=_ten_oclock_frame(["2024-01-03", "2024-01-04", "2024-01-05"]),
        strategy=_HoldOnce,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    times = [t.strftime("%Y-%m-%d %H:%M") for t in _snapshot_times(r)]
    assert times == ["2024-01-03 15:00", "2024-01-04 15:00", "2024-01-05 15:00"]
    # 收盘快照即期末状态：首根 bar 买入的 100 股在最后一条快照里
    assert r.positions["600000.SH"].iloc[-1] == 100.0
    # 净值曲线仍逐事件记点，期末点留在末根 bar 时刻（不新增收盘点）
    last_equity_ts = r.equity_curve.index[-1].tz_convert("Asia/Shanghai")
    assert last_equity_ts.strftime("%Y-%m-%d %H:%M") == "2024-01-05 10:00"
    assert len(r.equity_curve) == 3


def test_resume_same_day_does_not_reopen_or_resettle(tmp_path: Any) -> None:
    """存档当日（已结算）再续跑同一天的后续 bar：不重新开日、不重复结算."""
    part1 = run_backtest(
        data=_intraday_frame(["2024-01-05 10:00"]),
        strategy=_Idle,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    ckpt = tmp_path / "c.pkl"
    assert part1.engine is not None
    save_checkpoint(part1.engine, part1.strategy, str(ckpt))
    part2 = run_from_checkpoint(
        checkpoint_path=str(ckpt),
        data=_intraday_frame(["2024-01-05 14:00"]),
        symbols=["600000.SH"],
        show_progress=False,
    )
    assert part2.engine is not None
    assert part2.engine.current_trading_day == dt.date(2024, 1, 5)
    # 存档段已在 15:00 结算过当日；续跑段不得再结算一次（整体恰一条 15:00 快照）
    closes = [
        t
        for t in _snapshot_times(part1) + _snapshot_times(part2)
        if t.strftime("%Y-%m-%d %H:%M") == "2024-01-05 15:00"
    ]
    assert len(closes) == 1


def _live_stock_instrument() -> Any:
    from akquant.akquant import AssetType, Instrument

    return Instrument(
        symbol="600000",
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=100,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _local_bars(stamps: List[str]) -> List[Bar]:
    idx = pd.to_datetime(stamps).tz_localize("Asia/Shanghai")
    return [Bar(int(t.value), 10.0, 10.0, 10.0, 10.0, 100.0, "600000") for t in idx]


def _local_minute(ts_ns: int) -> str:
    local = pd.Timestamp(ts_ns, tz="UTC").tz_convert("Asia/Shanghai")
    return local.strftime("%Y-%m-%d %H:%M")


class _LiveDayOrderProbe(Strategy):
    """首根 bar 下一张够不着的 Day 限价单.

    on_start 里等 replay 推送线程结束：全部行情先进 feed，引擎取下一动作时
    “有事件可取”是确定的，不受推送线程与主循环的竞态影响。
    """

    def __init__(self) -> None:
        super().__init__()
        self.done = False
        self.active_at: Dict[str, int] = {}

    def on_start(self) -> None:
        import threading

        for thread in threading.enumerate():
            if thread.name == "replay-market":
                thread.join(timeout=10)

    def on_bar(self, bar: Bar) -> None:
        assert self.ctx is not None
        self.active_at[_local_minute(int(bar.timestamp))] = len(self.ctx.active_orders)
        if not self.done:
            self.done = True
            self.buy("600000", 100, price=1.0, time_in_force=aq.TimeInForce.Day)


def _run_live_replay(
    monkeypatch: pytest.MonkeyPatch, bars: List[Bar], **kwargs: Any
) -> Any:
    """run_live(broker="replay") 跑完后返回 runner，供读取引擎结果.

    run_live 不返回结果，且日终过期不触发 on_order（订单离开在途账本时
    只留档、不回调），只能从引擎结果里读订单终态。
    """
    from akquant import run_live
    from akquant.live import _facade

    captured: List[Any] = []

    class _Recording(_facade.LiveRunner):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kw: Any) -> None:
            super().__init__(*args, **kw)
            captured.append(self)

    monkeypatch.setattr(_facade, "LiveRunner", _Recording)
    run_live(
        strategy_cls=_LiveDayOrderProbe,
        instruments=[_live_stock_instrument()],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars, "freq": "1min"},
        cash=1e6,
        show_progress=False,
        duration="30s",
        **kwargs,
    )
    assert len(captured) == 1
    return captured[0]


def _expired_at(runner: Any) -> List[str]:
    orders = runner.engine.get_results().orders
    return [
        _local_minute(int(o.updated_at))
        for o in orders
        if o.status == aq.OrderStatus.Expired
    ]


def _snapshot_minutes(runner: Any) -> List[str]:
    return [_local_minute(int(ts)) for ts, _ in runner.engine.get_results().snapshots]


def test_live_replay_settles_at_trading_day_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """实盘 replay：日终结算在交易日收盘触发（Day 单在所属交易日收盘过期）."""
    runner = _run_live_replay(
        monkeypatch,
        _local_bars(
            [
                "2024-01-02 10:00",
                "2024-01-02 14:00",
                "2024-01-03 10:00",
                "2024-01-03 14:00",
            ]
        ),
        trading_calendar=["2023-12-29", "2024-01-02", "2024-01-03"],
    )
    # 10:00 下单（非当日最后切片，不豁免）→ 01-02 15:00 过期
    assert _expired_at(runner) == ["2024-01-02 15:00"]


def test_live_replay_forwards_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_live 的 sessions 下发到引擎：全天连续时交易日在本地 0 点收盘."""
    runner = _run_live_replay(
        monkeypatch,
        _local_bars(
            [
                "2024-01-02 10:00",
                "2024-01-02 14:00",
                "2024-01-03 10:00",
                "2024-01-03 14:00",
            ]
        ),
        sessions={"default": "ALWAYS_OPEN"},
    )
    assert _expired_at(runner) == ["2024-01-03 00:00"]


def test_live_replay_forwards_trading_calendar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_live 的 trading_calendar 下发到引擎：01-02 非交易日并入 01-03."""
    runner = _run_live_replay(
        monkeypatch,
        _local_bars(
            [
                "2024-01-02 10:00",
                "2024-01-02 14:00",
                "2024-01-03 10:00",
                "2024-01-03 14:00",
                "2024-01-04 10:00",
            ]
        ),
        trading_calendar=["2023-12-29", "2024-01-03", "2024-01-04"],
    )
    # 01-02 的行情属于交易日 01-03（区间 (12-29 15:00, 01-03 15:00]）
    assert _expired_at(runner) == ["2024-01-03 15:00"]


def test_live_replay_next_day_event_inside_grace_window_settles_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """实盘宽限窗口内到达的下一交易日行情：先补结算上一交易日，定时器随后空转.

    全天连续时 01-02 收盘 = 01-03 00:00，实盘收盘定时器在 00:01。00:00:30 的
    行情已在 feed 中且早于定时器，引擎先取事件，开日时补结算 01-02；之后到期
    的定时器发现已结算，不再重复。
    """
    runner = _run_live_replay(
        monkeypatch,
        _local_bars(
            [
                "2024-01-02 10:00:00",
                "2024-01-02 14:00:00",
                "2024-01-03 00:00:30",
                "2024-01-03 10:00:00",
            ]
        ),
        sessions={"default": "ALWAYS_OPEN"},
    )
    assert _expired_at(runner) == ["2024-01-03 00:00"]
    # 补结算先于新交易日首个事件的 on_bar
    assert runner._stop_target_strategy.active_at["2024-01-03 00:00"] == 0
    assert runner._stop_target_strategy.active_at["2024-01-02 14:00"] == 1
    # 01-02 只结算一次
    assert _snapshot_minutes(runner).count("2024-01-03 00:00") == 1
