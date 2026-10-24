"""交易时段重构阶段 2：交易日生命周期与日终结算的验收测试."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def test_always_open_date_only_last_day_is_settled() -> None:
    """全天连续时段表下，只有日期数据改写为收盘前 1ns，最后交易日仍视同收盘并结算."""
    idx = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]).tz_localize(
        "America/New_York"
    )
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
    result = run_backtest(
        strategy=_Recorder,
        data=df,
        symbols=["AAPL"],
        initial_cash=1e5,
        show_progress=False,
        timezone="America/New_York",
    )
    assert result.engine is not None
    assert result.engine.last_settled_trading_day == dt.date(2024, 1, 5)


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


class _DailyBuyAt0935(Strategy):
    """每日 09:35（连续竞价时段内）定时买 1 股，记录所有事件时刻."""

    def __init__(self) -> None:
        super().__init__()
        self.events: List[tuple[str, int]] = []

    def on_start(self) -> None:
        self.schedule_daily("09:35:00", "buy")

    def on_bar(self, bar: Bar) -> None:
        self.events.append(("bar", int(bar.timestamp)))

    def on_timer(self, payload: str) -> None:
        assert self.ctx is not None
        self.events.append(("timer", int(self.ctx.current_time)))
        if payload == "buy":
            self.buy(symbol="600000.SH", quantity=100, tag="timer-buy")


def test_first_day_intraday_timer_not_backfilled_before_first_bar() -> None:
    """首根 bar 改写到 15:00 后，当日 09:35 定时器不再补触发（无前视、时间不倒退）."""
    result = run_backtest(
        strategy=_DailyBuyAt0935,
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
        pd.Timestamp("2024-04-02 09:35", tz="Asia/Shanghai")
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
            "2024-01-09 15:00",
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
    """当日 09:25 定时器买入，记录同日收盘 bar 上的可用数量.

    09:25 落在开盘集合竞价，订单排队到当日 15:00 日线 bar 成交（不在 09:25 成交）。
    """

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
    """日线改到 15:00：09:25 下单、15:00 bar 成交，该 bar 的 on_bar 里可用为 0."""
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
    assert seen[dt.date(2024, 1, 3)] == 0.0  # 01-03 15:00 bar 成交，当日不可卖
    assert seen[dt.date(2024, 1, 4)] == 100.0  # 01-03 收盘结算后解冻


class _BuyOnceAt0935SellAt1530(Strategy):
    """01-03 09:35（连续竞价）按当前价买入一次；每日 15:30 记录可用并卖出全部可用."""

    def __init__(self) -> None:
        super().__init__()
        self.bought = False
        self.available_on_bar: List[tuple] = []
        self.available_at_sell_timer: List[tuple] = []

    def on_start(self) -> None:
        self.schedule_daily("09:35", "buy")
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
    """09:35 立即成交的买入：当日 15:00 bar 不可卖；15:30 已属下一交易日.

    当日收盘结算已解冻 T+1，15:30 下的卖单在下一交易日成交.
    """
    result = run_backtest(
        strategy=_BuyOnceAt0935SellAt1530,
        data=_daily_frame(["2024-01-02", "2024-01-03", "2024-01-04"]),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        t_plus_one=True,
    )
    strategy = result.strategy
    assert strategy is not None
    # 首个 09:35 是 01-03 的（01-02 的已早于首根 bar），按当前价立即成交
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
    assert [t.strftime("%Y-%m-%d %H:%M") for t in buy_times] == ["2024-01-03 09:35"]
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
    ts = pd.to_datetime(["2024-01-03 10:00", "2024-01-03 15:00"]).tz_localize(
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
    # 数据走到收盘（15:00 bar），收盘结算强平；最后一行是收盘快照
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
    ts = pd.to_datetime(["2024-01-03 10:00", "2024-01-04 15:00"]).tz_localize(
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


class _BuyFirstThenSellAvailable(Strategy):
    """首根 bar 买 100 股，之后有可用持仓就全部卖出；记录每根 bar 的可用持仓.

    首根 bar 下的单在第二根 bar 成交，存档前已处理完（``_known_orders`` 为空，
    策略实例可 pickle）。模块顶层定义以便 pickle。
    """

    def __init__(self) -> None:
        """初始化."""
        super().__init__()
        self.n = 0
        self.available: List[tuple] = []

    def on_bar(self, bar: Bar) -> None:
        """首根买入，之后卖出全部可用."""
        self.n += 1
        local = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert("Asia/Shanghai")
        avail = float(self.get_available_position(bar.symbol))
        self.available.append((local.strftime("%Y-%m-%d %H:%M"), avail))
        if self.n == 1:
            self.buy(bar.symbol, 100)
        elif avail > 0:
            self.sell(bar.symbol, avail)


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


def test_checkpoint_roundtrips_trading_day_progress() -> None:
    """存档保存当前交易日与已结算交易日."""
    from akquant.akquant import Engine

    r = run_backtest(
        data=_intraday_frame(
            ["2024-01-04 10:00", "2024-01-05 10:00", "2024-01-05 15:00"]
        ),
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
        data=_intraday_frame(
            [
                "2024-01-03 10:00",
                "2024-01-04 10:00",
                "2024-01-05 10:00",
                "2024-01-05 15:00",
            ]
        ),
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
        data=_intraday_frame(
            ["2024-01-08 10:00", "2024-01-09 10:00", "2024-01-09 15:00"]
        ),
        symbols=["600000.SH"],
        show_progress=False,
    )
    # 周一续跑时持仓（周三买入的 100 股）跨周末仍在。
    # 续跑结果的 positions 只含续跑段的快照：周一、周二各一条日终快照（若重复
    # 结算周五会多出一条）。最后交易日（周二）末根 bar 在 15:00，已在收盘结算，
    # get_results 不再补期末快照（类别 (b)）。
    times = [t.strftime("%Y-%m-%d %H:%M") for t in _snapshot_times(part2)]
    assert times == ["2024-01-08 15:00", "2024-01-09 15:00"]
    assert (part2.positions["600000.SH"] == 100.0).all()


def test_settled_last_day_has_no_duplicate_terminal_snapshot() -> None:
    """最后交易日已在收盘结算：期末不再补快照，快照时间严格递增."""
    r = run_backtest(
        data=_intraday_frame(
            [
                "2024-01-03 10:00",
                "2024-01-04 10:00",
                "2024-01-05 10:00",
                "2024-01-05 15:00",
            ]
        ),
        strategy=_HoldOnce,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    times = [t.strftime("%Y-%m-%d %H:%M") for t in _snapshot_times(r)]
    assert times == ["2024-01-03 15:00", "2024-01-04 15:00", "2024-01-05 15:00"]
    # 收盘快照即期末状态：首根 bar 买入的 100 股在最后一条快照里
    assert r.positions["600000.SH"].iloc[-1] == 100.0
    # 净值曲线仍逐事件记点，期末点留在末根 bar 时刻（不新增点）
    last_equity_ts = r.equity_curve.index[-1].tz_convert("Asia/Shanghai")
    assert last_equity_ts.strftime("%Y-%m-%d %H:%M") == "2024-01-05 15:00"
    assert len(r.equity_curve) == 4


def test_resume_same_day_does_not_reopen_or_resettle(tmp_path: Any) -> None:
    """数据盘中结束不结算当日；同日续跑时当日买入仍不可卖，收盘只结算一次.

    第一段数据止于 10:00（当日未收盘），旧实现在数据耗尽后仍触发 15:00 结算、
    解冻 T+1，续跑段 13:01 就把当日 09:32 买入的股票卖掉了。
    """
    part1 = run_backtest(
        data=_intraday_frame(
            ["2024-01-04 09:31", "2024-01-04 09:32", "2024-01-04 10:00"]
        ),
        strategy=_BuyFirstThenSellAvailable,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        t_plus_one=True,
    )
    assert part1.engine is not None
    # 盘中结束：当日未结算，期末快照记在末根 bar（与旧收尾逻辑一致）
    assert part1.engine.current_trading_day == dt.date(2024, 1, 4)
    assert part1.engine.last_settled_trading_day is None
    times1 = [t.strftime("%Y-%m-%d %H:%M") for t in _snapshot_times(part1)]
    assert times1 == ["2024-01-04 10:00"]
    assert part1.positions["600000.SH"].iloc[-1] == 100.0

    ckpt = tmp_path / "c.pkl"
    save_checkpoint(part1.engine, part1.strategy, str(ckpt))
    part2 = run_from_checkpoint(
        checkpoint_path=str(ckpt),
        data=_intraday_frame(
            ["2024-01-04 13:00", "2024-01-04 13:01", "2024-01-04 15:00"]
        ),
        symbols=["600000.SH"],
        show_progress=False,
    )
    assert part2.engine is not None
    assert part2.engine.current_trading_day == dt.date(2024, 1, 4)
    assert part2.engine.last_settled_trading_day == dt.date(2024, 1, 4)
    strategy = part2.strategy
    assert strategy is not None
    # 当日买入的 100 股在当日任何 bar 上都不可卖（T+1）
    assert all(avail == 0.0 for _, avail in strategy.available), strategy.available
    fills = part2.executions_df
    assert list(fills["side"].astype(str).str.lower()) == ["buy"], fills
    # 当日只在续跑段 15:00 结算一次
    closes = [
        t
        for t in _snapshot_times(part1) + _snapshot_times(part2)
        if t.strftime("%Y-%m-%d %H:%M") == "2024-01-04 15:00"
    ]
    assert len(closes) == 1


class _BuyFirstSellAvailableRebuy(_BuyFirstThenSellAvailable):
    """在 _BuyFirstThenSellAvailable 基础上，于 2024-01-05 09:32 再买 100 股.

    用来区分续跑段 T+1 是否仍开启：再买的 100 股当日应不可卖。
    """

    def on_bar(self, bar: Bar) -> None:
        """先按父类处理，再在指定时刻再买一次."""
        super().on_bar(bar)
        if self.available[-1][0] == "2024-01-05 09:32":
            self.buy(bar.symbol, 100)


def _frozen_t_plus_one_checkpoint(tmp_path: Any) -> Path:
    """T+1 下当日买入、数据盘中结束（当日未结算）时存档：100 股冻结."""
    part1 = run_backtest(
        data=_intraday_frame(
            ["2024-01-04 09:31", "2024-01-04 09:32", "2024-01-04 10:00"]
        ),
        strategy=_BuyFirstSellAvailableRebuy,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        t_plus_one=True,
    )
    assert part1.engine is not None
    portfolio = part1.engine.portfolio
    assert portfolio.positions.get("600000.SH") == 100.0
    assert portfolio.available_positions.get("600000.SH", 0.0) == 0.0
    ckpt = tmp_path / "c.pkl"
    save_checkpoint(part1.engine, part1.strategy, str(ckpt))
    return Path(ckpt)


def test_resume_without_t_plus_one_unfreezes_next_day(tmp_path: Any) -> None:
    """续跑不传 t_plus_one、存档未带 config=：存档前冻结的持仓次日可卖.

    旧实现续跑时退回 SimpleMarket，其日终结算不解冻，冻结持仓永不可卖。
    现在按快照中的冻结持仓推断 T+1 仍开启：当日仍不可卖，次日解冻；
    续跑段次日 09:32 再买的 100 股当日（10:00）不可卖。
    """
    ckpt = _frozen_t_plus_one_checkpoint(tmp_path)
    part2 = run_from_checkpoint(
        checkpoint_path=str(ckpt),
        data=_intraday_frame(
            [
                "2024-01-04 13:00",
                "2024-01-05 09:31",
                "2024-01-05 09:32",
                "2024-01-05 10:00",
            ]
        ),
        symbols=["600000.SH"],
        show_progress=False,
    )
    strategy = part2.strategy
    assert strategy is not None
    assert strategy.available[-4:] == [
        ("2024-01-04 13:00", 0.0),
        ("2024-01-05 09:31", 100.0),
        ("2024-01-05 09:32", 0.0),
        ("2024-01-05 10:00", 0.0),
    ], strategy.available
    sides = list(part2.executions_df["side"].astype(str).str.lower())
    # 成交记录含存档前第一段的买入
    assert sides == ["buy", "sell", "buy"], part2.executions_df


def test_resume_with_t_plus_one_false_still_unfreezes(tmp_path: Any) -> None:
    """续跑显式 t_plus_one=False：已冻结持仓不依赖当前开关，次日照样解冻."""
    ckpt = _frozen_t_plus_one_checkpoint(tmp_path)
    part2 = run_from_checkpoint(
        checkpoint_path=str(ckpt),
        data=_intraday_frame(["2024-01-05 09:31", "2024-01-05 09:32"]),
        symbols=["600000.SH"],
        show_progress=False,
        t_plus_one=False,
    )
    strategy = part2.strategy
    assert strategy is not None
    assert strategy.available[-2:] == [
        ("2024-01-05 09:31", 100.0),
        ("2024-01-05 09:32", 0.0),
    ], strategy.available
    sides = list(part2.executions_df["side"].astype(str).str.lower())
    assert sides == ["buy", "sell"], part2.executions_df


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


# _run_live_replay 截获的网关线程（按线程名），供 on_start 等待推送线程结束。
_GATEWAY_THREADS: Dict[str, Any] = {}
# 每次 on_start 是否找到并等到了 replay 推送线程（_run_live_replay 校验）。
_REPLAY_THREAD_JOINED: List[bool] = []
# _LiveDayOrderProbe.on_order 收到的（订单号, 状态名）序列。
_LIVE_ORDER_EVENTS: List[Tuple[str, str]] = []


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
        # 线程名由 live/_runner.py 按 f"{broker}-market" 命名；改名后这里找不到线程，
        # 等待静默失效、测试变成竞态。记录是否找到并等到，由 _run_live_replay 断言。
        # 不用 threading.enumerate()：推送线程常在 on_start 之前就已结束并退出枚举，
        # 那样“找不到”与“已结束”无法区分。
        thread = _GATEWAY_THREADS.get("replay-market")
        if thread is not None:
            thread.join(timeout=10)
        _REPLAY_THREAD_JOINED.append(thread is not None and not thread.is_alive())

    def on_bar(self, bar: Bar) -> None:
        assert self.ctx is not None
        self.active_at[_local_minute(int(bar.timestamp))] = len(self.ctx.active_orders)
        if not self.done:
            self.done = True
            self.buy("600000", 100, price=1.0, time_in_force=aq.TimeInForce.Day)

    def on_order(self, order: Any) -> None:
        _LIVE_ORDER_EVENTS.append((order.id, str(order.status).split(".")[-1]))


def _run_live_replay(
    monkeypatch: pytest.MonkeyPatch,
    bars: Any,
    strategy_cls: Any = None,
    freq: str = "1min",
    ticks: Any = None,
    **kwargs: Any,
) -> Any:
    """run_live(broker="replay") 跑完后返回 runner，供读取引擎结果.

    run_live 不返回结果，订单终态从引擎结果里读；on_order 收到的状态序列记在
    _LIVE_ORDER_EVENTS。
    """
    from akquant import run_live
    from akquant.live import _facade
    from akquant.live._gateway_setup import start_gateway_thread

    captured: List[Any] = []

    class _Recording(_facade.LiveRunner):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kw: Any) -> None:
            super().__init__(*args, **kw)
            captured.append(self)

        def _start_gateway_thread(self, target: Any, name: str) -> None:
            _GATEWAY_THREADS[name] = start_gateway_thread(target, name)

    monkeypatch.setattr(_facade, "LiveRunner", _Recording)
    _GATEWAY_THREADS.clear()
    _REPLAY_THREAD_JOINED.clear()
    _LIVE_ORDER_EVENTS.clear()
    gateway_options: Dict[str, Any] = {"bars": bars, "freq": freq}
    if ticks is not None:
        gateway_options["ticks"] = ticks
    run_live(
        strategy_cls=strategy_cls or _LiveDayOrderProbe,
        instruments=[_live_stock_instrument()],
        broker="replay",
        trading_mode="paper",
        gateway_options=gateway_options,
        cash=1e6,
        show_progress=False,
        duration="30s",
        **kwargs,
    )
    assert len(captured) == 1
    assert _REPLAY_THREAD_JOINED == [True], (
        "未找到或未等到名为 replay-market 的推送线程（线程命名见 live/_runner.py）"
    )
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
    # 日终过期同样回调 on_order（Expired），且只回调一次
    statuses = [status for _, status in _LIVE_ORDER_EVENTS]
    assert statuses[-1] == "Expired", _LIVE_ORDER_EVENTS
    assert statuses.count("Expired") == 1, _LIVE_ORDER_EVENTS


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


_LIVE_DAY_PROBES: List[Any] = []


class _LiveTradingDayProbe(_LiveDayOrderProbe):
    """记录每根 bar 的（本地时刻, 交易日, 时段）."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[Tuple[str, Any, str]] = []
        _LIVE_DAY_PROBES.append(self)

    def on_bar(self, bar: Bar) -> None:
        assert self.ctx is not None
        self.rows.append(
            (
                _local_minute(int(bar.timestamp)),
                self.trading_day,
                str(self.ctx.session).split(".")[-1],
            )
        )
        super().on_bar(bar)


def test_live_replay_forwards_custom_session_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_live 的自定义时段表下发到引擎：时段、收盘时刻与交易日划分按自定义表.

    10:00-11:00 一段时交易日在 11:00 收盘，14:00 的 bar 处于闭市、属下一交易日。
    """
    _LIVE_DAY_PROBES.clear()
    runner = _run_live_replay(
        monkeypatch,
        _local_bars(["2024-01-02 10:15", "2024-01-02 10:30", "2024-01-02 14:00"]),
        strategy_cls=_LiveTradingDayProbe,
        sessions={"default": [("10:00", "11:00", "continuous")]},
        trading_calendar=["2023-12-29", "2024-01-02", "2024-01-03"],
    )
    assert len(_LIVE_DAY_PROBES) == 1
    assert _LIVE_DAY_PROBES[0].rows == [
        ("2024-01-02 10:15", dt.date(2024, 1, 2), "Continuous"),
        ("2024-01-02 10:30", dt.date(2024, 1, 2), "Continuous"),
        ("2024-01-02 14:00", dt.date(2024, 1, 3), "Closed"),
    ]
    # 10:15 下的 Day 单（非当日最后切片）在自定义收盘 11:00 过期（默认为 15:00）
    assert _expired_at(runner) == ["2024-01-02 11:00"]


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


_GRACE_PROBES: List[Any] = []


class _LiveGraceProbe(_LiveDayOrderProbe):
    """记录行情回调的（本地时刻到秒, 交易日, 时段）与生命周期钩子时刻."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[Tuple[str, Any, str]] = []
        self.hooks: List[Tuple[str, str]] = []
        _GRACE_PROBES.append(self)

    def _row(self, ts_ns: int) -> None:
        assert self.ctx is not None
        local = pd.Timestamp(ts_ns, tz="UTC").tz_convert("Asia/Shanghai")
        self.rows.append(
            (
                local.strftime("%Y-%m-%d %H:%M:%S"),
                self.trading_day,
                str(self.ctx.session).split(".")[-1],
            )
        )
        self.hooks.append(("on_data", local.strftime("%Y-%m-%d %H:%M:%S")))

    def _hook(self, name: str) -> None:
        now = self.now
        assert now is not None
        self.hooks.append((name, now.strftime("%Y-%m-%d %H:%M:%S")))

    def on_bar(self, bar: Bar) -> None:
        self._row(int(bar.timestamp))
        super().on_bar(bar)

    def on_tick(self, tick: Any) -> None:
        self._row(int(tick.timestamp))

    def on_pre_open(self, event: Dict[str, Any]) -> None:
        self._hook("on_pre_open")

    def on_after_trading(self, trading_date: dt.date, timestamp: int) -> None:
        self._hook("on_after_trading")


def test_live_replay_post_close_tick_inside_grace_belongs_to_same_day(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """收盘后 day_close_delay 内到达的行情仍归当日，不提前开下一交易日."""
    from akquant import Tick

    _GRACE_PROBES.clear()
    tick_ts = pd.Timestamp("2024-01-02 15:00:20", tz="Asia/Shanghai").value
    runner = _run_live_replay(
        monkeypatch,
        _local_bars(["2024-01-02 14:59", "2024-01-03 09:31"]),
        strategy_cls=_LiveGraceProbe,
        ticks=[Tick(timestamp=int(tick_ts), price=10.0, volume=100.0, symbol="600000")],
        trading_calendar=["2024-01-02", "2024-01-03"],
        day_close_delay=60,
    )
    assert len(_GRACE_PROBES) == 1
    probe = _GRACE_PROBES[0]
    assert probe.rows == [
        ("2024-01-02 14:59:00", dt.date(2024, 1, 2), "ClosingAuction"),
        ("2024-01-02 15:00:20", dt.date(2024, 1, 2), "Closed"),
        ("2024-01-03 09:31:00", dt.date(2024, 1, 3), "Continuous"),
    ], probe.rows
    after = [at for name, at in probe.hooks if name == "on_after_trading"]
    assert after == ["2024-01-02 15:01:00"], probe.hooks
    names = [name for name, _ in probe.hooks]
    pre_open_0103 = [
        i
        for i, (name, at) in enumerate(probe.hooks)
        if name == "on_pre_open" and at.startswith("2024-01-03")
    ]
    bar_0103 = probe.hooks.index(("on_data", "2024-01-03 09:31:00"))
    assert len(pre_open_0103) == 1 and pre_open_0103[0] < bar_0103, names
    # 结算快照记在收盘时刻、只一次
    assert _snapshot_minutes(runner).count("2024-01-02 15:00") == 1
    # 宽限期内的行情不触发“闭市时段”告警
    closed_warnings = [
        r.getMessage() for r in caplog.records if "闭市时段" in r.getMessage()
    ]
    assert not closed_warnings, closed_warnings


class _BuyFirstBarProbe(_LiveDayOrderProbe):
    """首根 bar 下一张市价买单（下一根 bar 成交），使权益随价格变化."""

    def on_bar(self, bar: Bar) -> None:
        if not self.done:
            self.done = True
            self.buy("600000", 100)


def _daily_equity(runner: Any) -> Any:
    from akquant.analysis.benchmark import daily_last_by_trading_day

    raw = runner.engine.get_results()
    stamps = [int(ts) for ts, _ in raw.equity_curve]
    series = pd.Series(
        [float(v) for _, v in raw.equity_curve],
        index=pd.to_datetime(stamps, utc=True),
    )
    ends = [(day, int(end)) for day, end in raw.trading_day_ends]
    return stamps, ends, daily_last_by_trading_day(series, ends, "Asia/Shanghai")


def test_live_replay_grace_ticks_do_not_change_daily_equity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """宽限期内的行情不产生权益点：日权益与去掉这些行情时相同."""
    from akquant import Tick

    bars = _local_bars(["2024-01-02 14:58", "2024-01-02 14:59", "2024-01-03 09:31"])
    for bar, close in zip(bars, (10.0, 10.0, 11.0)):
        bar.close = close
    tick_ts = int(pd.Timestamp("2024-01-02 15:00:20", tz="Asia/Shanghai").value)
    tick = Tick(timestamp=tick_ts, price=20.0, volume=100.0, symbol="600000")
    common: Dict[str, Any] = {
        "strategy_cls": _BuyFirstBarProbe,
        "trading_calendar": ["2024-01-02", "2024-01-03"],
        "day_close_delay": 60,
    }
    with_grace = _run_live_replay(monkeypatch, bars, ticks=[tick], **common)
    without = _run_live_replay(monkeypatch, bars, **common)
    stamps, ends, daily = _daily_equity(with_grace)
    _, ends_plain, daily_plain = _daily_equity(without)
    close_ns = int(pd.Timestamp("2024-01-02 15:00", tz="Asia/Shanghai").value)
    # 权益曲线在 (收盘, 收盘 + 宽限] 内无点；trading_day_ends 为收盘时刻
    assert not [t for t in stamps if close_ns < t <= close_ns + 60 * 10**9], stamps
    assert ends == ends_plain
    assert ends[0] == (dt.date(2024, 1, 2), close_ns)
    assert len(with_grace.engine.get_results().executions) == 1
    pd.testing.assert_series_equal(daily, daily_plain)


class _BuyStockOnce(Strategy):
    """记录每根 bar 的本地时间；首根 600000.SH bar 下一张 NextOpen 市价单."""

    def __init__(self) -> None:
        """初始化."""
        super().__init__()
        self.done = False
        self.seen: List[tuple] = []

    def on_bar(self, bar: Bar) -> None:
        """记录并下单."""
        local = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert("Asia/Shanghai")
        self.seen.append((bar.symbol, local.strftime("%Y-%m-%d %H:%M")))
        if bar.symbol == "600000.SH" and not self.done:
            self.done = True
            self.buy("600000.SH", 100)


def test_date_only_mixed_sessions_rewrite_to_session_close() -> None:
    """股票 + 国债期货（15:15 收盘）的只有日期数据：同日两标的都改写到 15:15.

    按标的收盘改写时股票在 15:00、国债期货在 15:15，同一天拆成两个切片；股票
    首日下的 NextOpen 单在 15:15 的切片里找不到股票行情，被拒为
    "Missing market data"。会话级收盘改写后同日只有一个切片，次日正常成交。
    """
    days = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    df = pd.concat(
        [_daily_frame(days, "600000.SH"), _daily_frame(days, "T2403")]
    ).sort_values("timestamp")
    result = run_backtest(
        data=df,
        strategy=_BuyStockOnce,
        symbols=["600000.SH", "T2403"],
        initial_cash=1e6,
        show_progress=False,
        sessions={"T": "CN_FUTURES_CFFEX_BOND_DAY"},
    )
    seen = _seen(result)
    assert {t for _, t in seen} == {f"{d} 15:15" for d in days}
    assert sorted(seen) == sorted(
        (s, f"{d} 15:15") for d in days for s in ("600000.SH", "T2403")
    )
    orders = result.orders_df
    assert list(orders["status"].astype(str).str.lower()) == ["filled"], orders
    reasons = orders["reject_reason"].fillna("").astype(str)
    assert not reasons.str.contains("Missing market data").any()
    fills = result.executions_df
    assert len(fills) == 1
    fill_local = pd.Timestamp(fills.iloc[0]["timestamp"])
    if fill_local.tzinfo is None:
        fill_local = fill_local.tz_localize("UTC")
    assert fill_local.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") == "2024-01-03"


def test_session_resolver_session_day_end_uses_active_symbols() -> None:
    """SessionResolver.set_active 后 session_day_end 取在用时段表收盘的最大值."""
    resolver = aq.SessionResolver(sessions={"T": "CN_FUTURES_CFFEX_BOND_DAY"})
    day = dt.date(2024, 1, 2)
    stock_close = pd.Timestamp("2024-01-02 15:00", tz="Asia/Shanghai").value
    bond_close = pd.Timestamp("2024-01-02 15:15", tz="Asia/Shanghai").value
    assert resolver.session_day_end(day) == stock_close
    resolver.set_active(
        [("600000.SH", aq.AssetType.Stock), ("T2403", aq.AssetType.Futures)]
    )
    assert resolver.session_day_end(day) == bond_close
    resolver.set_active([("600000.SH", aq.AssetType.Stock)])
    assert resolver.session_day_end(day) == stock_close


class _PreOpenTimerBuyer(Strategy):
    """每日 09:35 定时器以当前收盘价买 30 股，每根 bar 也买 30 股."""

    def on_start(self) -> None:
        """登记盘前定时器."""
        self.schedule_daily("09:35", "pre")

    def on_timer(self, payload: str) -> None:
        """定时买入."""
        self.buy("RISK", 30, fill_mode=aq.CurrentClose())

    def on_bar(self, bar: Bar) -> None:
        """每根 bar 买入."""
        self.buy(bar.symbol, 30)


def test_daily_loss_resets_for_timer_order_before_first_bar_of_day() -> None:
    """01-08 触发日亏损；01-09 09:35（当日首根 bar 之前）的定时器单属新交易日.

    当日尚未开日（首个行情事件 10:00 才开日），旧实现风控交易日仍取上一交易日
    01-08，09:35 的单被按 01-08 的亏损拒绝。
    """
    result = run_backtest(
        data=_bars_at(
            [
                "2024-01-08 09:31",
                "2024-01-08 10:00",
                "2024-01-08 10:01",
                "2024-01-09 10:00",
            ],
            [100.0, 100.0, 50.0, 50.0],
            "RISK",
        ),
        strategy=_PreOpenTimerBuyer,
        symbols="RISK",
        initial_cash=100000.0,
        show_progress=False,
        fill_policy=aq.CurrentClose(),
        lot_size=1,
        risk_config=aq.config.RiskConfig(max_daily_loss=0.01),
    )
    orders = result.orders_df
    local = pd.to_datetime(orders["created_at"])
    if local.dt.tz is None:
        local = local.dt.tz_localize("UTC")
    stamps = local.dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d %H:%M")
    reasons = orders["reject_reason"].fillna("").astype(str)
    by_time = dict(zip(stamps, reasons))
    view = orders[["created_at", "status", "reject_reason"]]
    # 01-08 10:01 的单在亏损当日被拒（规则本身仍生效）
    assert "Daily loss" in by_time["2024-01-08 10:01"], view
    # 01-09 09:35 的定时器单属 01-09，不因 01-08 的亏损被拒
    assert "Daily loss" not in by_time["2024-01-09 09:35"], view


@pytest.mark.parametrize(
    ("kwargs", "marker"),
    [
        (
            {"strategy_id": "alpha", "strategy_max_daily_loss": {"alpha": 100.0}},
            "daily loss",
        ),
        (
            {"portfolio_risk_budget": 9000.0, "risk_budget_reset_daily": True},
            "risk budget",
        ),
    ],
    ids=["strategy_daily_loss", "risk_budget"],
)
def test_strategy_risk_resets_for_timer_order_before_first_bar_of_day(
    kwargs: Dict[str, Any], marker: str
) -> None:
    """策略级日亏损与风险预算同样按 now 所属交易日重置（盘前定时器单不被拒）."""
    result = run_backtest(
        data=_bars_at(
            [
                "2024-01-08 09:31",
                "2024-01-08 10:00",
                "2024-01-08 10:01",
                "2024-01-09 10:00",
            ],
            [100.0, 100.0, 50.0, 50.0],
            "RISK",
        ),
        strategy=_PreOpenTimerBuyer,
        symbols="RISK",
        initial_cash=100000.0,
        show_progress=False,
        fill_policy=aq.CurrentClose(),
        lot_size=1,
        **kwargs,
    )
    orders = result.orders_df
    local = pd.to_datetime(orders["created_at"])
    if local.dt.tz is None:
        local = local.dt.tz_localize("UTC")
    stamps = local.dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d %H:%M")
    by_time = dict(zip(stamps, orders["reject_reason"].fillna("").astype(str)))
    view = orders[["created_at", "status", "reject_reason"]]
    assert marker in by_time["2024-01-08 10:01"], view
    assert marker not in by_time["2024-01-09 09:35"], view


_HOLIDAY_CALENDAR = ["2024-01-05", "2024-01-08", "2024-01-10", "2024-01-11"]


def _save_part1(tmp_path: Path, bars: List[Bar], symbol: str, **kwargs: Any) -> str:
    """跑第一段并存档，返回存档路径."""
    part1 = run_backtest(
        data=bars,
        strategy=_Recorder,
        symbols=[symbol],
        initial_cash=1e6,
        show_progress=False,
        **kwargs,
    )
    assert part1.engine is not None
    ckpt = tmp_path / "c.pkl"
    save_checkpoint(part1.engine, part1.strategy, str(ckpt))
    return str(ckpt)


def test_resume_reuses_saved_sessions_and_calendar(tmp_path: Path) -> None:
    """续跑未传 sessions/trading_calendar 时沿用存档：节前夜盘仍归节后交易日.

    01-09 为假日：周一 21:30 的夜盘属 01-10。旧实现续跑不恢复配置，退回按工作日
    推算，把它划进 01-09。
    """
    ckpt = _save_part1(
        tmp_path,
        _bars_at(["2024-01-08 10:00", "2024-01-08 15:00"], [100.0, 100.0], "au2406"),
        "au2406",
        sessions={"au": "CN_FUTURES_NIGHT_0230"},
        trading_calendar=_HOLIDAY_CALENDAR,
    )
    part2 = run_from_checkpoint(
        checkpoint_path=ckpt,
        data=_bars_at(["2024-01-08 21:30"], [100.0], "au2406"),
        symbols=["au2406"],
        show_progress=False,
    )
    assert part2.engine is not None
    assert part2.engine.last_settled_trading_day == dt.date(2024, 1, 8)
    assert part2.engine.current_trading_day == dt.date(2024, 1, 10)


def test_resume_reuses_saved_sessions_close_time(tmp_path: Path) -> None:
    """存档的时段表（国债期货 15:15 收盘）续跑后仍生效：15:10 的 bar 属当日."""
    ckpt = _save_part1(
        tmp_path,
        _bars_at(["2024-01-08 10:00", "2024-01-08 15:15"], [100.0, 100.0], "T2403"),
        "T2403",
        sessions={"T": "CN_FUTURES_CFFEX_BOND_DAY"},
    )
    part2 = run_from_checkpoint(
        checkpoint_path=ckpt,
        data=_bars_at(["2024-01-09 15:10"], [100.0], "T2403"),
        symbols=["T2403"],
        show_progress=False,
    )
    assert part2.engine is not None
    assert part2.engine.last_settled_trading_day == dt.date(2024, 1, 8)
    assert part2.engine.current_trading_day == dt.date(2024, 1, 9)


def test_resume_passed_calendar_overrides_saved_with_one_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """续跑传入与存档不一致的 trading_calendar：以传入为准，告警一次."""
    ckpt = _save_part1(
        tmp_path,
        _bars_at(["2024-01-08 10:00", "2024-01-08 15:00"], [100.0, 100.0], "au2406"),
        "au2406",
        sessions={"au": "CN_FUTURES_NIGHT_0230"},
        trading_calendar=_HOLIDAY_CALENDAR,
    )
    with caplog.at_level("WARNING", logger="akquant"):
        part2 = run_from_checkpoint(
            checkpoint_path=ckpt,
            data=_bars_at(["2024-01-08 21:30"], [100.0], "au2406"),
            symbols=["au2406"],
            show_progress=False,
            trading_calendar=["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"],
        )
    assert part2.engine is not None
    assert part2.engine.current_trading_day == dt.date(2024, 1, 9)
    warnings = [r for r in caplog.records if "与存档不一致" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    assert "trading_calendar" in warnings[0].getMessage()


@pytest.mark.parametrize(
    "calendar",
    [None, ["2024-01-05", "2024-01-08"]],
    ids=["weekdays", "explicit"],
)
def test_date_only_non_trading_date_warns_once(
    calendar: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """只有日期的数据含非交易日（周六）：改写时告警一次，多个标的也只告警一次."""
    days = ["2024-01-05", "2024-01-06", "2024-01-08"]
    df = pd.concat(
        [_daily_frame(days, "600000.SH"), _daily_frame(days, "000001.SZ")]
    ).sort_values("timestamp")
    kwargs: Dict[str, Any] = {}
    if calendar is not None:
        kwargs["trading_calendar"] = calendar
    with caplog.at_level("WARNING", logger="akquant"):
        run_backtest(
            data=df,
            strategy=_Recorder,
            symbols=["600000.SH", "000001.SZ"],
            initial_cash=1e5,
            show_progress=False,
            **kwargs,
        )
    warnings = [r for r in caplog.records if "非交易日" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    assert "2024-01-06" in warnings[0].getMessage()


def test_date_only_trading_dates_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """只有日期的数据全在交易日上：不告警."""
    with caplog.at_level("WARNING", logger="akquant"):
        run_backtest(
            data=_daily_frame(["2024-01-05", "2024-01-08"]),
            strategy=_Recorder,
            symbols=["600000.SH"],
            initial_cash=1e5,
            show_progress=False,
        )
    assert not [r for r in caplog.records if "非交易日" in r.getMessage()]


class _ExpiryProbe(Strategy):
    """首根 bar 下一张够不着的 Day 限价单，记录 on_order 收到的状态序列."""

    def __init__(self) -> None:
        super().__init__()
        self.done = False
        self.events: List[Tuple[str, str]] = []

    def on_bar(self, bar: Bar) -> None:
        if not self.done:
            self.done = True
            self.buy(bar.symbol, 100, price=1.0, time_in_force=aq.TimeInForce.Day)

    def on_order(self, order: Any) -> None:
        self.events.append((order.id, str(order.status).split(".")[-1]))


def test_day_order_expiry_triggers_on_order() -> None:
    """日终结算过期的 Day 单回调 on_order（Expired），只回调一次."""
    r = run_backtest(
        strategy=_ExpiryProbe,
        data=_daily_frame(["2024-01-02", "2024-01-03", "2024-01-04"]),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r.strategy is not None
    statuses = [s for _, s in r.strategy.events]
    assert statuses[-1] == "Expired"
    assert statuses.count("Expired") == 1


def test_day_order_expiring_on_last_day_still_triggers_on_order() -> None:
    """最后一个交易日收盘过期（之后没有策略事件）也要回调."""
    r = run_backtest(
        strategy=_ExpiryProbe,
        data=_daily_frame(["2024-01-02", "2024-01-03"]),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r.strategy is not None
    assert [s for _, s in r.strategy.events][-1] == "Expired"


def test_day_order_expiry_emits_order_stream_event() -> None:
    """日终过期的 Day 单发 order 流事件（status=Expired），带收盘时刻."""
    events: List[Any] = []
    run_backtest(
        strategy=_ExpiryProbe,
        data=_daily_frame(["2024-01-02", "2024-01-03", "2024-01-04"]),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        on_event=events.append,
    )
    expired = [
        e
        for e in events
        if e["event_type"] == "order" and e["payload"].get("status") == "Expired"
    ]
    assert len(expired) == 1, expired
    assert expired[0]["symbol"] == "600000.SH"
    assert expired[0]["payload"]["order_id"]
    local = pd.Timestamp(int(expired[0]["ts"]), tz="UTC").tz_convert("Asia/Shanghai")
    assert local == pd.Timestamp("2024-01-03 15:00", tz="Asia/Shanghai")


class _BorrowOnce(Strategy):
    """首根 bar 融资买入 150 股（借入约 5000），之后不动。模块顶层定义以便 pickle."""

    def __init__(self) -> None:
        super().__init__()
        self.ordered = False

    def on_bar(self, bar: Bar) -> None:
        if not self.ordered:
            self.ordered = True
            self.buy(bar.symbol, 150)


_NY_DAYS = ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]


def _ny_frame(dates: List[str]) -> pd.DataFrame:
    """纽约时区 10:00 的 AAPL 分钟级时刻数据（价格恒 100）."""
    idx = pd.to_datetime([f"{d} 10:00" for d in dates]).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "AAPL",
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1e6,
        }
    )


def _margin_ny_kwargs() -> Dict[str, Any]:
    """融资账户（高利率便于观察利息）+ 纽约时区的回测参数."""
    from akquant.config import RiskConfig

    return dict(
        symbols="AAPL",
        show_progress=False,
        fill_policy=aq.CurrentClose(),
        lot_size=1,
        timezone="America/New_York",
        risk_config=RiskConfig(
            account_mode="margin",
            initial_margin_ratio=0.5,
            financing_rate_annual=36.5,
            borrow_rate_annual=0.0,
            allow_force_liquidation=False,
            max_position_size=1_000_000.0,
        ),
    )


def _save_margin_ny_checkpoint(
    tmp_path: Any, *, with_config: bool = True
) -> Tuple[Any, str]:
    """跑前三天并存档；返回（连续跑五天的结果，存档路径）."""
    kw = _margin_ny_kwargs()
    full = run_backtest(
        data=_ny_frame(_NY_DAYS), strategy=_BorrowOnce, initial_cash=10000.0, **kw
    )
    part1 = run_backtest(
        data=_ny_frame(_NY_DAYS[:3]), strategy=_BorrowOnce, initial_cash=10000.0, **kw
    )
    ckpt = str(tmp_path / "c.pkl")
    assert part1.engine is not None
    save_checkpoint(
        part1.engine, part1.strategy, ckpt, config=part1 if with_config else None
    )
    return full, ckpt


def _resume_warnings(caplog: pytest.LogCaptureFixture) -> List[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelname == "WARNING"
        and r.getMessage().startswith("run_from_checkpoint 传入的")
        and "不一致" in r.getMessage()
    ]


def test_resume_restores_risk_config_and_timezone(tmp_path: Any) -> None:
    """融资账户 + 纽约时区：续跑不传 risk_config/timezone 时沿用存档值.

    利息与连续跑一致。
    """
    full, ckpt = _save_margin_ny_checkpoint(tmp_path)
    part2 = run_from_checkpoint(
        checkpoint_path=ckpt,
        data=_ny_frame(_NY_DAYS[3:]),
        symbols="AAPL",
        show_progress=False,
    )
    assert part2.engine is not None and full.engine is not None
    assert float(part2.engine.portfolio.cash) == pytest.approx(
        float(full.engine.portfolio.cash), abs=1e-6
    )
    assert part2.engine.current_trading_day == dt.date(2024, 1, 9)


def test_chained_resume_keeps_risk_config_and_timezone(tmp_path: Any) -> None:
    """存档—续跑—再存档—再续跑：续跑结果携带已解析配置，期末现金与连续跑一致."""
    kw = _margin_ny_kwargs()
    full = run_backtest(
        data=_ny_frame(_NY_DAYS), strategy=_BorrowOnce, initial_cash=10000.0, **kw
    )
    part1 = run_backtest(
        data=_ny_frame(_NY_DAYS[:2]), strategy=_BorrowOnce, initial_cash=10000.0, **kw
    )
    ckpt1 = str(tmp_path / "c1.pkl")
    assert part1.engine is not None
    save_checkpoint(part1.engine, part1.strategy, ckpt1, config=part1)
    part2 = run_from_checkpoint(
        checkpoint_path=ckpt1,
        data=_ny_frame(_NY_DAYS[2:4]),
        symbols="AAPL",
        show_progress=False,
    )
    resolved = part2.resolved_config
    assert resolved is not None
    assert resolved["timezone"] == "America/New_York"
    assert resolved["risk_config"]["account_mode"] == "margin"
    ckpt2 = str(tmp_path / "c2.pkl")
    assert part2.engine is not None
    save_checkpoint(part2.engine, part2.strategy, ckpt2, config=part2)
    part3 = run_from_checkpoint(
        checkpoint_path=ckpt2,
        data=_ny_frame(_NY_DAYS[4:]),
        symbols="AAPL",
        show_progress=False,
    )
    assert part3.engine is not None and full.engine is not None
    assert float(part3.engine.portfolio.cash) == pytest.approx(
        float(full.engine.portfolio.cash), abs=1e-6
    )
    assert part3.engine.current_trading_day == dt.date(2024, 1, 9)


def test_resume_old_archive_without_config_keeps_old_behavior(
    tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """存档未传 config=：续跑不报错、不告警，退回默认时区与默认（现金）风控."""
    _, ckpt = _save_margin_ny_checkpoint(tmp_path, with_config=False)
    with caplog.at_level("WARNING", logger="akquant"):
        part2 = run_from_checkpoint(
            checkpoint_path=ckpt,
            data=_ny_frame(_NY_DAYS[3:]),
            symbols="AAPL",
            show_progress=False,
        )
    assert _resume_warnings(caplog) == []
    assert part2.engine is not None
    assert part2.engine.risk_manager.config.account_mode == "cash"
    assert part2.strategy is not None
    assert part2.strategy.timezone == "Asia/Shanghai"


def test_resume_explicit_differing_risk_and_timezone_warn_once_each(
    tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """显式传入与存档不同的 RiskConfig/timezone：以传入为准，各告警一次.

    RiskConfig 对象整体替换存档（未设的字段取默认值，不沿用存档）。
    """
    from akquant.config import RiskConfig

    _, ckpt = _save_margin_ny_checkpoint(tmp_path)
    with caplog.at_level("WARNING", logger="akquant"):
        part2 = run_from_checkpoint(
            checkpoint_path=ckpt,
            data=_ny_frame(_NY_DAYS[3:]),
            symbols="AAPL",
            show_progress=False,
            timezone="America/Chicago",
            risk_config=RiskConfig(max_order_size=1000),
        )
    warnings = _resume_warnings(caplog)
    assert len([m for m in warnings if "timezone" in m]) == 1
    assert len([m for m in warnings if "风控配置" in m]) == 1
    assert len(warnings) == 2
    assert part2.engine is not None
    rust_config = part2.engine.risk_manager.config
    assert rust_config.account_mode == "cash"
    assert rust_config.max_order_size == 1000
    assert rust_config.max_position_size is None
    assert part2.strategy is not None
    assert part2.strategy.timezone == "America/Chicago"
    assert part2.resolved_config is not None
    assert part2.resolved_config["timezone"] == "America/Chicago"
    assert part2.resolved_config["risk_config"]["account_mode"] == "cash"


def test_resume_partial_risk_dict_overlays_archive(
    tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """risk_config 传 dict 时按键叠加在存档上：未给的键（账户模式等）沿用存档."""
    _, ckpt = _save_margin_ny_checkpoint(tmp_path)
    with caplog.at_level("WARNING", logger="akquant"):
        part2 = run_from_checkpoint(
            checkpoint_path=ckpt,
            data=_ny_frame(_NY_DAYS[3:]),
            symbols="AAPL",
            show_progress=False,
            risk_config={"max_order_size": 5},
        )
    warnings = _resume_warnings(caplog)
    assert len(warnings) == 1 and "风控配置" in warnings[0]
    assert part2.engine is not None
    rust_config = part2.engine.risk_manager.config
    assert rust_config.account_mode == "margin"
    assert rust_config.financing_rate_annual == pytest.approx(36.5)
    assert rust_config.max_order_size == 5


_LIVE_SESSION_PROBES: List[Any] = []


class _LiveSessionProbe(_LiveDayOrderProbe):
    """记录 session_of / is_market_open（实盘按真实时段）."""

    def __init__(self) -> None:
        super().__init__()
        self.sessions: List[Tuple[str, str, str, bool]] = []
        _LIVE_SESSION_PROBES.append(self)

    def on_bar(self, bar: Bar) -> None:
        self.sessions.append(
            (
                _local_minute(int(bar.timestamp)),
                str(self.session_of()).split(".")[-1],
                str(self.session_of("600000")).split(".")[-1],
                self.is_market_open(),
            )
        )


def test_live_replay_session_api_reports_real_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """实盘：ctx.session 按真实时段，14:58 的 bar 处于收盘集合竞价."""
    _LIVE_SESSION_PROBES.clear()
    _run_live_replay(
        monkeypatch,
        _local_bars(["2024-01-02 10:00", "2024-01-02 14:58"]),
        strategy_cls=_LiveSessionProbe,
        trading_calendar=["2023-12-29", "2024-01-02", "2024-01-03"],
    )
    assert len(_LIVE_SESSION_PROBES) == 1
    strategy = _LIVE_SESSION_PROBES[0]
    assert [row[0] for row in strategy.sessions] == [
        "2024-01-02 10:00",
        "2024-01-02 14:58",
    ]
    assert [row[1:] for row in strategy.sessions] == [
        ("Continuous", "Continuous", True),
        ("ClosingAuction", "ClosingAuction", False),
    ], strategy.sessions


class _LiveMarketOrderProbe(_LiveDayOrderProbe):
    """首根 bar 下一张市价单（on_start 等推送线程沿用父类）."""

    def on_bar(self, bar: Bar) -> None:
        if not self.done:
            self.done = True
            self.buy("600000", 100)


def test_live_replay_auction_order_waits_for_continuous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """实盘 paper：午休 bar 上下的市价单不在午休成交，等到 13:00 后首根连续竞价 bar."""
    # 11:31、11:32 的 bar 区间 (11:30, 11:31]、(11:31, 11:32] 属午休；13:01 的
    # bar 属连续竞价。市价单在下一根 bar 撮合：强制连续竞价时会在 11:32 成交。
    runner = _run_live_replay(
        monkeypatch,
        _local_bars(["2024-01-02 11:31", "2024-01-02 11:32", "2024-01-02 13:01"]),
        strategy_cls=_LiveMarketOrderProbe,
        trading_calendar=["2023-12-29", "2024-01-02", "2024-01-03"],
    )
    orders = runner.engine.get_results().orders
    filled = [o for o in orders if o.status == aq.OrderStatus.Filled]
    assert len(filled) == 1, [(o.id, o.status) for o in orders]
    assert _local_minute(int(filled[0].updated_at)) == "2024-01-02 13:01"


_LIVE_FILLS: List[Tuple[str, float]] = []


class _LiveFirstBarBuyProbe(_LiveDayOrderProbe):
    """首根 bar 下一张市价单与一张 Day 限价单，记录成交（本地日期, 价格）."""

    def on_bar(self, bar: Bar) -> None:
        if not self.done:
            self.done = True
            self.buy(bar.symbol, 100)
            self.buy(bar.symbol, 100, price=100.0, time_in_force=aq.TimeInForce.Day)

    def on_trade(self, trade: Any) -> None:
        local = pd.Timestamp(int(trade.timestamp), tz="UTC").tz_convert("Asia/Shanghai")
        _LIVE_FILLS.append((local.strftime("%Y-%m-%d"), float(trade.price)))


def test_live_replay_date_only_daily_matches_backtest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay 只有日期的日线 DataFrame：与同数据 run_backtest 成交日、成交价一致.

    只有日期的 bar 停在本地 0 点会被判为闭市、订单永不成交；replay 须与回测一样
    把它改写到会话收盘时刻并补区间起点。
    """
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    n = len(dates)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": [10.0 + i for i in range(n)],
            "high": [10.5 + i for i in range(n)],
            "low": [9.5 + i for i in range(n)],
            "close": [10.2 + i for i in range(n)],
            "volume": [1e6] * n,
        }
    )
    calendar = ["2023-12-29", *dates, "2024-01-08"]

    _LIVE_FILLS.clear()
    run_backtest(
        strategy=_LiveFirstBarBuyProbe,
        data=df.assign(symbol="600000"),
        symbols=["600000"],
        lot_size=100,
        initial_cash=1e6,
        show_progress=False,
        freq="1d",
        trading_calendar=calendar,
    )
    backtest_fills = sorted(_LIVE_FILLS)
    assert len(backtest_fills) == 2, backtest_fills

    _LIVE_FILLS.clear()
    _run_live_replay(
        monkeypatch,
        df,
        strategy_cls=_LiveFirstBarBuyProbe,
        freq="1d",
        trading_calendar=calendar,
    )
    assert sorted(_LIVE_FILLS) == backtest_fills


_LIVE_TIMER_SESSIONS: List[Tuple[str, str, bool]] = []


class _LiveTimerSessionProbe(_LiveDayOrderProbe):
    """首根 bar 登记午休时刻的定时器，记录 on_timer 里的时段."""

    def on_bar(self, bar: Bar) -> None:
        if not self.done:
            self.done = True
            self.schedule("2024-01-02 12:00", "lunch")

    def on_timer(self, payload: str) -> None:
        assert self.ctx is not None
        _LIVE_TIMER_SESSIONS.append(
            (
                payload,
                str(self.ctx.session).split(".")[-1],
                self.is_market_open(),
            )
        )


def test_live_replay_timer_reports_real_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """实盘 on_timer 的 ctx.session 按真实时段：午休时刻的定时器为 Break."""
    _LIVE_TIMER_SESSIONS.clear()
    _run_live_replay(
        monkeypatch,
        _local_bars(["2024-01-02 10:00", "2024-01-02 13:30"]),
        strategy_cls=_LiveTimerSessionProbe,
        trading_calendar=["2023-12-29", "2024-01-02", "2024-01-03"],
    )
    assert _LIVE_TIMER_SESSIONS == [("lunch", "Break", False)]


class _LiveAfterTradingProbe(_LiveDayOrderProbe):
    """记录每次 on_after_trading 时 self.now 的本地时刻（精确到秒）."""

    def __init__(self) -> None:
        super().__init__()
        self.after_trading_at: List[str] = []

    def on_after_trading(self, trading_date: dt.date, timestamp: int) -> None:
        now = self.now
        assert now is not None
        self.after_trading_at.append(now.strftime("%Y-%m-%d %H:%M:%S"))


def _after_trading_times(runner: Any) -> List[str]:
    strategy = runner._stop_target_strategy
    assert isinstance(strategy, _LiveAfterTradingProbe)
    return list(strategy.after_trading_at)


def test_live_day_close_delay_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_live(day_close_delay=) 决定实盘日终结算时刻 = 会话收盘 + 宽限."""
    runner = _run_live_replay(
        monkeypatch,
        _local_bars(["2024-01-02 10:00", "2024-01-02 14:00", "2024-01-03 10:00"]),
        strategy_cls=_LiveAfterTradingProbe,
        trading_calendar=["2023-12-29", "2024-01-02", "2024-01-03"],
        day_close_delay=5.0,
    )
    after = [ts for ts in _after_trading_times(runner)]
    assert after == ["2024-01-02 15:00:05"]
    # 结算快照仍记在会话收盘时刻，且只结算一次
    assert _snapshot_minutes(runner).count("2024-01-02 15:00") == 1


def test_live_day_close_delay_defaults_to_sixty_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不传 day_close_delay 时实盘日终结算在会话收盘后 60 秒."""
    runner = _run_live_replay(
        monkeypatch,
        _local_bars(["2024-01-02 10:00", "2024-01-02 14:00", "2024-01-03 10:00"]),
        strategy_cls=_LiveAfterTradingProbe,
        trading_calendar=["2023-12-29", "2024-01-02", "2024-01-03"],
    )
    assert _after_trading_times(runner) == ["2024-01-02 15:01:00"]


def test_live_day_close_delay_rejects_negative() -> None:
    """负的 day_close_delay 在连网关之前报中文 ValueError."""
    from akquant import run_live

    with pytest.raises(ValueError, match="day_close_delay"):
        run_live(
            strategy_cls=_LiveDayOrderProbe,
            instruments=[_live_stock_instrument()],
            broker="replay",
            trading_mode="paper",
            gateway_options={"bars": [], "freq": "1min"},
            cash=1e6,
            show_progress=False,
            duration="1s",
            day_close_delay=-1,
        )


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (-1, ValueError),
        (-0.5, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (86_400.5, ValueError),
        (True, TypeError),
        ("60", TypeError),
        (None, TypeError),
    ],
)
def test_live_runner_validates_day_close_delay_before_engine(
    value: Any, error: type
) -> None:
    """LiveRunner 构造时就校验 day_close_delay（不等到引擎 setter）."""
    from akquant.live._runner import LiveRunner

    with pytest.raises(error, match="day_close_delay 须为"):
        LiveRunner(
            strategy_cls=_LiveDayOrderProbe,
            instruments=[_live_stock_instrument()],
            broker="replay",
            gateway_options={"bars": [], "freq": "1min"},
            day_close_delay=value,
        )


@pytest.mark.parametrize(
    ("value", "expected_ns"), [(0, 0), (5.0, 5_000_000_000), (86_400, 86_400 * 10**9)]
)
def test_live_runner_day_close_delay_converts_to_ns(
    value: Any, expected_ns: int
) -> None:
    """合法的 day_close_delay 换算为纳秒（含边界 0 与 86400）."""
    from akquant.live._runner import LiveRunner

    runner = LiveRunner(
        strategy_cls=_LiveDayOrderProbe,
        instruments=[_live_stock_instrument()],
        broker="replay",
        gateway_options={"bars": [], "freq": "1min"},
        day_close_delay=value,
    )
    assert runner.day_close_delay_ns == expected_ns


class _LiveHookSequenceProbe(_LiveDayOrderProbe):
    """记录（钩子名, 本地时刻到秒）序列；不下单（on_start 等推送线程沿用父类）."""

    def __init__(self) -> None:
        super().__init__()
        self.hooks: List[Tuple[str, str]] = []

    def _record(self, name: str) -> None:
        now = self.now
        assert now is not None
        self.hooks.append((name, now.strftime("%Y-%m-%d %H:%M:%S")))

    def on_pre_open(self, event: Dict[str, Any]) -> None:
        self._record("on_pre_open")

    def on_before_trading(self, trading_date: dt.date, timestamp: int) -> None:
        self._record("on_before_trading")

    def on_bar(self, bar: Bar) -> None:
        self._record("on_bar")

    def on_after_trading(self, trading_date: dt.date, timestamp: int) -> None:
        self._record("on_after_trading")


_HOOK_CALENDAR = ["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-04"]


def _backtest_hooks(stamps: List[str]) -> List[Tuple[str, str]]:
    """同一份 bar 转 DataFrame 走 run_backtest(freq="1min")，返回钩子序列."""
    bars = _local_bars(stamps)
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp(int(b.timestamp), tz="UTC") for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
            "symbol": "600000",
        }
    )
    result = run_backtest(
        strategy=_LiveHookSequenceProbe,
        data=df,
        symbols=["600000"],
        lot_size=100,
        initial_cash=1e6,
        show_progress=False,
        freq="1min",
        trading_calendar=_HOOK_CALENDAR,
    )
    strategy = result.strategy
    assert isinstance(strategy, _LiveHookSequenceProbe)
    return list(strategy.hooks)


def _replay_hooks(
    monkeypatch: pytest.MonkeyPatch, stamps: List[str]
) -> List[Tuple[str, str]]:
    runner = _run_live_replay(
        monkeypatch,
        _local_bars(stamps),
        strategy_cls=_LiveHookSequenceProbe,
        trading_calendar=_HOOK_CALENDAR,
        day_close_delay=0,
    )
    strategy = runner._stop_target_strategy
    assert isinstance(strategy, _LiveHookSequenceProbe)
    return list(strategy.hooks)


def _hook_names(hooks: List[Tuple[str, str]], name: str) -> List[str]:
    return [at for hook, at in hooks if hook == name]


def test_live_replay_lifecycle_matches_backtest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay 两个交易日、末日数据到收盘：钩子序列与同数据回测一致."""
    stamps = [
        "2024-01-02 10:00",
        "2024-01-02 15:00",
        "2024-01-03 10:00",
        "2024-01-03 15:00",
    ]
    backtest = _backtest_hooks(stamps)
    replay = _replay_hooks(monkeypatch, stamps)
    assert replay == backtest
    assert [at[:10] for at in _hook_names(replay, "on_pre_open")] == [
        "2024-01-02",
        "2024-01-03",
    ], replay
    assert _hook_names(replay, "on_after_trading") == [
        "2024-01-02 15:00:00",
        "2024-01-03 15:00:00",
    ], replay


def test_live_replay_data_ending_intraday_does_not_settle_last_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay 末日数据只到 10:00：与回测一样末日不结算，只有一次 on_after_trading."""
    import time

    stamps = ["2024-01-02 10:00", "2024-01-02 15:00", "2024-01-03 10:00"]
    backtest = _backtest_hooks(stamps)
    started = time.monotonic()
    replay = _replay_hooks(monkeypatch, stamps)
    elapsed = time.monotonic() - started
    assert replay == backtest
    assert _hook_names(replay, "on_after_trading") == ["2024-01-02 15:00:00"]
    assert [at[:10] for at in _hook_names(replay, "on_pre_open")] == [
        "2024-01-02",
        "2024-01-03",
    ], replay
    # 未到收盘不等结算：不应吃满收尾等待（5 秒）
    assert elapsed < 4.0, elapsed


def test_replay_gateway_exposes_first_and_last_event_ns() -> None:
    """ReplayMarketGateway 暴露当前订阅集下首个/最后事件时刻；无事件时为 None."""
    from akquant.gateway.brokers.replay.gateway import ReplayMarketGateway

    bars = _local_bars(["2024-01-02 10:00", "2024-01-02 09:31", "2024-01-02 14:00"])
    gateway = ReplayMarketGateway(feed=None, symbols=["600000"], bars=bars)
    assert gateway.first_event_ns() == min(int(b.timestamp) for b in bars)
    assert gateway.last_event_ns() == max(int(b.timestamp) for b in bars)
    gateway.unsubscribe(["600000"])
    assert gateway.first_event_ns() is None
    assert gateway.last_event_ns() is None


class _NoJoinHookSequenceProbe(_LiveHookSequenceProbe):
    """不在 on_start 里等推送线程：顺序完全靠 runner 的推送等待保证."""

    def on_start(self) -> None:
        pass


def test_live_replay_waits_for_push_before_engine_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """推送线程变慢时 replay 仍等全部行情入 feed 再开跑，钩子序列与回测一致.

    生命周期定时器在历史时刻、按墙钟恒已到期；不等推送时 feed 起初为空，
    引擎会先触发 01-02 收盘结算（无行情日结算），序列与回测不同。
    """
    import time

    from akquant import run_live
    from akquant.gateway.brokers.replay.gateway import ReplayMarketGateway
    from akquant.live import _facade

    original_start = ReplayMarketGateway.start

    def _slow_start(self: Any) -> None:
        time.sleep(0.5)
        original_start(self)

    monkeypatch.setattr(ReplayMarketGateway, "start", _slow_start)
    captured: List[Any] = []

    class _Recording(_facade.LiveRunner):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kw: Any) -> None:
            super().__init__(*args, **kw)
            captured.append(self)

    monkeypatch.setattr(_facade, "LiveRunner", _Recording)
    stamps = [
        "2024-01-02 10:00",
        "2024-01-02 15:00",
        "2024-01-03 10:00",
        "2024-01-03 15:00",
    ]
    run_live(
        strategy_cls=_NoJoinHookSequenceProbe,
        instruments=[_live_stock_instrument()],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": _local_bars(stamps), "freq": "1min"},
        cash=1e6,
        show_progress=False,
        duration="30s",
        trading_calendar=_HOOK_CALENDAR,
        day_close_delay=0,
    )
    assert len(captured) == 1
    strategy = captured[0]._stop_target_strategy
    assert isinstance(strategy, _NoJoinHookSequenceProbe)
    assert list(strategy.hooks) == _backtest_hooks(stamps)


def test_live_replay_future_data_skips_close_wait_with_clear_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """回放晚于当前时刻的数据：告警说明生命周期退回墙钟排程，且不空等末日结算."""
    import logging
    import time

    day = (pd.Timestamp.now(tz="Asia/Shanghai") + pd.Timedelta(days=400)).normalize()
    while day.weekday() >= 5:
        day += pd.Timedelta(days=1)
    stamps = [
        (day + pd.Timedelta(hours=10)).strftime("%Y-%m-%d %H:%M"),
        (day + pd.Timedelta(hours=15)).strftime("%Y-%m-%d %H:%M"),
    ]
    started = time.monotonic()
    with caplog.at_level(logging.WARNING):
        _run_live_replay(monkeypatch, _local_bars(stamps), day_close_delay=0)
    elapsed = time.monotonic() - started
    text = " ".join(record.getMessage() for record in caplog.records)
    assert "replay 首个事件晚于当前时刻" in text, text
    assert "回放数据已结束" not in text, text
    assert elapsed < 4.0, elapsed
