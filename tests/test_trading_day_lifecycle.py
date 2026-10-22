"""交易时段重构阶段 2：交易日生命周期与日终结算的验收测试."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, List

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
