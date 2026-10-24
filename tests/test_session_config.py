"""阶段 6：时段配置接口收敛与旧接口删除."""

from __future__ import annotations

import datetime as dt
from typing import Any, List

import akquant as aq
import pandas as pd
import pytest

TZ = "Asia/Shanghai"


def _ns(text: str) -> int:
    return int(pd.Timestamp(text, tz=TZ).value)


def test_trading_session_has_only_new_values() -> None:
    """TradingSession 只保留 7 个新值."""
    names = {
        n
        for n in dir(aq.TradingSession)
        if n[0].isupper()
        and isinstance(getattr(aq.TradingSession, n), aq.TradingSession)
    }
    assert names == {
        "Closed",
        "OpeningAuction",
        "AuctionHold",
        "Continuous",
        "Break",
        "ClosingAuction",
        "PostClose",
    }


def test_engine_has_no_set_market_sessions() -> None:
    """旧的全局单表时段接口已删除."""
    assert not hasattr(aq.Engine(), "set_market_sessions")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enforce_sessions": False},
        {"session_profile": "CN_FUTURES_NIGHT_23"},
        {"sessions": []},
    ],
)
def test_removed_futures_session_fields_give_migration_error(kwargs: dict) -> None:
    """已删除的期货时段字段给出中文迁移报错而非默认的 unexpected keyword."""
    with pytest.raises(TypeError, match="sessions="):
        aq.ChinaFuturesConfig(**kwargs)


def test_session_profile_error_suggests_template() -> None:
    """session_profile 的报错带上传入的模板名."""
    with pytest.raises(TypeError, match="CN_FUTURES_NIGHT_23"):
        aq.ChinaFuturesConfig(
            session_profile="CN_FUTURES_NIGHT_23"  # type: ignore[call-arg]
        )


def test_removed_options_sessions_field() -> None:
    """期权 sessions 字段已删除."""
    with pytest.raises(TypeError, match="sessions="):
        aq.ChinaOptionsConfig(sessions=[])  # type: ignore[call-arg]


def test_futures_config_other_fields_still_work() -> None:
    """保留字段（含位置参数）不受拦截影响."""
    cfg = aq.ChinaFuturesConfig(False, enforce_lot_size=False)
    assert cfg.use_china_futures_market is False
    assert cfg.enforce_lot_size is False


@pytest.mark.parametrize(
    "name", ["ChinaFuturesSessionConfig", "ChinaOptionsSessionConfig"]
)
def test_removed_session_config_classes_import_error(name: str) -> None:
    """已删除的时段配置类导入时给出中文 ImportError."""
    with pytest.raises(ImportError, match="sessions="):
        getattr(aq, name)
    assert name not in aq.__all__
    with pytest.raises(AttributeError):
        aq.NoSuchThingAtAll  # noqa: B018


def test_force_session_continuous_is_private() -> None:
    """强制连续开关只保留私有入口."""
    engine = aq.Engine()
    assert not hasattr(engine, "set_force_session_continuous")
    engine._set_force_session_continuous(True)


def test_custom_session_table_with_night_segment() -> None:
    """自定义时段表：21:00 起的段锚定前一交易日，段外为闭市."""
    sessions = {
        "xx": [
            ("21:00", "02:30", "continuous"),
            ("09:00", "10:15", "continuous"),
            ("10:30", "11:30", "continuous"),
            ("13:30", "15:00", "continuous"),
        ]
    }
    r = aq.SessionResolver(sessions=sessions, timezone=TZ)
    day, session = r.resolve_instant(
        "xx2405", _ns("2024-01-05 21:30"), aq.AssetType.Futures
    )
    assert day == dt.date(2024, 1, 8)
    assert session == aq.TradingSession.Continuous
    _, s2 = r.resolve_instant("xx2405", _ns("2024-01-08 10:20"), aq.AssetType.Futures)
    assert s2 == aq.TradingSession.Closed


@pytest.mark.parametrize(
    "bad, pattern",
    [
        ([("09:30", "09:30", "continuous")], "长度"),
        ([("25:00", "26:00", "continuous")], "HH:MM"),
        (
            [("09:30", "11:30", "continuous"), ("11:00", "12:00", "continuous")],
            "重叠",
        ),
        (
            [("21:00", "09:45", "continuous"), ("09:30", "11:30", "continuous")],
            "重叠",
        ),
        ([("09:30", "11:30", "closed")], "closed"),
        ([], "至少"),
        # 夜盘段落在前一交易日的日历日上，与前一交易日自身的日盘段重叠
        (
            [("13:00", "19:00", "continuous"), ("18:30", "02:30", "continuous")],
            "重叠",
        ),
        (
            [
                ("09:30", "11:30", "continuous"),
                ("10:00", "11:00", "continuous", "prev"),
            ],
            "重叠",
        ),
        # 日盘段结束早于开始：无锚定与显式 "day" 都拒绝
        ([("11:30", "09:30", "continuous")], "跨午夜"),
        ([("11:30", "09:30", "continuous", "day")], "跨午夜"),
        # 时刻须严格两位 HH:MM
        ([("9:30", "11:30", "continuous")], "HH:MM"),
        ([(" 09:30", "11:30", "continuous")], "HH:MM"),
        ([("09:30:00", "11:30", "continuous")], "HH:MM"),
        ([("09:30", "11:30:00", "continuous")], "HH:MM"),
    ],
)
def test_custom_session_validation(bad: list, pattern: str) -> None:
    """自定义时段表的非法写法报中文 ValueError."""
    with pytest.raises(ValueError, match=pattern):
        aq.SessionResolver(sessions={"xx": bad}, timezone=TZ)


def test_same_anchor_overlap_message_names_segments_and_fix() -> None:
    """同锚定重叠：按用户写法指明两段（跨午夜结束取模显示），并说明怎么改."""
    with pytest.raises(ValueError) as ei:
        aq.SessionResolver(
            sessions={
                "xx": [
                    ("21:00", "23:00", "continuous"),
                    ("22:00", "01:00", "continuous"),
                ]
            },
            timezone=TZ,
        )
    msg = str(ei.value)
    assert "21:00-23:00" in msg and "22:00-01:00" in msg, msg
    assert "25:00" not in msg, msg
    assert "调整" in msg, msg
    assert "  " not in msg, msg


def test_cross_anchor_overlap_with_prev_day_message() -> None:
    """夜盘段与前一交易日日盘段重叠时说明是“前一交易日”的日盘段."""
    with pytest.raises(ValueError, match="前一交易日的日盘") as ei:
        aq.SessionResolver(
            sessions={
                "xx": [
                    ("13:00", "19:00", "continuous"),
                    ("18:30", "02:30", "continuous"),
                ]
            },
            timezone=TZ,
        )
    assert "13:00-19:00" in str(ei.value) and "  " not in str(ei.value)


def test_custom_session_error_messages_have_no_stray_spaces() -> None:
    """绑定层报错文案里没有续行合并留下的连续空格."""
    with pytest.raises(TypeError) as e1:
        aq.SessionResolver(sessions=["CN_STOCK"], timezone=TZ)  # type: ignore[arg-type]
    with pytest.raises(TypeError) as e2:
        aq.SessionResolver(sessions={"xx": 1}, timezone=TZ)  # type: ignore[dict-item]
    for e in (e1, e2):
        assert "  " not in str(e.value), str(e.value)


def test_engine_configure_sessions_rejects_invalid_custom_segments() -> None:
    """Engine.configure_sessions 自身校验自定义段（实盘 runner 依赖这一处）."""
    engine = aq.Engine()
    engine.configure_sessions({"xx": [("09:30", "11:30", "continuous")]})
    with pytest.raises(ValueError, match="跨午夜"):
        engine.configure_sessions({"xx": [("11:30", "09:30", "continuous")]})
    with pytest.raises(ValueError, match="重叠"):
        engine.configure_sessions(
            {
                "xx": [
                    ("13:00", "19:00", "continuous"),
                    ("18:30", "02:30", "continuous"),
                ]
            }
        )
    with pytest.raises(ValueError, match="closed"):
        engine.configure_sessions({"xx": [("09:30", "11:30", "closed")]})


_CUSTOM_NIGHT = {
    "xx": [("21:00", "23:00", "continuous"), ("09:00", "11:00", "continuous")]
}


def _custom_night_minutes() -> pd.DataFrame:
    """周五夜盘 21:01～23:00 与周一日盘 09:01～11:00 的 1 分钟 bar."""
    night = pd.date_range("2024-01-05 21:01", "2024-01-05 23:00", freq="1min")
    day = pd.date_range("2024-01-08 09:01", "2024-01-08 11:00", freq="1min")
    idx = pd.DatetimeIndex(night.append(day)).tz_localize(TZ)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "xx2405",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )


class _CustomMemAdapter(aq.feed_adapter.BasePandasFeedAdapter):
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def load(self, request: Any) -> pd.DataFrame:
        return self._frame


def test_custom_sessions_bar_generator_and_replay_agree() -> None:
    """同一份自定义夜盘表：replay 60min 与 BarGenerator 相同，1d 夜盘归下一交易日."""
    df = _custom_night_minutes()
    got: List[str] = []
    gen = aq.BarGenerator(
        lambda b: got.append(
            pd.Timestamp(b.timestamp, tz="UTC").tz_convert(TZ).strftime("%m-%d %H:%M")
        ),
        window=60,
        interval="minute",
        timezone=TZ,
        sessions=_CUSTOM_NIGHT,
        asset_type=aq.AssetType.Futures,
        base_interval_min=1,
    )
    for row in df.to_dict("records"):
        ts = pd.Timestamp(row["timestamp"])
        gen.update_bar(
            aq.Bar(
                timestamp=int(ts.value),
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=1.0,
                symbol="xx2405",
            )
        )
    gen.flush()
    expected = ["01-05 22:00", "01-05 23:00", "01-08 10:00", "01-08 11:00"]
    assert got == expected

    src = _CustomMemAdapter(df.set_index("timestamp"))
    request = aq.feed_adapter.FeedSlice(symbol="xx2405", timezone=TZ)
    hourly = src.replay(
        "60min",
        align="session",
        emit_partial=True,
        sessions=_CUSTOM_NIGHT,
        asset_type=aq.AssetType.Futures,
    ).load(request)
    assert [i.tz_convert(TZ).strftime("%m-%d %H:%M") for i in hourly.index] == (
        expected
    )
    daily = src.replay(
        "1D",
        align="day",
        emit_partial=True,
        sessions=_CUSTOM_NIGHT,
        asset_type=aq.AssetType.Futures,
    ).load(request)
    assert len(daily) == 1
    assert daily.index[0].tz_convert(TZ).date() == dt.date(2024, 1, 8)
    assert float(daily["volume"].iloc[0]) == 240.0


def test_session_keys_duplicate_after_normalization() -> None:
    """键 trim+小写后重复时报错."""
    with pytest.raises(ValueError, match="重复"):
        aq.SessionResolver(
            sessions={"default": "CN_STOCK", " DEFAULT ": "ALWAYS_OPEN"},
            timezone=TZ,
        )


class _SessionProbe(aq.Strategy):
    """记录每根 bar 的时段（模块级定义，存档需能 pickle）."""

    def __init__(self) -> None:
        super().__init__()
        self.sessions: List[str] = []

    def on_bar(self, bar: aq.Bar) -> None:
        assert self.ctx is not None
        self.sessions.append(str(self.ctx.session).split(".")[-1])


def test_custom_sessions_through_run_backtest_and_checkpoint(tmp_path: Any) -> None:
    """自定义时段表经 run_backtest 生效，并随存档在续跑时恢复."""
    from akquant import run_backtest, run_from_checkpoint, save_checkpoint

    def frame(times: List[str]) -> pd.DataFrame:
        idx = pd.to_datetime(times).tz_localize(TZ)
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

    custom = {"600000": [("10:00", "11:00", "continuous")]}
    r1 = run_backtest(
        strategy=_SessionProbe,
        data=frame(["2024-01-02 10:30", "2024-01-02 14:00"]),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        sessions=custom,
    )
    assert r1.strategy is not None
    assert r1.strategy.sessions == ["Continuous", "Closed"]
    assert r1.engine is not None
    ckpt = tmp_path / "c.pkl"
    save_checkpoint(r1.engine, r1.strategy, str(ckpt), config=r1)
    r2 = run_from_checkpoint(
        checkpoint_path=str(ckpt),
        data=frame(["2024-01-03 10:30", "2024-01-03 14:00"]),
        symbols=["600000.SH"],
        show_progress=False,
    )
    assert r2.strategy is not None
    assert r2.strategy.sessions[-2:] == ["Continuous", "Closed"]


def test_commodity_option_follows_underlying_futures_schedule() -> None:
    """标的为夜盘期货的期权：夜盘 21:30 的期权行情属连续竞价、归下一交易日.

    注意：期权代码 au2406C500 与 sessions 键 au 同前缀，期权直接命中前缀，本用例
    只验证前缀路径，去掉期权→标的分支也会通过。跟随标的的路径由
    test_commodity_option_without_sessions_uses_futures_default、
    test_option_underlyings_order_independent_and_prefix_wins 等用例区分。
    """
    r = aq.SessionResolver(sessions={"au": "CN_FUTURES_NIGHT_0230"}, timezone=TZ)
    r.set_active(
        [("au2406", aq.AssetType.Futures), ("au2406C500", aq.AssetType.Option)]
    )
    r.set_option_underlyings({"au2406C500": "au2406"})
    day, session = r.resolve_instant(
        "au2406C500", _ns("2024-01-05 21:30"), aq.AssetType.Option
    )
    assert (day, session) == (dt.date(2024, 1, 8), aq.TradingSession.Continuous)


def test_commodity_option_without_sessions_uses_futures_default() -> None:
    """未配置 sessions：标的为已登记期货的期权用期货默认表（14:58 连续竞价）."""
    r = aq.SessionResolver(timezone=TZ)
    r.set_active(
        [("au2406", aq.AssetType.Futures), ("au2406C500", aq.AssetType.Option)]
    )
    ts = _ns("2024-01-08 14:58")
    _, before = r.resolve_instant("au2406C500", ts, aq.AssetType.Option)
    assert before == aq.TradingSession.ClosingAuction
    r.set_option_underlyings({"au2406C500": "au2406"})
    _, after = r.resolve_instant("au2406C500", ts, aq.AssetType.Option)
    assert after == aq.TradingSession.Continuous
    # 标的未登记为期货（未出现在 set_active 中）：保持 CN_OPTION
    r2 = aq.SessionResolver(timezone=TZ)
    r2.set_active([("au2406C500", aq.AssetType.Option)])
    r2.set_option_underlyings({"au2406C500": "au2406"})
    _, s2 = r2.resolve_instant("au2406C500", ts, aq.AssetType.Option)
    assert s2 == aq.TradingSession.ClosingAuction


def test_option_underlyings_order_independent_and_prefix_wins() -> None:
    """先给映射后 set_active 结果相同；期权自身命中 sessions 前缀时前缀优先."""
    r = aq.SessionResolver(sessions={"au": "CN_FUTURES_NIGHT_0230"}, timezone=TZ)
    r.set_option_underlyings({"m2405-C-3000": "au2406"})
    r.set_active(
        [("au2406", aq.AssetType.Futures), ("m2405-C-3000", aq.AssetType.Option)]
    )
    _, session = r.resolve_instant(
        "m2405-C-3000", _ns("2024-01-05 21:30"), aq.AssetType.Option
    )
    assert session == aq.TradingSession.Continuous
    # 期权代码自身命中前缀 "m"（日盘表）：不再跟随标的
    r2 = aq.SessionResolver(
        sessions={"au": "CN_FUTURES_NIGHT_0230", "m": "CN_FUTURES_DAY"}, timezone=TZ
    )
    r2.set_active(
        [("au2406", aq.AssetType.Futures), ("m2405-C-3000", aq.AssetType.Option)]
    )
    r2.set_option_underlyings({"m2405-C-3000": "au2406"})
    _, session = r2.resolve_instant(
        "m2405-C-3000", _ns("2024-01-05 21:30"), aq.AssetType.Option
    )
    assert session == aq.TradingSession.Closed


def test_etf_option_keeps_cn_option_default() -> None:
    """标的不是已登记的期货（ETF）：期权仍用 CN_OPTION."""
    r = aq.SessionResolver(timezone=TZ)
    r.set_option_underlyings({"10007000": "510050.SH"})
    _, session = r.resolve_instant(
        "10007000", _ns("2024-01-02 21:30"), aq.AssetType.Option
    )
    assert session == aq.TradingSession.Closed
    # 标的已登记但为基金：即使标的命中全天表也不跟随（跟随则 21:30 为连续竞价）
    r2 = aq.SessionResolver(sessions={"510050": "ALWAYS_OPEN"}, timezone=TZ)
    r2.set_active([("510050.SH", aq.AssetType.Fund), ("10007000", aq.AssetType.Option)])
    r2.set_option_underlyings({"10007000": "510050.SH"})
    _, session = r2.resolve_instant(
        "10007000", _ns("2024-01-02 21:30"), aq.AssetType.Option
    )
    assert session == aq.TradingSession.Closed


class _SymbolSessionProbe(aq.Strategy):
    """记录每根 bar 的 (品种, 时段)."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: List[tuple[str, str]] = []

    def on_bar(self, bar: aq.Bar) -> None:
        assert self.ctx is not None
        self.seen.append(
            (bar.symbol, str(self.ctx.session_of(bar.symbol)).split(".")[-1])
        )


def _commodity_option_frames() -> pd.DataFrame:
    idx = pd.to_datetime(["2024-01-08 10:20", "2024-01-08 14:58"]).tz_localize(TZ)
    rows = []
    for sym, px in (("au2406", 500.0), ("au2406C500", 10.0)):
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": idx,
                    "symbol": sym,
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "volume": 1.0,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_run_backtest_commodity_option_follows_registered_futures() -> None:
    """引擎从 instruments 收集期权→标的映射：商品期权用期货默认日盘表而非 CN_OPTION.

    10:20 期货日盘为小节休息（CN_OPTION 为连续竞价），14:58 期货为连续竞价
    （CN_OPTION 为收盘集合竞价）。
    """
    from akquant import run_backtest

    instruments = [
        aq.InstrumentConfig(
            symbol="au2406", asset_type="FUTURES", multiplier=1000.0, margin_ratio=0.1
        ),
        aq.InstrumentConfig(
            symbol="au2406C500",
            asset_type="OPTION",
            multiplier=1000.0,
            option_type="CALL",
            strike_price=500.0,
            expiry_date=20240425,
            underlying_symbol="au2406",
        ),
    ]
    cfg = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=1e6),
        instruments_config=instruments,
    )
    res = run_backtest(
        strategy=_SymbolSessionProbe,
        data=_commodity_option_frames(),
        config=cfg,
        show_progress=False,
    )
    assert res.strategy is not None
    opt = [sess for sym, sess in res.strategy.seen if sym == "au2406C500"]
    assert opt == ["Break", "Continuous"]


def _futures_night_warnings(
    caplog: Any, times: List[str], sessions: Any = None, symbol: str = "au2406"
) -> List[str]:
    from akquant import run_backtest

    idx = pd.to_datetime(times).tz_localize(TZ)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": symbol,
            "open": 500.0,
            "high": 500.0,
            "low": 500.0,
            "close": 500.0,
            "volume": 1.0,
        }
    )
    cfg = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=1e6),
        instruments_config=[
            aq.InstrumentConfig(
                symbol=symbol,
                asset_type="FUTURES",
                multiplier=1000.0,
                margin_ratio=0.1,
            )
        ],
    )
    caplog.clear()
    with caplog.at_level("WARNING", logger="akquant"):
        run_backtest(
            strategy=_SessionProbe,
            data=df,
            config=cfg,
            show_progress=False,
            sessions=sessions,
        )
    return [
        r.getMessage()
        for r in caplog.records
        if "闭市时段" in r.getMessage() and symbol.lower() in r.getMessage().lower()
    ]


def test_futures_night_data_without_night_template_warns_with_hint(
    caplog: Any,
) -> None:
    """期货默认日盘表、夜盘时刻的行情：告警列出三个夜盘模板并给出 sessions 写法."""
    msgs = _futures_night_warnings(caplog, ["2024-01-05 21:30", "2024-01-08 10:00"])
    assert len(msgs) == 1
    m = msgs[0]
    for name in ("CN_FUTURES_NIGHT_23", "CN_FUTURES_NIGHT_01", "CN_FUTURES_NIGHT_0230"):
        assert name in m
    assert 'run_backtest(sessions={"au": "CN_FUTURES_NIGHT_' in m
    assert "21:30" in m and "  " not in m


def test_futures_night_hint_key_with_exchange_prefix_matches_sessions(
    caplog: Any,
) -> None:
    """交易所在前的代码（SHFE.AU2406）：示例键保留交易所取品种（shfe.au），照抄即生效.

    只取开头字母会得到 ``shfe``（覆盖整个交易所）；只取 ``au`` 又匹配不上
    ``SHFE.AU2406``（sessions 按完整代码前缀匹配）。
    """
    times = ["2024-01-05 21:30", "2024-01-08 10:00"]
    msgs = _futures_night_warnings(caplog, times, symbol="SHFE.AU2406")
    assert len(msgs) == 1
    assert 'run_backtest(sessions={"shfe.au": "CN_FUTURES_NIGHT_0230"})' in msgs[0]
    assert "或 CN_FUTURES_NIGHT_0230（02:30 收）" in msgs[0]
    fixed = _futures_night_warnings(
        caplog,
        times,
        sessions={"shfe.au": "CN_FUTURES_NIGHT_0230"},
        symbol="SHFE.AU2406",
    )
    assert fixed == []


def test_futures_closed_outside_night_window_has_no_night_hint(caplog: Any) -> None:
    """20:00（夜盘窗口外）或所用表已含夜盘时，不附加夜盘模板提示."""
    msgs = _futures_night_warnings(caplog, ["2024-01-05 20:00", "2024-01-08 10:00"])
    assert len(msgs) == 1
    assert "CN_FUTURES_NIGHT_01" not in msgs[0]
    msgs = _futures_night_warnings(
        caplog,
        ["2024-01-05 23:30", "2024-01-08 10:00"],
        sessions={"au": "CN_FUTURES_NIGHT_23"},
    )
    assert len(msgs) == 1
    assert "CN_FUTURES_NIGHT_01" not in msgs[0]


class _BarTimeProbe(aq.Strategy):
    """记录每根 bar 的 (品种, 本地时刻)（模块级定义，存档需能 pickle）."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: List[tuple[str, str]] = []

    def on_bar(self, bar: aq.Bar) -> None:
        local = pd.Timestamp(bar.timestamp, unit="ns", tz="UTC").tz_convert(TZ)
        self.seen.append((bar.symbol, local.strftime("%H:%M")))


def _dce_option_config() -> aq.BacktestConfig:
    return aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=1e6),
        instruments_config=[
            aq.InstrumentConfig(
                symbol="m2405", asset_type="FUTURES", multiplier=10.0, margin_ratio=0.1
            ),
            aq.InstrumentConfig(
                symbol="m2405-C-3000",
                asset_type="OPTION",
                multiplier=10.0,
                option_type="CALL",
                strike_price=3000.0,
                expiry_date=20240408,
                underlying_symbol="m2405",
            ),
        ],
    )


def _daily_frames(symbols: List[str], times: List[str]) -> pd.DataFrame:
    idx = pd.to_datetime(times).tz_localize(TZ)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "timestamp": idx,
                    "symbol": sym,
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 1.0,
                }
            )
            for sym in symbols
        ],
        ignore_index=True,
    )


def _fixed_time_warnings(caplog: Any) -> List[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name.startswith("akquant") and 'freq="1d"' in r.getMessage()
    ]


_DCE_1500 = ["2024-01-08 15:00", "2024-01-09 15:00", "2024-01-10 15:00"]


def test_commodity_option_1500_daily_no_closing_auction_warning(caplog: Any) -> None:
    """不配 sessions、期货与商品期权同为 15:00 日线：期权跟随期货日盘表，不误报.

    引擎里期权按期货日盘表撮合（15:00 前 1ns 为连续竞价）；Python 侧预检若不知道
    期权标的，会按 CN_OPTION 判为收盘集合竞价并告警，与实际行为相反。
    """
    from akquant import run_backtest

    with caplog.at_level("WARNING", logger="akquant"):
        run_backtest(
            strategy=_BarTimeProbe,
            data=_daily_frames(["m2405", "m2405-C-3000"], _DCE_1500),
            config=_dce_option_config(),
            show_progress=False,
        )
    assert _fixed_time_warnings(caplog) == []


def test_commodity_option_1500_daily_resume_no_closing_auction_warning(
    caplog: Any, tmp_path: Any
) -> None:
    """续跑同样从合约配置与存档合约收集期权→标的映射，不误报收盘集合竞价."""
    from akquant import run_backtest, run_from_checkpoint, save_checkpoint

    r1 = run_backtest(
        strategy=_BarTimeProbe,
        data=_daily_frames(["m2405", "m2405-C-3000"], _DCE_1500),
        config=_dce_option_config(),
        show_progress=False,
    )
    assert r1.engine is not None
    ckpt = tmp_path / "opt.pkl"
    save_checkpoint(r1.engine, r1.strategy, str(ckpt), config=r1)
    more = ["2024-01-11 15:00", "2024-01-12 15:00", "2024-01-15 15:00"]
    caplog.clear()
    with caplog.at_level("WARNING", logger="akquant"):
        run_from_checkpoint(
            checkpoint_path=str(ckpt),
            data=_daily_frames(["m2405", "m2405-C-3000"], more),
            config=_dce_option_config(),
            show_progress=False,
        )
    assert _fixed_time_warnings(caplog) == []


def test_commodity_option_date_only_rewrite_matches_engine_close() -> None:
    """只有日期的数据改写到会话级收盘：期权跟随标的的 14:00 收盘表，不混入 CN_OPTION.

    期权代码与 sessions 前缀不同（未命中前缀），只能经期权→标的映射跟随标的；
    预检若不知道映射，会把 CN_OPTION 的 15:00 计入会话级收盘，改写时刻与引擎不一致。
    """
    from akquant import run_backtest

    cfg = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=1e6),
        instruments_config=[
            aq.InstrumentConfig(
                symbol="au2406",
                asset_type="FUTURES",
                multiplier=1000.0,
                margin_ratio=0.1,
            ),
            aq.InstrumentConfig(
                symbol="OPT_au2406C500",
                asset_type="OPTION",
                multiplier=1000.0,
                option_type="CALL",
                strike_price=500.0,
                expiry_date=20240425,
                underlying_symbol="au2406",
            ),
        ],
    )
    res = run_backtest(
        strategy=_BarTimeProbe,
        data=_daily_frames(
            ["au2406", "OPT_au2406C500"], ["2024-01-08", "2024-01-09", "2024-01-10"]
        ),
        config=cfg,
        show_progress=False,
        sessions={"au": [("09:00", "14:00", "continuous")]},
    )
    assert res.strategy is not None
    times = {t for _, t in res.strategy.seen}
    assert len(res.strategy.seen) == 6
    assert times == {"14:00"}


def test_commodity_option_1500_daily_prebuilt_instruments_no_warning(
    caplog: Any,
) -> None:
    """预构建合约（instruments=）同样提供期权→标的映射，不误报收盘集合竞价."""
    from akquant import run_backtest

    instruments = [
        aq.Instrument("m2405", aq.AssetType.Futures, multiplier=10.0, margin_ratio=0.1),
        aq.Instrument(
            "m2405-C-3000",
            aq.AssetType.Option,
            multiplier=10.0,
            option_type=aq.OptionType.Call,
            strike_price=3000.0,
            expiry_date=20240408,
            underlying_symbol="m2405",
        ),
    ]
    assert instruments[1].underlying_symbol == "m2405"
    assert instruments[0].underlying_symbol is None
    with caplog.at_level("WARNING", logger="akquant"):
        run_backtest(
            strategy=_BarTimeProbe(),
            data=_daily_frames(["m2405", "m2405-C-3000"], _DCE_1500),
            instruments=instruments,
            initial_cash=1e6,
            show_progress=False,
        )
    assert _fixed_time_warnings(caplog) == []


def test_commodity_option_saved_only_resume_date_only_rewrite_matches_engine(
    tmp_path: Any,
) -> None:
    """续跑段只给期货：存档里的期权仍按存档合约的标的跟随 14:00 收盘表.

    期权只存在于存档（本次数据与配置都没有），映射只能取自存档合约快照；
    取不到时 CN_OPTION 的 15:00 会混入会话级收盘，改写时刻与引擎不一致。
    """
    from akquant import run_backtest, run_from_checkpoint, save_checkpoint

    fut = aq.InstrumentConfig(
        symbol="au2406", asset_type="FUTURES", multiplier=1000.0, margin_ratio=0.1
    )
    opt = aq.InstrumentConfig(
        symbol="OPT_au2406C500",
        asset_type="OPTION",
        multiplier=1000.0,
        option_type="CALL",
        strike_price=500.0,
        expiry_date=20240425,
        underlying_symbol="au2406",
    )
    sessions = {"au": [("09:00", "14:00", "continuous")]}

    def cfg(items: List[Any]) -> aq.BacktestConfig:
        return aq.BacktestConfig(
            strategy_config=aq.StrategyConfig(initial_cash=1e6),
            instruments_config=items,
        )

    r1 = run_backtest(
        strategy=_BarTimeProbe,
        data=_daily_frames(["au2406", "OPT_au2406C500"], ["2024-01-08", "2024-01-09"]),
        config=cfg([fut, opt]),
        show_progress=False,
        sessions=sessions,
    )
    assert r1.engine is not None
    ckpt = tmp_path / "opt_saved.pkl"
    save_checkpoint(r1.engine, r1.strategy, str(ckpt), config=r1)
    r2 = run_from_checkpoint(
        checkpoint_path=str(ckpt),
        data=_daily_frames(["au2406"], ["2024-01-10", "2024-01-11"]),
        config=cfg([fut]),
        show_progress=False,
    )
    assert r2.strategy is not None
    tail = r2.strategy.seen[-2:]
    assert tail == [("au2406", "14:00"), ("au2406", "14:00")]
