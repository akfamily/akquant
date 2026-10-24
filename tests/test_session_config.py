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
    ],
)
def test_custom_session_validation(bad: list, pattern: str) -> None:
    """自定义时段表的非法写法报中文 ValueError."""
    with pytest.raises(ValueError, match=pattern):
        aq.SessionResolver(sessions={"xx": bad}, timezone=TZ)


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
