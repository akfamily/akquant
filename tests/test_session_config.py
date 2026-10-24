"""阶段 6：时段配置接口收敛与旧接口删除."""

from __future__ import annotations

import akquant as aq
import pytest


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
