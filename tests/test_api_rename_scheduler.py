"""定时器注册 API 收敛测试: 改名 schedule_daily + 新增 weekly/monthly/helper.

对应 timer-api-rfc.md。回调端 on_timer 不改,只收敛注册端命名并补齐频率方法族。
"""

from typing import Any, cast
from unittest.mock import MagicMock

import pandas as pd
from akquant.akquant import StrategyContext
from akquant.strategy import Strategy


def _make_strategy(trading_days: list[str]) -> Strategy:
    strategy = Strategy.__new__(Strategy)
    strategy.ctx = MagicMock(spec=StrategyContext)
    strategy.timezone = "Asia/Shanghai"
    strategy._trading_days = [
        pd.Timestamp(d).tz_localize("Asia/Shanghai") for d in trading_days
    ]
    return strategy


def _scheduled_dates(strategy: Strategy) -> list[str]:
    """从 ctx.schedule 调用里取出被注册的日期(升序)."""
    out = []
    assert strategy.ctx is not None
    schedule_mock = cast(MagicMock, strategy.ctx.schedule)
    for c in schedule_mock.call_args_list:
        ts = pd.Timestamp(c.args[0], tz="Asia/Shanghai")
        out.append(str(ts.date()))
    return sorted(out)


def test_add_daily_timer_renamed_to_schedule_daily() -> None:
    """旧名 add_daily_timer 应已移除,新名 schedule_daily 应存在."""
    assert not hasattr(Strategy, "add_daily_timer")
    assert hasattr(Strategy, "schedule_daily")


def test_schedule_daily_registers_one_timer_per_trading_day() -> None:
    """schedule_daily 应对每个交易日在指定时点注册一枚 timer(等价旧 API)."""
    strategy = Strategy.__new__(Strategy)
    strategy.ctx = MagicMock(spec=StrategyContext)
    strategy.timezone = "Asia/Shanghai"
    strategy._trading_days = [
        pd.Timestamp("2023-01-03").tz_localize("Asia/Shanghai"),
        pd.Timestamp("2023-01-04").tz_localize("Asia/Shanghai"),
    ]

    strategy.schedule_daily("14:55:00", "eod")

    assert strategy.ctx is not None
    schedule_mock = cast(MagicMock, strategy.ctx.schedule)
    calls = cast(list[Any], schedule_mock.call_args_list)
    assert len(calls) == 2
    for c, day in zip(calls, ("2023-01-03", "2023-01-04")):
        ts_ns = c.args[0]
        expected = pd.Timestamp(f"{day} 14:55:00").tz_localize("Asia/Shanghai").value
        assert ts_ns == expected
        # payload 仍走 __daily__ 路由,回调端 on_timer 收到原始 payload
        assert c.args[1] == "__daily__|14:55:00|eod"


# 跨年、跨月、跨周的交易日序列(含"周/月首日非自然首日"以验证顺延):
#   周首个交易日: 2023-12-28, 2024-01-02, 2024-01-08
#   月首个交易日: 2023-12-28, 2024-01-02
_TRADING_DAYS = [
    "2023-12-28",  # 周52首日 / 12月首日(本序列内)
    "2023-12-29",
    "2024-01-02",  # 周1首日 / 1月首日(1/1 元旦休市,顺延到 1/2)
    "2024-01-03",
    "2024-01-08",  # 周2首日
    "2024-01-09",
]


def test_schedule_weekly_fires_on_first_trading_day_of_each_week() -> None:
    """schedule_weekly 应在每个 ISO 周的首个交易日触发(含节假日顺延)."""
    strategy = _make_strategy(_TRADING_DAYS)
    strategy.schedule_weekly("09:30:00", "wk")
    assert _scheduled_dates(strategy) == ["2023-12-28", "2024-01-02", "2024-01-08"]


def test_schedule_monthly_fires_on_first_trading_day_of_each_month() -> None:
    """schedule_monthly 应在每个自然月的首个交易日触发(含节假日顺延)."""
    strategy = _make_strategy(_TRADING_DAYS)
    strategy.schedule_monthly("09:30:00", "mo")
    assert _scheduled_dates(strategy) == ["2023-12-28", "2024-01-02"]


def test_weekly_monthly_payload_routes_through_on_timer() -> None:
    """weekly/monthly 仍走 __daily__ 路由,回调端 on_timer 收到原始 payload."""
    strategy = _make_strategy(_TRADING_DAYS)
    strategy.schedule_weekly("09:30:00", "wk")
    assert strategy.ctx is not None
    schedule_mock = cast(MagicMock, strategy.ctx.schedule)
    payloads = {c.args[1] for c in cast(list[Any], schedule_mock.call_args_list)}
    assert payloads == {"__daily__|09:30:00|wk"}


def test_trading_days_is_exposed_readonly() -> None:
    """trading_days 只读属性应暴露引擎交易日序列."""
    strategy = _make_strategy(_TRADING_DAYS)
    assert [str(d.date()) for d in strategy.trading_days] == _TRADING_DAYS


def _dates(days: list) -> list:
    return [str(d.date()) for d in days]


def test_nth_trading_day_of_month() -> None:
    """每月第 N 个交易日 helper."""
    strategy = _make_strategy(_TRADING_DAYS)
    assert _dates(strategy.nth_trading_day_of_month(1)) == ["2023-12-28", "2024-01-02"]
    assert _dates(strategy.nth_trading_day_of_month(2)) == ["2023-12-29", "2024-01-03"]


def test_nth_last_trading_day_of_month() -> None:
    """每月倒数第 N 个交易日 helper."""
    strategy = _make_strategy(_TRADING_DAYS)
    assert _dates(strategy.nth_last_trading_day_of_month(1)) == [
        "2023-12-29",
        "2024-01-09",
    ]


def test_nth_trading_day_of_week() -> None:
    """每周第 N 个交易日 helper."""
    strategy = _make_strategy(_TRADING_DAYS)
    assert _dates(strategy.nth_trading_day_of_week(1)) == [
        "2023-12-28",
        "2024-01-02",
        "2024-01-08",
    ]


def test_calendar_helper_out_of_range_group_is_skipped() -> None:
    """某月/周不足 N 个交易日时,该组被跳过而非报错."""
    strategy = _make_strategy(_TRADING_DAYS)
    # 12 月只有 2 个交易日,取第 3 个 → 只剩 1 月的结果
    assert _dates(strategy.nth_trading_day_of_month(3)) == ["2024-01-08"]
