"""回测 tick 输入适配层."""

import pytest
from akquant.akquant import Bar, Tick
from akquant.backtest.tick_input import (
    normalize_market_input,
    parse_freq_to_interval_min,
)

_BASE_NS = 1_672_707_000_000_000_000
_MINUTE_NS = 60_000_000_000


def _ns(minutes: int) -> int:
    """构造纳秒级时间戳."""
    return _BASE_NS + minutes * _MINUTE_NS


def _bar(minutes: int, symbol: str = "A") -> Bar:
    """构造一根 bar."""
    return Bar(
        timestamp=_ns(minutes),
        open=10.0,
        high=10.5,
        low=9.5,
        close=10.0,
        volume=1000.0,
        symbol=symbol,
    )


def _tick(minutes: int, symbol: str = "A") -> Tick:
    """构造一个 tick."""
    return Tick(timestamp=_ns(minutes), price=10.0, volume=100.0, symbol=symbol)


def test_splits_bars_and_ticks() -> None:
    """混合列表按类型分成两组."""
    bars, ticks = normalize_market_input([_bar(1), _tick(2), _bar(3)])

    assert len(bars) == 2
    assert len(ticks) == 1


def test_sorts_each_group_by_timestamp() -> None:
    """两组各自按时间戳升序: 引擎所见顺序由此确定."""
    bars, ticks = normalize_market_input([_bar(30), _tick(20), _bar(10), _tick(5)])

    assert [b.timestamp for b in bars] == [_ns(10), _ns(30)]
    assert [t.timestamp for t in ticks] == [_ns(5), _ns(20)]


def test_rejects_empty_input() -> None:
    """空列表早失败, 而非让引擎空跑."""
    with pytest.raises(ValueError, match="空"):
        normalize_market_input([])


def test_rejects_foreign_element_naming_index_and_type() -> None:
    """非 Bar/Tick 元素须指名位置与实际类型.

    现有 engine.py 只查 ``data[0]``, 混合列表里的非法元素会漏到 Rust 层
    抛出难以定位的错误。
    """
    with pytest.raises(TypeError) as exc_info:
        normalize_market_input([_bar(1), "garbage", _tick(2)])

    message = str(exc_info.value)
    assert "1" in message
    assert "str" in message


def test_parse_freq_minutes() -> None:
    """分钟级 freq 解析."""
    assert parse_freq_to_interval_min("1min") == 1
    assert parse_freq_to_interval_min("5min") == 5


def test_parse_freq_hours() -> None:
    """小时级 freq 折算成分钟."""
    assert parse_freq_to_interval_min("1h") == 60


def test_parse_freq_rejects_sub_minute() -> None:
    """秒级 freq 明确报错而非静默取整."""
    with pytest.raises(ValueError, match="整数分钟"):
        parse_freq_to_interval_min("30s")
