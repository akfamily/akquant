"""Tests for fetch_akshare_symbol error semantics (no network)."""

import sys
from types import ModuleType
from typing import Any, Callable, List

import pandas as pd
import pytest
from akquant.utils import fetch_akshare_symbol


def _install(
    monkeypatch: pytest.MonkeyPatch, behaviors: List[Callable[[], Any]]
) -> List[str]:
    """Inject a fake akshare whose stock_zh_a_daily follows ``behaviors``."""
    calls: List[str] = []
    fake = ModuleType("akshare")

    def stock_zh_a_daily(symbol: str, **kwargs: Any) -> pd.DataFrame:
        calls.append(symbol)
        return behaviors[len(calls) - 1]()  # type: ignore[no-any-return]

    fake.stock_zh_a_daily = stock_zh_a_daily  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "akshare", fake)
    return calls


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日期": ["2024-01-02"],
            "开盘": [1.0],
            "最高": [1.2],
            "最低": [0.9],
            "收盘": [1.1],
            "成交量": [100.0],
        }
    )


def _empty() -> pd.DataFrame:
    return pd.DataFrame()


def test_prefixed_call_success_maps_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefixed call returning data is used and columns are renamed."""
    calls = _install(monkeypatch, [_data])
    df = fetch_akshare_symbol("600000", "20240101", "20240131")
    assert calls == ["sh600000"]
    assert {"date", "open", "high", "low", "close", "volume"} <= set(df.columns)


def test_empty_prefixed_then_unprefixed_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty prefixed result falls back to the unprefixed call."""
    calls = _install(monkeypatch, [_empty, _data])
    df = fetch_akshare_symbol("600000", "20240101", "20240131")
    assert calls == ["sh600000", "600000"]
    assert len(df) == 1


def test_both_empty_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Genuinely empty data keeps raising ValueError."""
    _install(monkeypatch, [_empty, _empty])
    with pytest.raises(ValueError, match="No data found"):
        fetch_akshare_symbol("600000", "20240101", "20240131")


def test_both_raise_runtime_error_chained(monkeypatch: pytest.MonkeyPatch) -> None:
    """Request failures surface as RuntimeError chained to the first exception."""
    first = ConnectionError("down")
    second = ConnectionError("down")

    def f1() -> pd.DataFrame:
        raise first

    def f2() -> pd.DataFrame:
        raise second

    _install(monkeypatch, [f1, f2])
    with pytest.raises(RuntimeError) as ei:
        fetch_akshare_symbol("600000", "20240101", "20240131")
    assert "AKShare 请求失败" in str(ei.value)
    assert "ConnectionError: down" in str(ei.value)
    assert ei.value.__cause__ is first


def test_one_raises_one_empty_is_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One exception plus one empty result is still a request failure."""

    def boom() -> pd.DataFrame:
        raise TimeoutError("slow")

    _install(monkeypatch, [boom, _empty])
    with pytest.raises(RuntimeError, match="TimeoutError: slow"):
        fetch_akshare_symbol("600000", "20240101", "20240131")

    _install(monkeypatch, [_empty, boom])
    with pytest.raises(RuntimeError, match="AKShare 请求失败"):
        fetch_akshare_symbol("600000", "20240101", "20240131")
