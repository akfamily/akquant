# -*- coding: utf-8 -*-
"""PrecomputedIndicator.get_value 的取值语义与热路径约束.

取值语义以 pandas ``Series.asof`` 为准(该时刻及之前最后一个非 NaN 值); 热路径
不得再逐次构造 ``pd.Timestamp`` / 调 ``asof``(每次约 18µs, 是 ``self.I()``
预计算指标每 bar 开销的主要来源).
"""

import pickle
from typing import Any, List

import numpy as np
import pandas as pd
import pytest
from akquant import PrecomputedIndicator


def _ns(index: pd.DatetimeIndex) -> np.ndarray:
    """时间索引的 UTC 纳秒整数."""
    return np.asarray(index.as_unit("ns").astype("int64"))


def _reference(series: pd.Series, timestamp_ns: int) -> float:
    """改造前的实现: 逐次构造 Timestamp 再 asof."""
    try:
        ts = pd.Timestamp(timestamp_ns, unit="ns", tz="UTC")
        if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is None:
            ts = ts.tz_localize(None)
        return float(series.asof(ts))  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _indicator(series: pd.Series, symbol: str = "S") -> PrecomputedIndicator:
    ind = PrecomputedIndicator("x", lambda df: df["v"])
    out = ind(pd.DataFrame({"v": series}), symbol)
    assert out is ind._data[symbol]
    return ind


def _queries(index_ns: np.ndarray, rng: np.random.Generator) -> List[int]:
    lo, hi = int(index_ns[0]), int(index_ns[-1])
    walk = [int(v) for v in index_ns]  # 逐 bar 前进(回测的真实访问模式)
    between = [int(v) + 1 for v in index_ns[::7]]
    jumps = [int(v) for v in rng.integers(lo - 10**12, hi + 10**12, 200)]
    return [lo - 1, lo - 10**15] + walk + between + jumps + [hi, hi + 1]


def _same(a: float, b: float) -> bool:
    return (a != a and b != b) or a == b


@pytest.mark.parametrize("tz", [None, "UTC", "Asia/Shanghai"])
@pytest.mark.parametrize("unit", ["ns", "us", "s"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_get_value_matches_asof(tz: Any, unit: Any, seed: int) -> None:
    """随机序列(含 NaN、重复时间戳)上与 asof 逐值相同."""
    rng = np.random.default_rng(seed)
    n = 400
    steps = rng.integers(0, 3, n) * 60  # 步长含 0 → 重复时间戳
    secs = 1_600_000_000 + np.cumsum(steps)
    index = pd.DatetimeIndex(pd.to_datetime(secs, unit="s")).as_unit(unit)
    if tz is not None:
        index = index.tz_localize(tz)
    values = rng.normal(size=n)
    values[rng.random(n) < 0.3] = np.nan
    values[:5] = np.nan  # 开头全 NaN: 之前没有可用值
    series = pd.Series(values, index=index)
    ind = _indicator(series)

    # 参照用纳秒精度的同一序列: 粗精度索引遇到换算不尽的时间戳时, 原实现的 asof
    # 会抛 "Cannot losslessly convert units" 并被吞成 NaN, 那是缺陷不是语义
    # (见 test_sub_unit_timestamp_on_coarse_index)。
    index_ns = index.as_unit("ns")
    reference = pd.Series(values, index=index_ns)
    for q in _queries(_ns(index_ns), rng):
        got, want = ind.get_value("S", q), _reference(reference, q)
        assert _same(got, want), (q, got, want)


def test_sub_unit_timestamp_on_coarse_index() -> None:
    """秒精度索引 + 带纳秒零头的时间戳: 取到前一个值, 而不是 NaN."""
    index = pd.date_range("2024-01-02", periods=5, freq="D").as_unit("s")
    series = pd.Series(np.arange(5, dtype=float), index=index)
    ind = _indicator(series)
    assert ind.get_value("S", int(index[2].value) + 1) == 2.0


def test_hot_path_does_not_call_asof(monkeypatch: pytest.MonkeyPatch) -> None:
    """时间索引有序时逐 bar 取值不得落到 Series.asof."""
    index = pd.date_range("2024-01-02 09:31", periods=50, freq="1min")
    series = pd.Series(np.arange(50, dtype=float), index=index)
    ind = _indicator(series)
    want = [_reference(series, int(v)) for v in _ns(index)]

    calls = []
    original = pd.Series.asof

    def _spy(self: pd.Series, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "asof", _spy)
    got = [ind.get_value("S", int(v)) for v in _ns(index)]
    assert got == want
    assert calls == []


def test_unknown_symbol_and_empty_series_are_nan() -> None:
    """未知标的、空序列返回 NaN."""
    ind = _indicator(pd.Series([], index=pd.DatetimeIndex([]), dtype=float))
    assert ind.get_value("S", 1) != ind.get_value("S", 1)
    assert ind.get_value("OTHER", 1) != ind.get_value("OTHER", 1)


@pytest.mark.parametrize(
    "series",
    [
        # 时间索引乱序: asof 要求有序, 改造前恒为 NaN
        pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"]),
        ),
        # 非时间索引
        pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2]),
        # 值不是数值
        pd.Series(["a", "b"], index=pd.to_datetime(["2024-01-02", "2024-01-03"])),
    ],
    ids=["unsorted", "non-datetime", "non-numeric"],
)
def test_irregular_series_keep_previous_behaviour(series: pd.Series) -> None:
    """走不了快路径的序列, 结果与改造前一致."""
    ind = _indicator(series)
    for q in [0, 1, int(pd.Timestamp("2024-01-03").value), 10**19 // 2]:
        assert _same(ind.get_value("S", q), _reference(series, q)), q


def test_non_integer_timestamp_keeps_previous_behaviour() -> None:
    """传入 Timestamp / None 时与改造前一致."""
    index = pd.date_range("2024-01-02", periods=5, freq="D", tz="Asia/Shanghai")
    series = pd.Series(np.arange(5, dtype=float), index=index)
    ind = _indicator(series)
    assert ind.get_value("S", index[2]) == 2.0
    assert ind.get_value("S", index[2].tz_convert("UTC").tz_localize(None)) == 2.0
    none_value = ind.get_value("S", None)
    assert none_value != none_value


def test_replaced_series_is_picked_up() -> None:
    """同一标的的序列被换掉后, 取值跟着换(不读旧缓存)."""
    index = pd.date_range("2024-01-02", periods=5, freq="D")
    ind = _indicator(pd.Series(np.arange(5, dtype=float), index=index))
    ts = int(index[3].value)
    assert ind.get_value("S", ts) == 3.0
    ind._data["S"] = pd.Series(np.arange(5, dtype=float) * 10, index=index)
    assert ind.get_value("S", ts) == 30.0


def test_symbols_keep_independent_positions() -> None:
    """多标的交替取值互不干扰."""
    index = pd.date_range("2024-01-02", periods=30, freq="D")
    ind = PrecomputedIndicator("x", lambda df: df["v"])
    a = ind(pd.DataFrame({"v": np.arange(30, dtype=float)}, index=index), "A")
    b = ind(pd.DataFrame({"v": np.arange(15, dtype=float) * -1}, index=index[::2]), "B")
    for v in _ns(index):
        assert ind.get_value("A", int(v)) == _reference(a, int(v))
        assert ind.get_value("B", int(v)) == _reference(b, int(v))


def test_pickle_roundtrip_after_lookup() -> None:
    """取过值的指标可序列化(optimize 子进程、热启动), 还原后取值不变."""
    index = pd.date_range("2024-01-02", periods=5, freq="D")
    ind = PrecomputedIndicator("x", _double)
    ind(pd.DataFrame({"v": np.arange(5, dtype=float)}, index=index), "S")
    ts = int(index[3].value)
    assert ind.get_value("S", ts) == 6.0
    restored = pickle.loads(pickle.dumps(ind))
    assert restored.get_value("S", ts) == 6.0
    assert restored.get_value("S", int(index[1].value)) == 2.0


def _double(df: pd.DataFrame) -> pd.Series:
    return df["v"] * 2
