"""akquant.talib 与真实 TA-Lib 逐值对拍.

基准钉在 ta-lib==0.8.2(内置 C 库 0.8.1), 口径与有意偏离见
docs/superpowers/specs/2026-10-28-talib-alignment-design.md。两个后端都要与
TA-Lib 一致; 输入含非有限值时, 期望值是"TA-Lib 只算有限行、再散回原位置"。
"""

from dataclasses import dataclass, field
from typing import Any

import akquant.talib as ta
import numpy as np
import pytest

talib = pytest.importorskip("talib")
from talib import abstract  # noqa: E402

BACKENDS = ("rust", "python")


def _walk(seed: int, length: int = 400) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, length))
    high = close + np.abs(rng.normal(0.0, 0.5, length)) + 0.05
    low = close - np.abs(rng.normal(0.0, 0.5, length)) - 0.05
    if seed % 2 == 0:
        # 30 根完全不动: 停牌按前值填充 / 一字板, 测退化窗口
        close[200:230] = close[199]
        high[200:230] = close[199]
        low[200:230] = close[199]
    if seed == 3:
        # 整数价: 制造大量并列极值
        close, high, low = np.round(close), np.round(high), np.round(low)
    volume = rng.uniform(1e5, 2e5, length)
    open_ = np.clip(close + rng.normal(0.0, 0.3, length), low, high)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _flat_start(length: int = 300) -> dict[str, np.ndarray]:
    """开头 60 根完全不动且成交量为 0, 中间再有 20 根不动."""
    rng = np.random.default_rng(9)
    close = np.r_[
        np.full(60, 50.0), 50.0 + np.cumsum(rng.normal(0.0, 1.0, length - 60))
    ]
    high = close.copy()
    low = close.copy()
    high[60:] += 0.3
    low[60:] -= 0.3
    high[150:170] = close[149]
    low[150:170] = close[149]
    close[150:170] = close[149]
    volume = np.r_[np.zeros(60), rng.uniform(1e5, 2e5, length - 60)]
    return {
        "open": close.copy(),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _with_derived(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    close = data["close"]
    real1 = close * 1.01 + (close - np.nanmean(close)) * 0.3 + 5.0
    real2 = real1.copy()
    real2[100:140] = real1[99]  # 第二条序列单边平盘, 测 CORREL 的退化判定
    return {
        **data,
        "real0": close,
        "real1": real1,
        "real2": real2,
        "unit": np.sin(close),
    }


DATASETS: dict[str, dict[str, np.ndarray]] = {
    "walk1": _with_derived(_walk(1)),
    "walk2_flat": _with_derived(_walk(2)),
    "walk3_int": _with_derived(_walk(3)),
    "walk4_flat": _with_derived(_walk(4)),
    "flat_start": _with_derived(_flat_start()),
}
CLEAN_DATASETS = tuple(DATASETS)


def _with_bad_rows(
    base: dict[str, np.ndarray], rows: slice, bad: float
) -> dict[str, np.ndarray]:
    data = {key: value.copy() for key, value in base.items()}
    for key in data:
        data[key][rows] = bad
    return data


DATASETS["lead_nan"] = _with_bad_rows(DATASETS["walk1"], slice(0, 10), np.nan)
DATASETS["mid_nan"] = _with_bad_rows(DATASETS["walk1"], slice(100, 106), np.nan)
DATASETS["mid_inf"] = _with_bad_rows(DATASETS["walk1"], slice(50, 52), np.inf)


@dataclass(frozen=True)
class Case:
    """一条对拍用例: akquant 与 TA-Lib 用同名函数、同样的输入列."""

    name: str
    inputs: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)
    #: TA-Lib 的参数名或取值与 akquant 不同时单独给; 给了就整体替换 params
    #: (不是合并, 没写的参数走 TA-Lib 默认值), 不给则沿用 params
    talib_params: dict[str, Any] | None = None
    #: 输出是下标(MAXINDEX 等): TA-Lib 预热期填 0, 我们填 NaN(设计 C6)
    index_output: bool = False
    rtol: float = 1e-9

    @property
    def case_id(self) -> str:
        """Pytest 用例 id: 函数名(参数)."""
        args = ",".join(f"{key}={value}" for key, value in self.params.items())
        return f"{self.name}({args})"


def _call_talib(case: Case, data: dict[str, np.ndarray]) -> list[np.ndarray]:
    params = case.params if case.talib_params is None else case.talib_params
    raw = getattr(talib, case.name)(*[data[key] for key in case.inputs], **params)
    outputs = [
        np.array(item, dtype=float)
        for item in (raw if isinstance(raw, tuple) else (raw,))
    ]
    if case.index_output:
        function = abstract.Function(case.name)
        function.parameters = params
        for output in outputs:
            output[: function.lookback] = np.nan
    return outputs


def _expected(case: Case, data: dict[str, np.ndarray]) -> list[np.ndarray]:
    mask = np.logical_and.reduce([np.isfinite(data[key]) for key in case.inputs])
    if bool(mask.all()):
        return _call_talib(case, data)
    positions = np.flatnonzero(mask)
    compact = {key: data[key][mask] for key in case.inputs}
    expected = []
    for output in _call_talib(case, compact):
        if case.index_output:
            found = ~np.isnan(output)
            output[found] = positions[output[found].astype(np.int64)]
        full = np.full(mask.shape[0], np.nan)
        full[mask] = output
        expected.append(full)
    return expected


def _assert_matches(actual: object, expected: np.ndarray, rtol: float) -> None:
    values = np.asarray(actual, dtype=float)
    assert values.shape == expected.shape
    np.testing.assert_array_equal(
        np.isnan(values), np.isnan(expected), err_msg="NaN 位置(预热期长度)不一致"
    )
    valid = ~np.isnan(expected)
    np.testing.assert_allclose(values[valid], expected[valid], rtol=rtol, atol=rtol)


# 任务 1: 现在已经与 TA-Lib 一致的函数(两个后端、5 组数据都通过)。
# 后续任务把各自修好的函数追加到这里, 任务 8 检查覆盖到 TA-Lib 里有的全部函数。
CASES: list[Case] = [
    Case("AD", ("high", "low", "close", "volume")),
    Case(
        "ADOSC", ("high", "low", "close", "volume"), {"fastperiod": 3, "slowperiod": 10}
    ),
    Case("ADD", ("real0", "real1")),
    Case("SUB", ("real0", "real1")),
    Case("MULT", ("real0", "real1")),
    Case("DIV", ("real0", "real1")),
    Case("ACOS", ("unit",)),
    Case("ASIN", ("unit",)),
    Case("ATAN", ("close",)),
    Case("CEIL", ("close",)),
    Case("FLOOR", ("close",)),
    Case("COS", ("close",)),
    Case("SIN", ("close",)),
    Case("COSH", ("unit",)),
    Case("SINH", ("unit",)),
    Case("TAN", ("unit",)),
    Case("TANH", ("unit",)),
    Case("EXP", ("unit",)),
    Case("LN", ("close",)),
    Case("LOG10", ("close",)),
    Case("SQRT", ("close",)),
    Case("AVGDEV", ("close",), {"timeperiod": 14}),
    Case("AVGPRICE", ("open", "high", "low", "close")),
    Case("MEDPRICE", ("high", "low")),
    Case("TYPPRICE", ("high", "low", "close")),
    Case("WCLPRICE", ("high", "low", "close")),
    Case("BOP", ("open", "high", "low", "close")),
    Case("SMA", ("close",), {"timeperiod": 30}),
    Case("SMA", ("close",), {"timeperiod": 5}),
    Case("MA", ("close",), {"timeperiod": 30}),
    Case("WMA", ("close",), {"timeperiod": 30}),
    Case("TRIMA", ("close",), {"timeperiod": 30}),
    Case("SUM", ("close",), {"timeperiod": 30}),
    Case("MAX", ("close",), {"timeperiod": 30}),
    Case("MIN", ("close",), {"timeperiod": 30}),
    Case("MINMAX", ("close",), {"timeperiod": 30}),
    Case("MIDPOINT", ("close",), {"timeperiod": 14}),
    Case("MIDPRICE", ("high", "low"), {"timeperiod": 14}),
    Case("VAR", ("close",), {"timeperiod": 5}),
    Case("LINEARREG", ("close",), {"timeperiod": 14}),
    Case("LINEARREG_ANGLE", ("close",), {"timeperiod": 14}),
    Case("LINEARREG_INTERCEPT", ("close",), {"timeperiod": 14}),
    Case("LINEARREG_SLOPE", ("close",), {"timeperiod": 14}),
    Case("TSF", ("close",), {"timeperiod": 14}),
    Case("MOM", ("close",), {"timeperiod": 10}),
    Case("ROC", ("close",), {"timeperiod": 10}),
    Case("ROCP", ("close",), {"timeperiod": 10}),
    Case("ROCR", ("close",), {"timeperiod": 10}),
    Case("ROCR100", ("close",), {"timeperiod": 10}),
    # 任务 3: EMA 家族
    Case("EMA", ("close",), {"timeperiod": 2}),
    Case("EMA", ("close",), {"timeperiod": 14}),
    Case("EMA", ("close",), {"timeperiod": 30}),
    Case("DEMA", ("close",), {"timeperiod": 14}),
    Case("TEMA", ("close",), {"timeperiod": 14}),
    Case("TRIX", ("close",), {"timeperiod": 14}),
    Case("T3", ("close",), {"timeperiod": 5, "vfactor": 0.7}),
    Case("KAMA", ("close",), {"timeperiod": 10}),
    Case("KAMA", ("close",), {"timeperiod": 30}),
    Case("MACD", ("close",), {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}),
    Case("MACD", ("close",), {"fastperiod": 5, "slowperiod": 35, "signalperiod": 5}),
    # akquant 的 APO/PPO 固定用 EMA, 等同 TA-Lib 0.8.2 的默认 matype=1
    Case(
        "APO",
        ("close",),
        {"fastperiod": 12, "slowperiod": 26},
        {"fastperiod": 12, "slowperiod": 26, "matype": 1},
    ),
    Case(
        "PPO",
        ("close",),
        {"fastperiod": 12, "slowperiod": 26},
        {"fastperiod": 12, "slowperiod": 26, "matype": 1},
    ),
    Case(
        "PPO",
        ("close",),
        {"fastperiod": 3, "slowperiod": 10},
        {"fastperiod": 3, "slowperiod": 10, "matype": 1},
    ),
    # 任务 4: Wilder 平滑家族
    Case("TRANGE", ("high", "low", "close")),
    Case("ATR", ("high", "low", "close"), {"timeperiod": 1}),
    Case("ATR", ("high", "low", "close"), {"timeperiod": 5}),
    Case("ATR", ("high", "low", "close"), {"timeperiod": 14}),
    Case("NATR", ("high", "low", "close"), {"timeperiod": 14}),
    Case("PLUS_DI", ("high", "low", "close"), {"timeperiod": 14}),
    Case("MINUS_DI", ("high", "low", "close"), {"timeperiod": 14}),
    Case("DX", ("high", "low", "close"), {"timeperiod": 14}),
    Case("ADX", ("high", "low", "close"), {"timeperiod": 5}),
    Case("ADX", ("high", "low", "close"), {"timeperiod": 14}),
    Case("ADXR", ("high", "low", "close"), {"timeperiod": 14}),
    Case("RSI", ("close",), {"timeperiod": 2}),
    Case("RSI", ("close",), {"timeperiod": 14}),
    Case("CMO", ("close",), {"timeperiod": 14}),
    # 任务 5: 窗口类、退化窗口
    Case("CCI", ("high", "low", "close"), {"timeperiod": 14}),
    Case("CCI", ("high", "low", "close"), {"timeperiod": 20}),
    Case("MFI", ("high", "low", "close", "volume"), {"timeperiod": 14}),
    Case("WILLR", ("high", "low", "close"), {"timeperiod": 14}),
    Case(
        "STOCH",
        ("high", "low", "close"),
        {"fastk_period": 5, "slowk_period": 3, "slowd_period": 3},
    ),
    Case(
        "STOCH",
        ("high", "low", "close"),
        {"fastk_period": 14, "slowk_period": 1, "slowd_period": 1},
    ),
    Case("AROON", ("high", "low"), {"timeperiod": 14}),
    Case("AROONOSC", ("high", "low"), {"timeperiod": 14}),
    Case(
        "ULTOSC",
        ("high", "low", "close"),
        {"timeperiod1": 7, "timeperiod2": 14, "timeperiod3": 28},
    ),
    Case("BETA", ("real0", "real1"), {"timeperiod": 5}),
    Case("CORREL", ("real0", "real2"), {"timeperiod": 30}),
    Case("OBV", ("close", "volume")),
    Case("SAR", ("high", "low"), {"acceleration": 0.02, "maximum": 0.2}),
    Case("SAR", ("high", "low"), {"acceleration": 0.05, "maximum": 0.5}),
    Case("VAR", ("close",), {"timeperiod": 5, "nbdev": 2.0}),
    Case("STDDEV", ("close",), {"timeperiod": 5, "nbdev": 1.5}),
    Case("BBANDS", ("close",), {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0}),
    Case("MAXINDEX", ("close",), {"timeperiod": 30}, index_output=True),
    Case("MININDEX", ("close",), {"timeperiod": 30}, index_output=True),
    Case("MINMAXINDEX", ("close",), {"timeperiod": 30}, index_output=True),
    # 任务 6: Hilbert 变换(含 atan 与相位比较, 容差放宽到 1e-8)
    Case("MAMA", ("close",), {"fastlimit": 0.5, "slowlimit": 0.05}, rtol=1e-8),
    Case("MAMA", ("close",), {"fastlimit": 0.3, "slowlimit": 0.03}, rtol=1e-8),
    Case("HT_TRENDLINE", ("close",), rtol=1e-8),
]


def _datasets_for_test() -> list[str]:
    return list(DATASETS)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dataset", _datasets_for_test())
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_matches_talib(case: Case, dataset: str, backend: str) -> None:
    """每个后端在每组数据上都与 TA-Lib 的输出逐值一致."""
    data = DATASETS[dataset]
    raw = getattr(ta, case.name)(
        *[data[key] for key in case.inputs], **case.params, backend=backend
    )
    actual = list(raw) if isinstance(raw, tuple) else [raw]
    expected = _expected(case, data)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        _assert_matches(got, want, case.rtol)


def test_streaming_ema_returns_none_during_warmup() -> None:
    """EMA 预热期不出值, 第 n 个值是前 n 个的均值(#413 的现象之一)."""
    import akquant as aq

    ema = aq.EMA(3)
    assert [ema.update(v) for v in (1.0, 2.0)] == [None, None]
    assert not ema.is_ready
    assert ema.update(6.0) == pytest.approx(3.0)  # 种子 = (1+2+6)/3
    assert ema.update(7.0) == pytest.approx(5.0)  # (7-3)*0.5+3


def test_streaming_macd_first_value_is_not_fabricated() -> None:
    """MACD 在信号线就绪前不输出任何值, 而不是首值给 0.0(#413)."""
    import akquant as aq

    closes = DATASETS["walk1"]["close"]
    macd = aq.MACD(12, 26, 9)
    outputs = [macd.update(float(v)) for v in closes[:40]]
    assert all(out is None for out in outputs[:33])
    assert outputs[33] is not None


def test_streaming_constructors_swap_reversed_periods() -> None:
    """慢线周期短于快线时, Rust 流式类与 TA-Lib 一样互换两者(设计 C6).

    ``akquant.talib`` 的函数入口仍对 slowperiod <= fastperiod 抛 ValueError,
    这里只测流式类的构造行为。
    """
    import akquant as aq

    close = DATASETS["walk1"]["close"]
    cases: list[tuple[str, Any, Any]] = [
        ("APO", aq.APO(26, 12), talib.APO(close, 26, 12, 1)),
        ("PPO", aq.PPO(26, 12), talib.PPO(close, 26, 12, 1)),
        ("MACD", aq.MACD(26, 12, 9), talib.MACD(close, 26, 12, 9)),
    ]
    for name, indicator, raw in cases:
        rows = [indicator.update(float(value)) for value in close]
        wanted = list(raw) if isinstance(raw, tuple) else [raw]
        for position, want in enumerate(wanted):
            got = np.array(
                [
                    np.nan
                    if row is None
                    else (row[position] if isinstance(row, tuple) else row)
                    for row in rows
                ],
                dtype=float,
            )
            try:
                _assert_matches(got, np.asarray(want, dtype=float), 1e-9)
            except AssertionError as error:
                raise AssertionError(f"{name} 第 {position} 路输出不一致") from error


def test_issue_414_atr_matches_talib() -> None:
    """#414: ATR 首个有效值的下标与 TA-Lib 一致."""
    rng = np.random.default_rng(99)
    close = 100 + np.cumsum(rng.normal(0, 0.5, 20))
    high = close + np.abs(rng.normal(0, 0.4, 20)) + 0.05
    low = close - np.abs(rng.normal(0, 0.4, 20)) - 0.05
    expected = talib.ATR(high, low, close, timeperiod=5)
    assert np.isnan(expected[4]) and not np.isnan(expected[5])
    for backend in BACKENDS:
        _assert_matches(
            ta.ATR(high, low, close, timeperiod=5, backend=backend), expected, 1e-9
        )


def test_default_parameters_match_talib() -> None:
    """与 TA-Lib 同名的函数, 共有的数值型参数默认值必须相同."""
    import inspect

    mismatched = {}
    for name in ta.__all__:
        if not hasattr(talib, name):
            continue
        reference = abstract.Function(name).parameters
        for param, spec in inspect.signature(getattr(ta, name)).parameters.items():
            if param in reference and isinstance(spec.default, (int, float)):
                if float(spec.default) != float(reference[param]):
                    mismatched[f"{name}.{param}"] = (spec.default, reference[param])
    assert not mismatched, mismatched


def _series_inputs(func: Any, length: int, fill: float) -> list[np.ndarray]:
    """按签名里必填的序列参数(无默认值)造 length 长的输入."""
    import inspect

    params = inspect.signature(func).parameters.values()
    return [
        np.full(length, fill)
        for param in params
        if param.default is inspect.Parameter.empty
        and param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(ta.__all__))
def test_empty_and_all_nan_input(name: str, backend: str) -> None:
    """全部函数: 空输入 / 全 NaN 输入返回与输入等长的全 NaN, 不抛异常."""
    func = getattr(ta, name)
    for length, fill in ((0, 1.0), (70, np.nan)):
        inputs = _series_inputs(func, length, fill)
        assert inputs, name
        out = func(*inputs, backend=backend)
        outs = out if isinstance(out, tuple) else (out,)
        assert outs, name
        for item in outs:
            arr = np.asarray(item, dtype=float)
            assert len(arr) == length, (name, backend, length)
            assert np.isnan(arr).all(), (name, backend, length)


def test_lookback_table_matches_talib() -> None:
    """Lookback 表的默认参数与逐参数组合都与 TA-Lib 一致."""
    import itertools

    from akquant.talib.lookback import PERIOD_PARAMS, lookback

    for name, spec in PERIOD_PARAMS.items():
        function = abstract.Function(name)
        assert function.lookback == lookback(name), name
        for param, default in spec:
            assert function.parameters[param] == default, (name, param)
        for combo in itertools.product(*[(2, 3, 9, 26) for _ in spec]):
            params = dict(zip([param for param, _ in spec], combo))
            function = abstract.Function(name)
            function.parameters = params
            assert function.lookback == lookback(name, params), (name, params)


def test_every_talib_function_is_covered() -> None:
    """akquant.talib 里凡是 TA-Lib 也有的函数, 都必须有对拍用例."""
    names = [name for name in dir(ta) if name.isupper() and callable(getattr(ta, name))]
    covered = {case.name for case in CASES}
    missing = sorted(
        name for name in names if hasattr(talib, name) and name not in covered
    )
    assert not missing, missing


def test_natr_period_one_normalizes_by_close() -> None:
    """有意偏离 C6: NATR(1) 照常按收盘价归一化 (TA-Lib 此时直接返回 TRANGE)."""
    data = DATASETS["walk1"]
    high, low, close = data["high"], data["low"], data["close"]
    for backend in BACKENDS:
        natr = np.asarray(ta.NATR(high, low, close, 1, backend=backend), dtype=float)
        atr = np.asarray(ta.ATR(high, low, close, 1, backend=backend), dtype=float)
        np.testing.assert_allclose(natr, atr / close * 100, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", ["ROC", "ROCP", "ROCR", "ROCR100"])
@pytest.mark.parametrize("timeperiod", [1, 3, 10])
def test_roc_family_zero_base_matches_talib(
    name: str, timeperiod: int, backend: str
) -> None:
    """前值为 0 时 ROC 系列与 TA-Lib 一致输出 0(而不是 NaN / inf)."""
    close = DATASETS["walk1"]["close"].copy()
    close[[5, 6, 40, 41, 42, 150, 300]] = 0.0
    expected = getattr(talib, name)(close, timeperiod=timeperiod)
    actual = getattr(ta, name)(close, timeperiod=timeperiod, backend=backend)
    _assert_matches(actual, expected, 1e-9)
