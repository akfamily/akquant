"""TA-Lib 口径的预热长度(lookback): 第一个有效输出前的 NaN 个数.

与 TA-Lib 0.8.2 的 ``abstract.Function(name).lookback`` 逐项核对过。
``infer_warmup_period`` 用它把 ``aq.MACD(12, 26, 9)`` 这类调用换算成所需的
预热根数(lookback + 1), 而不是简单地取参数最大值。
"""

from collections.abc import Callable, Mapping

#: 每个指标的周期类参数名(按位置顺序)与 TA-Lib 默认值
PERIOD_PARAMS: Mapping[str, tuple[tuple[str, int], ...]] = {
    "SMA": (("timeperiod", 30),),
    "EMA": (("timeperiod", 30),),
    "WMA": (("timeperiod", 30),),
    "TRIMA": (("timeperiod", 30),),
    "DEMA": (("timeperiod", 30),),
    "TEMA": (("timeperiod", 30),),
    "TRIX": (("timeperiod", 30),),
    "T3": (("timeperiod", 5),),
    "KAMA": (("timeperiod", 30),),
    "MIDPOINT": (("timeperiod", 14),),
    "MIDPRICE": (("timeperiod", 14),),
    "MAX": (("timeperiod", 30),),
    "MIN": (("timeperiod", 30),),
    "MAXINDEX": (("timeperiod", 30),),
    "MININDEX": (("timeperiod", 30),),
    "MINMAX": (("timeperiod", 30),),
    "MINMAXINDEX": (("timeperiod", 30),),
    "SUM": (("timeperiod", 30),),
    "AVGDEV": (("timeperiod", 14),),
    "STDDEV": (("timeperiod", 5),),
    "VAR": (("timeperiod", 5),),
    "BBANDS": (("timeperiod", 20),),
    "LINEARREG": (("timeperiod", 14),),
    "LINEARREG_SLOPE": (("timeperiod", 14),),
    "LINEARREG_INTERCEPT": (("timeperiod", 14),),
    "LINEARREG_ANGLE": (("timeperiod", 14),),
    "TSF": (("timeperiod", 14),),
    "CORREL": (("timeperiod", 30),),
    "BETA": (("timeperiod", 5),),
    "CCI": (("timeperiod", 14),),
    "WILLR": (("timeperiod", 14),),
    "RSI": (("timeperiod", 14),),
    "CMO": (("timeperiod", 14),),
    "MOM": (("timeperiod", 10),),
    "ROC": (("timeperiod", 10),),
    "ROCP": (("timeperiod", 10),),
    "ROCR": (("timeperiod", 10),),
    "ROCR100": (("timeperiod", 10),),
    "MFI": (("timeperiod", 14),),
    "ATR": (("timeperiod", 14),),
    "NATR": (("timeperiod", 14),),
    "DX": (("timeperiod", 14),),
    "PLUS_DI": (("timeperiod", 14),),
    "MINUS_DI": (("timeperiod", 14),),
    "ADX": (("timeperiod", 14),),
    "ADXR": (("timeperiod", 14),),
    "AROON": (("timeperiod", 14),),
    "AROONOSC": (("timeperiod", 14),),
    "MACD": (("fastperiod", 12), ("slowperiod", 26), ("signalperiod", 9)),
    "APO": (("fastperiod", 12), ("slowperiod", 26)),
    "PPO": (("fastperiod", 12), ("slowperiod", 26)),
    "ADOSC": (("fastperiod", 3), ("slowperiod", 10)),
    "ULTOSC": (("timeperiod1", 7), ("timeperiod2", 14), ("timeperiod3", 28)),
    "STOCH": (("fastk_period", 5), ("slowk_period", 3), ("slowd_period", 3)),
    "SAR": (),
    "TRANGE": (),
    "MAMA": (),
    "HT_TRENDLINE": (),
    "OBV": (),
    "AD": (),
    "BOP": (),
    "AVGPRICE": (),
    "MEDPRICE": (),
    "TYPPRICE": (),
    "WCLPRICE": (),
}

_WINDOW = {
    "SMA", "WMA", "TRIMA", "MIDPOINT", "MIDPRICE", "MAX", "MIN", "MAXINDEX",
    "MININDEX", "MINMAX", "MINMAXINDEX", "SUM", "AVGDEV", "STDDEV", "VAR", "BBANDS",
    "LINEARREG", "LINEARREG_SLOPE", "LINEARREG_INTERCEPT", "LINEARREG_ANGLE", "TSF",
    "CORREL", "CCI", "WILLR",
}  # fmt: skip
_PERIOD = {
    "RSI", "CMO", "MOM", "ROC", "ROCP", "ROCR", "ROCR100", "MFI", "ATR", "NATR",
    "DX", "PLUS_DI", "MINUS_DI", "AROON", "AROONOSC", "BETA", "KAMA",
}  # fmt: skip
_FIXED = {
    "SAR": 1, "TRANGE": 1, "MAMA": 32, "HT_TRENDLINE": 63, "OBV": 0, "AD": 0,
    "BOP": 0, "AVGPRICE": 0, "MEDPRICE": 0, "TYPPRICE": 0, "WCLPRICE": 0,
}  # fmt: skip
_FORMULAS: Mapping[str, Callable[[tuple[int, ...]], int]] = {
    "EMA": lambda p: p[0] - 1,
    "DEMA": lambda p: 2 * (p[0] - 1),
    "TEMA": lambda p: 3 * (p[0] - 1),
    "TRIX": lambda p: 3 * (p[0] - 1) + 1,
    "T3": lambda p: 6 * (p[0] - 1),
    "ADX": lambda p: 2 * p[0] - 1,
    "ADXR": lambda p: 3 * p[0] - 2,
    "MACD": lambda p: max(p[0], p[1]) - 1 + p[2] - 1,
    "APO": lambda p: max(p[0], p[1]) - 1,
    "PPO": lambda p: max(p[0], p[1]) - 1,
    "ADOSC": lambda p: max(p[0], p[1]) - 1,
    "ULTOSC": lambda p: max(p),
    "STOCH": lambda p: p[0] - 1 + p[1] - 1 + p[2] - 1,
}


def lookback(name: str, periods: Mapping[str, int] | None = None) -> int | None:
    """返回指标 ``name`` 在给定周期参数下的 lookback; 未收录的指标返回 None.

    :param name: 指标名(大写, 与 ``akquant.talib`` / ``aq.*`` 同名)
    :param periods: 周期参数, 键为 :data:`PERIOD_PARAMS` 里的参数名; 缺省的参数
        取 TA-Lib 默认值
    """
    spec = PERIOD_PARAMS.get(name)
    if spec is None:
        return None
    given = dict(periods or {})
    values = tuple(int(given.get(param, default)) for param, default in spec)
    if name in _FIXED:
        return _FIXED[name]
    if name in _WINDOW:
        return values[0] - 1
    if name in _PERIOD:
        return values[0]
    return _FORMULAS[name](values)
