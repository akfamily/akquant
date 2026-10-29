import ast
import inspect
from functools import lru_cache
from typing import Optional, Type

from ..talib.lookback import PERIOD_PARAMS, lookback


def infer_warmup_period(strategy_cls: Type) -> int:
    """
    通过静态分析 (AST) 推断策略所需的最小预热期 (warmup_period).

    主要检测常见的指标调用模式，如:
    - SMA(30)
    - RSI(14)
    - MACD(fast=12, slow=26, signal=9) -> 按 TA-Lib lookback 计算再加 1（34 根）

    :param strategy_cls: 策略类
    :return: 推断出的预热根数，即各指标 lookback + 1 的最大值 (如果没有检测到则返回 0)
    """
    try:
        source = inspect.getsource(strategy_cls)
    except (OSError, TypeError):
        # 无法获取源码（如在 REPL 中定义，或者是编译后的扩展）
        return 0

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0

    visitor = IndicatorVisitor()
    visitor.visit(tree)
    return visitor.max_period


#: 类名与 TA-Lib 函数名不同的内置指标
_CLASS_ALIASES = {"BollingerBands": "BBANDS"}

#: 构造参数名/关键字 -> TA-Lib 参数名
_PARAM_ALIASES = {
    "period": "timeperiod",
    "window": "timeperiod",
    "span": "timeperiod",
    "fast_period": "fastperiod",
    "fast": "fastperiod",
    "slow_period": "slowperiod",
    "slow": "slowperiod",
    "signal_period": "signalperiod",
    "signal": "signalperiod",
    "period1": "timeperiod1",
    "period2": "timeperiod2",
    "period3": "timeperiod3",
}

#: 旧规则(取整数参数最大值)仍覆盖的指标名, 表里没有的走它
_LEGACY_NAMES = {
    "SMA", "EMA", "WMA", "RSI", "MACD", "BBANDS", "ATR", "NATR", "ADX", "CCI",
    "CMO", "DX", "KAMA", "MFI", "MOM", "PPO", "ROC", "ROCP", "ROCR", "ROCR100",
    "TRIX", "ULTOSC", "WILLR",
}  # fmt: skip


def _canonical(param: str) -> str:
    return _PARAM_ALIASES.get(param, param)


def _int_literal(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and type(node.value) is int and node.value > 0:
        return int(node.value)
    return None


def _is_number_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) in (int, float)


@lru_cache(maxsize=None)
def _talib_layout(name: str) -> Optional[tuple[int, tuple[str, ...]]]:
    """akquant.talib 函数的 (数据列个数, 其后的位置参数名)."""
    from .. import talib  # 惰性导入, 避免 utils 与 talib 互相导入
    from ..talib.core import SeriesLike

    func = getattr(talib, name, None)
    if func is None:
        return None
    data = 0
    names: list[str] = []
    for param in inspect.signature(func).parameters.values():
        if param.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            break
        if param.annotation is SeriesLike:
            data += 1
        else:
            names.append(param.name)
    return data, tuple(names)


@lru_cache(maxsize=None)
def _class_layout(name: str) -> Optional[tuple[str, ...]]:
    """内置 Rust 指标类构造函数的参数名(取自 ``__text_signature__``)."""
    from .. import akquant as _rust

    cls = getattr(_rust, name, None)
    text = getattr(cls, "__text_signature__", None)
    if not isinstance(text, str):
        return None
    inner = text.strip()[1:-1]
    return tuple(part.strip() for part in inner.split(",") if part.strip())


class IndicatorVisitor(ast.NodeVisitor):
    """遍历策略源码, 按 TA-Lib lookback 推断所需预热根数.

    能把一次调用的周期参数全部解析成整数字面量时, 预热 = lookback + 1
    (例如 ``MACD(12, 26, 9)`` 需要 34 根); 解析不全(参数是变量、或省略了
    周期参数)时退回旧规则: 取这次调用里整数字面量的最大值。
    """

    def __init__(self) -> None:
        """Initialize the visitor."""
        self.max_period = 0

    def visit_Call(self, node: ast.Call) -> None:
        """处理函数调用节点."""
        func_name = self._get_func_name(node)
        if func_name is not None:
            name = _CLASS_ALIASES.get(func_name, func_name)
            if name in PERIOD_PARAMS or name in _LEGACY_NAMES or name.endswith("MA"):
                self.max_period = max(self.max_period, self._warmup_for(name, node))
        self.generic_visit(node)

    def _get_func_name(self, node: ast.Call) -> Optional[str]:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _warmup_for(self, name: str, node: ast.Call) -> int:
        periods = self._resolve_periods(name, node)
        if periods is not None:
            value = lookback(name, periods)
            if value is not None:
                return value + 1
        return self._legacy_max(node)

    def _resolve_periods(self, name: str, node: ast.Call) -> Optional[dict[str, int]]:
        """把调用的周期参数解析成 {TA-Lib 参数名: 整数}; 解析不全返回 None."""
        spec = PERIOD_PARAMS.get(name)
        if spec is None:
            return None
        wanted = [param for param, _ in spec]
        args = list(node.args)
        if all(_is_number_literal(arg) for arg in args):
            names = _class_layout(name if name != "BBANDS" else "BollingerBands")
        else:
            layout = _talib_layout(name)
            if layout is None:
                return None
            data_count, names = layout
            if len(args) < data_count or any(
                _is_number_literal(arg) for arg in args[:data_count]
            ):
                return None
            args = args[data_count:]
        if names is None and args:
            return None
        resolved: dict[str, int] = {}
        for param, arg in zip(names or (), args):
            value = _int_literal(arg)
            canonical = _canonical(param)
            if canonical in wanted:
                if value is None:
                    return None
                resolved[canonical] = value
        for keyword in node.keywords:
            if keyword.arg is None:
                return None
            canonical = _canonical(keyword.arg)
            if canonical in wanted:
                value = _int_literal(keyword.value)
                if value is None:
                    return None
                resolved[canonical] = value
        if any(param not in resolved for param in wanted):
            return None
        return resolved

    def _legacy_max(self, node: ast.Call) -> int:
        best = 0
        for arg in node.args:
            value = _int_literal(arg)
            if value is not None:
                best = max(best, value)
        for keyword in node.keywords:
            if keyword.arg in _PARAM_ALIASES or keyword.arg in {
                "timeperiod",
                "fastperiod",
                "slowperiod",
                "signalperiod",
            }:
                value = _int_literal(keyword.value)
                if value is not None:
                    best = max(best, value)
        return best
