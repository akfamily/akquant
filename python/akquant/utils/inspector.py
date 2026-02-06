import ast
import inspect
from typing import Optional, Type


def infer_warmup_period(strategy_cls: Type) -> int:
    """
    通过静态分析 (AST) 推断策略所需的最小预热期 (warmup_period).

    主要检测常见的指标调用模式，如:
    - SMA(30)
    - RSI(14)
    - MACD(fast=12, slow=26) -> 取 max(12, 26)

    :param strategy_cls: 策略类
    :return: 推断出的最大周期 (如果没有检测到则返回 0)
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


class IndicatorVisitor(ast.NodeVisitor):
    """
    AST Visitor to detect indicator calls and infer warmup period.

    It traverses the abstract syntax tree of a strategy class to find common
    technical indicator usages (e.g. SMA(30)) and extracts the maximum
    required lookback period.
    """

    def __init__(self) -> None:
        """Initialize the visitor."""
        self.max_period = 0
        # 常见指标名称集合
        self.indicators = {
            "SMA",
            "EMA",
            "WMA",
            "RSI",
            "MACD",
            "BBANDS",
            "ATR",
            "NATR",
            "ADX",
            "CCI",
            "CMO",
            "DX",
            "KAMA",
            "MFI",
            "MOM",
            "PPO",
            "ROC",
            "ROCP",
            "ROCR",
            "ROCR100",
            "TRIX",
            "ULTOSC",
            "WILLR",
        }
        # 常见参数名称（用于关键字参数匹配）
        self.period_args = {
            "timeperiod",
            "period",
            "window",
            "fastperiod",
            "slowperiod",
            "signalperiod",
            "fast_period",
            "slow_period",
            "signal_period",
            "fast",
            "slow",
            "signal",
            "span",
        }

    def visit_Call(self, node: ast.Call) -> None:
        """
        处理函数调用节点.

        例如: SMA(30), RSI(window=14).
        """
        func_name = self._get_func_name(node)

        if func_name and (func_name in self.indicators or func_name.endswith("MA")):
            self._analyze_args(node)

        # 继续遍历子节点（处理嵌套调用）
        self.generic_visit(node)

    def _get_func_name(self, node: ast.Call) -> Optional[str]:
        """提取函数名称."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _analyze_args(self, node: ast.Call) -> None:
        """分析参数列表，提取最大整数值."""
        # 1. 位置参数 (Positional Args)
        # 通常第一个参数就是周期，或者是 MACD 的前几个参数
        for arg in node.args:
            self._check_value(arg)

        # 2. 关键字参数 (Keyword Args)
        for keyword in node.keywords:
            if keyword.arg in self.period_args:
                self._check_value(keyword.value)

    def _check_value(self, node: ast.AST) -> None:
        """检查节点值是否为整数常量."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            val = node.value
            if isinstance(val, int) and val > 0:
                self.max_period = max(self.max_period, val)
        elif isinstance(node, ast.Num):  # Python 3.7 及以下
            val = node.n
            if isinstance(val, int) and val > 0:
                self.max_period = max(self.max_period, val)
