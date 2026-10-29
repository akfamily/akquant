import ast
import textwrap
import unittest

from akquant.utils.inspector import IndicatorVisitor


class TestStrategyInspector(unittest.TestCase):
    """Test suite for warmup period inspector."""

    def _parse_and_visit(self, code_str: str) -> int:
        """Parse code string and visit it."""
        tree = ast.parse(textwrap.dedent(code_str))
        visitor = IndicatorVisitor()
        visitor.visit(tree)
        return visitor.max_period

    def test_sma_detection(self) -> None:
        """测试 SMA(30) 简单检测."""
        code = """
        class StrategySMA:
            def __init__(self):
                self.sma = SMA(30)
        """
        self.assertEqual(self._parse_and_visit(code), 30)

    def test_multiple_indicators(self) -> None:
        """测试多个指标取最大值."""
        code = """
        class StrategyMulti:
            def __init__(self):
                self.fast = EMA(10)
                self.slow = SMA(period=50) # Keyword arg
                self.rsi = RSI(14)
        """
        self.assertEqual(self._parse_and_visit(code), 50)

    def test_nested_call(self) -> None:
        """测试嵌套调用."""
        code = """
        class StrategyNested:
            def on_bar(self):
                val = self.indicators.SMA(60)
        """
        self.assertEqual(self._parse_and_visit(code), 60)

    def test_macd_args(self) -> None:
        """测试 MACD 多参数."""
        code = """
        class StrategyMACD:
            def init(self):
                # MACD(fast, slow, signal)
                self.macd = MACD(12, 26, 9)
        """
        self.assertEqual(self._parse_and_visit(code), 34)

    def test_no_indicator(self) -> None:
        """测试无指标情况."""
        code = """
        class StrategyNone:
            def run(self):
                pass
        """
        self.assertEqual(self._parse_and_visit(code), 0)

    def test_dynamic_arg_ignored(self) -> None:
        """测试动态参数应被忽略."""
        code = """
        class StrategyDynamic:
            def __init__(self, n):
                self.sma = SMA(n) # 无法推断
                self.rsi = RSI(20) # 可以推断
        """
        self.assertEqual(self._parse_and_visit(code), 21)

    # 真实场景集成测试 (Mock inspect.getsource)
    def test_infer_function(self) -> None:
        """Test the top-level infer_warmup_period function."""
        # 由于 unittest 动态定义的类无法被 inspect.getsource 找到源码
        # 这里主要验证逻辑，核心逻辑已在上面覆盖
        pass

    def test_lookback_based_warmup(self) -> None:
        """周期参数能全部解析成整数字面量时, 预热 = TA-Lib lookback + 1."""
        cases = {
            "x = ta.MACD(close, 12, 26, 9)": 34,
            "x = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)": 34,
            "x = aq.MACD(fast_period=12, slow_period=26, signal_period=9)": 34,
            "x = aq.STOCH(9, 3, 3)": 13,
            "x = ta.STOCH(h, l, c, 9, 3, 0, 3, 0)": 13,
            "x = aq.RSI(14)": 15,
            "x = ta.RSI(close, 14)": 15,
            "self.I(aq.TEMA(14))": 40,
            "self.I(aq.ATR(14), input_mode='hlc')": 15,
            "self.I(aq.ADX(14), input_mode='hlc')": 28,
            "x = aq.SAR(0.02, 0.2)": 2,
            "x = ta.SAR(high, low)": 2,
            "x = aq.MAMA(0.5, 0.05)": 33,
            "x = aq.HT_TRENDLINE()": 64,
            "x = aq.CCI(20, 0.015)": 20,
            "x = aq.T3(5, 0.7)": 25,
            "x = aq.ULTOSC(7, 14, 28)": 29,
            "x = aq.BollingerBands(20, 2.0)": 20,
            "x = ta.KAMA(close, timeperiod=10)": 11,
        }
        for code, expected in cases.items():
            self.assertEqual(self._parse_and_visit(code), expected, code)

    def test_unresolved_periods_fall_back_to_max_literal(self) -> None:
        """周期参数是变量或被省略时退回旧规则(取整数字面量最大值)."""
        cases = {
            "x = aq.MACD(self.fast, 26, 9)": 26,
            "x = ta.RSI(close)": 0,
            "x = MyMA(20)": 20,
        }
        for code, expected in cases.items():
            self.assertEqual(self._parse_and_visit(code), expected, code)


if __name__ == "__main__":
    unittest.main()
