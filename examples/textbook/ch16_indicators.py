"""
第 16 章：Rust 指标全景与工程化使用 (Indicators).

本示例对应教材第 16 章，演示把指标当作"可迁移、可验证、可工程化复用"的组件来使用，
重点训练三件事：

1. **读指标**：统一用"输入 / 输出 / warmup"模板理解任意指标。
2. **用指标**：EMA(方向) + ADX(强度) + NATR(风险) 的最小角色分工框架。
3. **迁移指标**：先用 ``backend="python"`` 对齐基线，再切 ``backend="rust"`` 提速。

运行方式::

    python examples/textbook/ch16_indicators.py
    python examples/textbook/ch16_indicators.py --symbol sh600000   # 可选真实数据

无网络环境下会自动回退到本地合成数据，保证示例始终可跑通。
"""

import argparse

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, Strategy
from akquant import talib as ta


def generate_mock_data(length: int = 500) -> pd.DataFrame:
    """生成带趋势的模拟 OHLCV 数据，用于无网络环境下的本地演示."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-02", periods=length, freq="B")

    trend = np.linspace(100, 150, length)
    noise = np.cumsum(np.random.randn(length) * 0.5)
    close = trend + noise

    open_ = close + np.random.randn(length) * 0.2
    high = np.maximum(open_, close) + np.abs(np.random.randn(length) * 0.5)
    low = np.minimum(open_, close) - np.abs(np.random.randn(length) * 0.5)
    volume = np.random.uniform(1e5, 5e5, length)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": "MOCK_IND",
        }
    )


def load_data(symbol: str | None, start_date: str, end_date: str) -> pd.DataFrame:
    """优先尝试 AKShare 真实数据，失败或未指定时回退到本地合成数据."""
    if symbol:
        try:
            from akquant.utils import fetch_akshare_symbol

            raw = fetch_akshare_symbol(
                symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq"
            )
            df = pd.DataFrame(raw)
            df["symbol"] = symbol
            print(f"[OK] 已从 AKShare 加载 {len(df)} 根数据（{symbol}）")
            return df
        except Exception as exc:  # noqa: BLE001 - 示例容错，网络/依赖问题一律回退
            print(f"警告: 无法获取 AKShare 数据（{exc}），改用本地合成数据。")

    print("使用本地合成数据（MOCK_IND）。")
    return generate_mock_data()


# ---------------------------------------------------------------------------
# 第一课：读指标 —— 统一的"输入 / 输出 / warmup"阅读方法
# ---------------------------------------------------------------------------
def demonstrate_indicator_properties() -> None:
    """直接计算指标数组，演示输入结构、输出结构与 warmup 空值区段."""
    print("\n" + "=" * 60)
    print("【第一课】读指标：输入 / 输出 / warmup 的统一阅读方法")
    print("=" * 60)

    data = generate_mock_data(length=100)
    close = np.asarray(data["close"], dtype=float)
    high = np.asarray(data["high"], dtype=float)
    low = np.asarray(data["low"], dtype=float)

    # 单输入、单输出
    ema = np.asarray(ta.EMA(close, timeperiod=20, backend="python"), dtype=float)
    print("\n1) EMA(close, timeperiod=20)")
    print("   输入: close（单序列）  输出: 单序列")
    print(f"   warmup 空值数（前 25 根）: {int(np.isnan(ema[:25]).sum())}")

    # 三输入、单输出、warmup 更长
    adx = np.asarray(
        ta.ADX(high, low, close, timeperiod=14, backend="python"), dtype=float
    )
    print("\n2) ADX(high, low, close, timeperiod=14)")
    print("   输入: high, low, close（三序列）  输出: 单序列")
    print(f"   warmup 空值数（前 30 根）: {int(np.isnan(adx[:30]).sum())}")

    # 单输入、三输出 —— 多输出解包顺序很关键
    upper, middle, lower = ta.BBANDS(close, timeperiod=20, backend="python")
    upper = np.asarray(upper, dtype=float)
    middle = np.asarray(middle, dtype=float)
    lower = np.asarray(lower, dtype=float)
    print("\n3) BBANDS(close, timeperiod=20)  -> (upper, middle, lower)")
    print("   输入: close（单序列）  输出: 三序列，解包顺序固定为 上/中/下轨")
    print(
        f"   第 30 根: upper={upper[29]:.2f}, "
        f"middle={middle[29]:.2f}, lower={lower[29]:.2f}"
    )


# ---------------------------------------------------------------------------
# 第三课：迁移指标 —— Python 与 Rust 后端的数值对齐
# ---------------------------------------------------------------------------
def demonstrate_backend_migration() -> None:
    """对相同指标分别用 python / rust 后端计算，打印有效区段的最大绝对差."""
    print("\n" + "=" * 60)
    print("【第三课】迁移指标：python -> rust 后端数值对齐")
    print("=" * 60)

    data = generate_mock_data(length=200)
    close = np.asarray(data["close"], dtype=float)
    high = np.asarray(data["high"], dtype=float)
    low = np.asarray(data["low"], dtype=float)

    cases = {
        "EMA": lambda backend: ta.EMA(close, timeperiod=20, backend=backend),
        "RSI": lambda backend: ta.RSI(close, timeperiod=14, backend=backend),
        "ADX": lambda backend: ta.ADX(high, low, close, timeperiod=14, backend=backend),
        "NATR": lambda backend: ta.NATR(
            high, low, close, timeperiod=14, backend=backend
        ),
    }

    print(
        "\n说明：对比'全区段'与'收敛尾段'两个口径。EMA 这类简单递推指标全程紧密一致；"
        "\nRSI/ADX/NATR 这类 Wilder 平滑指标在 warmup 区段易有初值差异，"
        "收敛后趋于接近。"
    )

    for name, fn in cases.items():
        py = np.asarray(fn("python"), dtype=float)
        rs = np.asarray(fn("rust"), dtype=float)
        idx = np.flatnonzero(~np.isnan(py) & ~np.isnan(rs))
        if idx.size == 0:
            print(f"\n  {name}: 无可比较的有效值（warmup 未完成）")
            continue
        full_diff = float(np.abs(py[idx] - rs[idx]).max())
        tail = idx[-60:]
        tail_diff = float(np.abs(py[tail] - rs[tail]).max())
        print(
            f"\n  {name}: 全区段最大差={full_diff:.2e}，收敛尾段最大差={tail_diff:.2e}"
        )


# ---------------------------------------------------------------------------
# 第二课：用指标 —— EMA(方向) + ADX(强度) + NATR(风险) 的角色分工
# ---------------------------------------------------------------------------
class IndicatorDemoStrategy(Strategy):
    """用一组指标演示"角色分工"而非"指标堆砌"的策略."""

    def __init__(self) -> None:
        """初始化指标参数与阈值."""
        super().__init__()
        self.ema_fast = 10
        self.ema_slow = 30
        self.adx_period = 14
        self.natr_period = 14
        self.rsi_period = 14

        self.adx_threshold = 20.0  # 趋势强度下限
        self.natr_threshold = 4.5  # 波动率上限（过高则减仓）
        self.rsi_upper = 70.0  # 超买界限

        # 预热：慢均线 + 缓冲，确保所有指标 warmup 完成
        self.warmup_period = self.ema_slow + 20

    def on_bar(self, bar: Bar) -> None:
        """每根 K 线计算指标并按角色分工下单."""
        symbol = bar.symbol
        lookback = self.ema_slow + 40

        close = np.asarray(
            self.get_history(count=lookback, symbol=symbol, field="close"), dtype=float
        )
        high = np.asarray(
            self.get_history(count=lookback, symbol=symbol, field="high"), dtype=float
        )
        low = np.asarray(
            self.get_history(count=lookback, symbol=symbol, field="low"), dtype=float
        )

        # get_history 会把窗口前部用 NaN 补齐到固定长度；计算指标前先剔除这段
        # 预热填充，避免 warmup 区段污染 ADX/NATR 等递推指标的输出。
        valid = ~(np.isnan(close) | np.isnan(high) | np.isnan(low))
        close, high, low = close[valid], high[valid], low[valid]
        if close.size < self.ema_slow + 5:
            return

        ema_fast = np.asarray(
            ta.EMA(close, timeperiod=self.ema_fast, backend="rust"), dtype=float
        )
        ema_slow = np.asarray(
            ta.EMA(close, timeperiod=self.ema_slow, backend="rust"), dtype=float
        )
        adx = np.asarray(
            ta.ADX(high, low, close, timeperiod=self.adx_period, backend="rust"),
            dtype=float,
        )
        natr = np.asarray(
            ta.NATR(high, low, close, timeperiod=self.natr_period, backend="rust"),
            dtype=float,
        )
        rsi = np.asarray(
            ta.RSI(close, timeperiod=self.rsi_period, backend="rust"), dtype=float
        )

        # warmup 区段不可直接参与信号判断
        latest = np.array([ema_fast[-1], ema_slow[-1], adx[-1], natr[-1], rsi[-1]])
        if np.isnan(latest).any():
            return

        pos = self.get_position(symbol)
        trend_up = ema_fast[-1] > ema_slow[-1]  # 方向：EMA
        strong = adx[-1] >= self.adx_threshold  # 强度：ADX
        calm = natr[-1] <= self.natr_threshold  # 风险：NATR
        overbought = rsi[-1] > self.rsi_upper  # 确认：RSI

        if pos == 0 and trend_up and strong and calm:
            self.order_target_percent(symbol=symbol, target_percent=0.95)
            self.log(
                f"买入: EMA{ema_fast[-1]:.2f}>{ema_slow[-1]:.2f}, "
                f"ADX={adx[-1]:.1f}, NATR={natr[-1]:.2f}, RSI={rsi[-1]:.1f}"
            )
        elif pos > 0 and (not trend_up or overbought):
            self.close_position(symbol)
            self.log(f"卖出: trend_up={trend_up}, overbought={overbought}")


def run_strategy_backtest(data: pd.DataFrame) -> None:
    """运行角色分工策略，并用 metrics_df / trades_df 打印关键结果."""
    print("\n" + "=" * 60)
    print("【第二课】用指标：EMA + ADX + NATR 角色分工策略回测")
    print("=" * 60)

    result = aq.run_backtest(
        strategy=IndicatorDemoStrategy,
        data=data,
        initial_cash=100_000,
        commission_rate=0.0003,
    )

    metrics = result.metrics_df

    def get_metric(name: str, default: float = 0.0) -> float:
        if name in metrics.index:
            return float(str(metrics.loc[name, "value"]))
        return default

    print(f"累计收益率: {get_metric('total_return_pct'):.2f}%")
    print(f"年化收益率: {get_metric('annualized_return'):.2%}")
    print(f"最大回撤  : {get_metric('max_drawdown_pct'):.2f}%")
    print(f"夏普比率  : {get_metric('sharpe_ratio'):.2f}")
    print(f"最终权益  : {get_metric('end_market_value'):.2f}")

    trades_df = result.trades_df
    if not trades_df.empty and "pnl" in trades_df.columns:
        total = len(trades_df)
        win_rate = len(trades_df[trades_df["pnl"] > 0]) / total
        print(f"成交笔数  : {total}")
        print(f"胜率      : {win_rate:.2%}")


def main() -> None:
    """运行三课演示：读指标 -> 用指标 -> 迁移指标."""
    parser = argparse.ArgumentParser(description="第 16 章：Rust 指标全景与工程化使用")
    parser.add_argument(
        "--symbol",
        default=None,
        help="可选：A 股代码（如 sh600000）。不指定则使用本地合成数据。",
    )
    parser.add_argument("--start-date", default="20240101", help="AKShare 起始日期")
    parser.add_argument("--end-date", default="20260301", help="AKShare 结束日期")
    args = parser.parse_args()

    demonstrate_indicator_properties()

    data = load_data(args.symbol, args.start_date, args.end_date)
    run_strategy_backtest(data)

    demonstrate_backend_migration()

    print("\n[完成] 第 16 章示例：读指标 -> 用指标 -> 迁移指标。")


if __name__ == "__main__":
    main()
