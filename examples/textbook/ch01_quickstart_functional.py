"""
第 1 章：快速开始（函数式写法）.

本示例与 ch01_quickstart.py 的交易逻辑、参数、数据源完全一致，
只把策略入口从类风格换成函数式：

1. 类属性 -> ctx 属性（在 initialize 中赋值）
2. Strategy.on_bar 方法 -> 模块级 on_bar(ctx, bar) 函数
3. strategy=DualMAStrategy -> strategy=on_bar + initialize=initialize

两份示例的回测统计输出应完全一致，这说明函数式与类风格在引擎层等价。
"""

from typing import Any

import akquant as aq
import akshare as ak
import numpy as np
import pandas as pd
from akquant import Bar


def generate_mock_data(length: int = 970) -> pd.DataFrame:
    """断网兜底：生成带趋势与波动的合成日线数据，保证示例可离线跑通."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-02", periods=length, freq="B")
    trend = np.linspace(8.0, 7.0, length)
    noise = np.cumsum(np.random.randn(length) * 0.05)
    close = trend + noise
    open_ = close + np.random.randn(length) * 0.02
    high = np.maximum(open_, close) + np.abs(np.random.randn(length) * 0.05)
    low = np.minimum(open_, close) - np.abs(np.random.randn(length) * 0.05)
    volume = np.random.uniform(1e6, 5e6, length)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": "600000",
        }
    )


def get_data() -> pd.DataFrame:
    """
    步骤 1: 数据获取（与类风格版完全一致）.

    优先使用 AKShare 获取浦发银行 (600000) 的历史日线数据；
    若无网络或接口异常，则回退到本地合成数据，保证示例断网也能跑通。
    """
    print("正在获取数据...")
    try:
        df = ak.stock_zh_a_daily(
            symbol="sh600000",
            start_date="20200101",
            end_date="20231231",
            adjust="qfq",
        )
        df["symbol"] = "600000"
        if "date" not in df.columns:
            df = df.reset_index().rename(columns={"index": "date"})
        print(f"已从 AKShare 获取 {len(df)} 条数据。")
        return df  # type: ignore[no-any-return]
    except Exception as exc:  # noqa: BLE001 - 示例容错：网络/依赖异常一律回退
        print(f"AKShare 获取失败（{exc}），改用本地合成数据。")
        return generate_mock_data()


def initialize(ctx: Any) -> None:
    """
    步骤 2: 策略状态初始化，等价于类风格的 __init__.

    关键差异：函数式没有类体，warmup_period 必须在这里挂到 ctx 上。
    引擎会取 ctx.warmup_period 与 run_backtest(warmup_period=...) 的较大值，
    所以在此赋值即可生效。
    """
    ctx.short_window = 5
    ctx.long_window = 20
    ctx.warmup_period = ctx.long_window


def on_start(ctx: Any) -> None:
    """策略启动回调，等价于类风格的 on_start."""
    print("策略初始化...")


def on_bar(ctx: Any, bar: Bar) -> None:
    """核心交易逻辑，等价于类风格的 on_bar；self 全部换成 ctx."""
    symbol = bar.symbol

    # 1. 获取历史数据
    closes = ctx.get_history(count=ctx.long_window, symbol=symbol, field="close")

    # 2. 计算均线
    ma5_curr = closes[-ctx.short_window :].mean()
    ma20_curr = closes[-ctx.long_window :].mean()

    # 3. 获取持仓
    position = ctx.get_position(symbol)

    # 4. 交易信号
    if ma5_curr > ma20_curr and position == 0:
        print(
            f"[{bar.timestamp_iso}] 金叉买入 (MA5={ma5_curr:.2f}, MA20={ma20_curr:.2f})"
        )
        ctx.order_target_percent(symbol=symbol, target_percent=0.95)

    elif ma5_curr < ma20_curr and position > 0:
        print(
            f"[{bar.timestamp_iso}] 死叉卖出 (MA5={ma5_curr:.2f}, MA20={ma20_curr:.2f})"
        )
        ctx.order_target_percent(symbol=symbol, target_percent=0.0)


if __name__ == "__main__":
    # 1. 准备数据
    df = get_data()

    # 2. 运行回测
    print("开始回测...")
    result = aq.run_backtest(
        strategy=on_bar,
        initialize=initialize,
        on_start=on_start,
        data=df,
        initial_cash=100_000,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
        lot_size=100,
    )

    # 3. 打印结果
    print("\n" + "=" * 30)
    print("回测结果摘要")
    print("=" * 30)
    print(result)
