"""
第 11 章：参数优化与过拟合 (Optimization & Overfitting)（函数式写法）.

本示例展示了如何使用 AKQuant 的网格搜索 (Grid Search) 功能来寻找最优的策略参数。
同时，我们也会探讨过度优化带来的风险。

策略逻辑：
- 依然使用双均线策略 (MA_Short vs MA_Long)
- 优化目标：寻找夏普比率 (Sharpe Ratio) 最高的参数组合
    - short_window: [3, 5, 10]
    - long_window: [15, 20, 30, 60]

AKQuant 特性：
- `run_grid_search`: 自动多进程并行回测，极大提高优化效率。

函数式改写说明：
本示例与 ch11_optimization.py 的策略逻辑、参数网格、数据源一致，
但参数传递路径不同，这是函数式与类风格的一处**真实能力差异**：

- 类风格：用类属性 IntParam 声明参数（含 ge/le/title 约束），
  通过 self.params.long_window 读取，run_grid_search 自动校验网格键名。
- 函数式：无类体，无法声明 IntParam。参数由 run_grid_search 的
  param_grid 经 context 注入 ctx，用 getattr(ctx, "long_window", 默认值) 读取，
  不享受 IntParam 的约束校验。

因此函数式版拼错网格键名不会报错，只会静默退回 initialize 里的默认值。
需要参数约束校验与页面化参数模型时，请用类风格。

运行耗时说明：
直接运行本脚本时，两版都会回退为单进程（策略定义在 __main__，见下方注释）。
本机实测 12 个组合、1000 根 Bar：类风格约 1.0 秒，函数式约 1.6 秒。
函数式略慢是因为每次回调多一跳 Python 转发（引擎的 on_bar -> 用户函数），
属固有成本；两版的网格搜索结果逐值一致，差异仅在速度。
"""

from typing import Any, cast

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar


# 模拟数据生成
def generate_mock_data(length: int = 1000) -> pd.DataFrame:
    """生成模拟数据（与类风格版一致）."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=length, freq="D")
    prices = 100 + np.cumsum(np.random.randn(length))
    return pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": 100000,
            "symbol": "MOCK",
        }
    )


def initialize(ctx: Any) -> None:
    """
    读取经 context 注入的扫描参数，并据此设置 warmup_period.

    param_grid 的每个组合都会新建一个策略实例并调用本函数，
    因此这里读到的就是当前组合的参数值；未注入时退回默认值。

    注意：函数式没有类体，warmup_period 必须像这样在 ctx 上显式赋值。
    引擎那一路“按指标调用自动推断 warmup”靠 AST 解析策略类体实现，
    函数式下被解析的是引擎内部的 FunctionalStrategy 而非本文件，推断结果恒为 0。
    """
    ctx.short_window = getattr(ctx, "short_window", 5)
    ctx.long_window = getattr(ctx, "long_window", 20)
    ctx.warmup_period = ctx.long_window + 1


def on_bar(ctx: Any, bar: Bar) -> None:
    """收到 Bar 事件的回调（与类风格版逻辑一致，self 全部换成 ctx）."""
    symbol = bar.symbol
    closes = ctx.get_history(count=ctx.long_window + 1, symbol=symbol, field="close")
    if len(closes) < ctx.long_window + 1:
        return

    history_closes = closes[:-1]
    ma_short = history_closes[-ctx.short_window :].mean()
    ma_long = history_closes[-ctx.long_window :].mean()

    pos = ctx.get_position(symbol)

    if ma_short > ma_long and pos == 0:
        ctx.order_target_percent(symbol=symbol, target_percent=0.95)
    elif ma_short < ma_long and pos > 0:
        ctx.close_position(symbol)


if __name__ == "__main__":
    df = generate_mock_data()

    print("开始运行第 11 章参数优化示例（函数式）...")
    print("正在进行网格搜索 (Grid Search)...")

    # 定义参数网格
    # 函数式下键名不受 IntParam 校验，会原样注入 ctx，须与 initialize 读取的名字一致
    param_grid = {"short_window": [3, 5, 10], "long_window": [15, 20, 30, 60]}

    # 与类风格版相同的并行回退结构：注意 Windows 下以 spawn 方式多进程时，策略必须
    # 可被子进程导入（不能定义在 __main__，详见第 11.4.0 节）。直接运行本脚本时
    # on_bar 属于 __main__，若并行不可用则自动回退为单进程，保证示例随处可跑。
    # cast(Any, on_bar)：run_grid_search 的 strategy 形参标注为 Type[Strategy]，
    # 运行期完全支持函数式策略，但静态类型上需显式放宽（同为函数式的“无类体”代价）。
    try:
        results: Any = aq.run_grid_search(
            strategy=cast(Any, on_bar),
            initialize=initialize,
            data=df,
            param_grid=param_grid,
            initial_cash=100_000,
            commission_rate=0.0003,
            max_workers=4,  # 限制为 4 个进程
        )
    except Exception as exc:  # noqa: BLE001 - 直接运行的脚本可能无法多进程，回退单进程
        print(f"并行优化不可用（{exc}），回退为单进程 (max_workers=1)。")
        results = aq.run_grid_search(
            strategy=cast(Any, on_bar),
            initialize=initialize,
            data=df,
            param_grid=param_grid,
            initial_cash=100_000,
            commission_rate=0.0003,
            max_workers=1,
        )

    # run_grid_search 默认 return_df=True，返回一个已按 sharpe_ratio 降序排好的
    # DataFrame，列中同时包含参数列（short_window/long_window）与指标列
    # （sharpe_ratio/total_return_pct/max_drawdown_pct 等）。
    print("\n" + "=" * 40)
    print("优化结果 (按夏普比率排序，前 5 名)")
    print("=" * 40)

    if isinstance(results, pd.DataFrame) and not results.empty:
        cols = [
            "short_window",
            "long_window",
            "sharpe_ratio",
            "total_return_pct",
            "max_drawdown_pct",
        ]
        available = [c for c in cols if c in results.columns]
        print(results[available].head().to_string(index=False))

        best = results.iloc[0]
        print("\n最佳参数组合:")
        print(
            f"  short_window={int(best['short_window'])}, "
            f"long_window={int(best['long_window'])}, "
            f"sharpe_ratio={float(best['sharpe_ratio']):.2f}"
        )
    else:
        print(results)
