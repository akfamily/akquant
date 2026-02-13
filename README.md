<p align="center">
  <img src="assets/logo.svg" alt="AKQuant" width="400">
</p>

<p align="center">
    <a href="https://pypi.org/project/akquant/">
        <img src="https://img.shields.io/pypi/v/akquant?style=flat-square&color=007ec6" alt="PyPI Version">
    </a>
    <a href="https://pypi.org/project/akquant/">
        <img src="https://img.shields.io/pypi/pyversions/akquant?style=flat-square" alt="Python Versions">
    </a>
    <a href="LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
    </a>
    <a href="https://github.com/akfamily/akshare">
        <img src="https://img.shields.io/badge/Data%20Science-AKShare-green?style=flat-square" alt="AKShare">
    </a>
    <a href="https://pepy.tech/projects/akquant">
        <img src="https://static.pepy.tech/personalized-badge/akquant?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads" alt="Downloads">
    </a>
</p>

**AKQuant** 是一款专为量化投研设计的**下一代高性能混合框架**。核心引擎采用 **Rust** 编写以确保极致的执行效率，同时提供优雅的 **Python** 接口以维持灵活的策略开发体验。

🚀 **核心亮点：**

*   **极致性能**：得益于 Rust 的零开销抽象与 **Zero-Copy** 数据架构，回测速度较传统纯 Python 框架（如 Backtrader）提升 **X倍+**。
*   **原生 ML 支持**：内置 **Walk-forward Validation**（滚动训练）框架，无缝集成 PyTorch/Scikit-learn，让 AI 策略开发从实验到回测一气呵成。
*   **参数优化**：内置多进程网格搜索（Grid Search）框架，支持策略参数的高效并行优化。
*   **专业级风控**：内置完善的订单流管理与即时风控模块，支持多资产组合回测。

👉 **[阅读完整文档](https://akquant.akfamily.xyz/)** | **[English Documentation](https://akquant.akfamily.xyz/en/)**

## 安装说明

**AKQuant** 已发布至 PyPI，无需安装 Rust 环境即可直接使用。

```bash
pip install akquant
```

## 快速开始

以下是一个简单的策略示例：

```python
import akquant as aq
import akshare as ak
from akquant import Strategy

# 1. 准备数据
# 使用 akshare 获取 A 股历史数据 (需安装: pip install akshare)
df = ak.stock_zh_a_daily(symbol="sh600000", start_date="20250212", end_date="20260212")


class MyStrategy(Strategy):
    def on_bar(self, bar):
        # 简单策略示例:
        # 当收盘价 > 开盘价 (阳线) -> 买入
        # 当收盘价 < 开盘价 (阴线) -> 卖出

        # 获取当前持仓
        current_pos = self.get_position(bar.symbol)

        if current_pos == 0 and bar.close > bar.open:
            self.buy(symbol=bar.symbol, quantity=100)
            print(f"[{bar.timestamp_str}] Buy 100 at {bar.close:.2f}")

        elif current_pos > 0 and bar.close < bar.open:
            self.close_position(symbol=bar.symbol)
            print(f"[{bar.timestamp_str}] Sell 100 at {bar.close:.2f}")


# 运行回测
result = aq.run_backtest(
    data=df,
    strategy=MyStrategy,
    initial_cash=100000.0,
    symbol="sh600000"
)

# 打印回测结果
print("\n=== Backtest Result ===")
print(result)
```

**运行结果示例:**

```text
BacktestResult:
                                            Value
name
start_time              2025-02-12 00:00:00+08:00
end_time                2026-02-12 00:00:00+08:00
duration                        365 days, 0:00:00
total_bars                                    249
trade_count                                  62.0
initial_market_value                     100000.0
end_market_value                          99804.0
total_pnl                                  -196.0
unrealized_pnl                                0.0
total_return_pct                           -0.196
annualized_return                        -0.00196
volatility                               0.002402
total_profit                                548.0
total_loss                                 -744.0
total_commission                              0.0
max_drawdown                                345.0
max_drawdown_pct                         0.344487
win_rate                                22.580645
loss_rate                               77.419355
winning_trades                               14.0
losing_trades                                48.0
avg_pnl                                  -3.16129
avg_return_pct                          -0.199577
avg_trade_bars                           1.967742
avg_profit                              39.142857
avg_profit_pct                           3.371156
avg_winning_trade_bars                        4.5
avg_loss                                    -15.5
avg_loss_pct                            -1.241041
avg_losing_trade_bars                    1.229167
largest_win                                 120.0
largest_win_pct                         10.178117
largest_win_bars                              7.0
largest_loss                                -70.0
largest_loss_pct                        -5.380477
largest_loss_bars                             1.0
max_wins                                      2.0
max_losses                                    9.0
sharpe_ratio                            -0.816142
sortino_ratio                           -1.066016
profit_factor                            0.736559
ulcer_index                              0.001761
upi                                     -1.113153
equity_r2                                0.399577
std_error                                68.64863
calmar_ratio                            -0.568962
exposure_time_pct                       48.995984
var_95                                   -0.00023
var_99                                   -0.00062
cvar_95                                 -0.000405
cvar_99                                  -0.00069
sqn                                     -0.743693
kelly_criterion                         -0.080763
```

## 可视化 (Visualization)

AKQuant 内置了基于 **Plotly** 的强大可视化模块，仅需一行代码即可生成包含权益曲线、回撤分析、月度热力图等详细指标的交互式 HTML 报告。

```python
# 生成交互式 HTML 报告，自动在浏览器中打开
result.report(title="我的策略报告", show=True)

# 或者单独绘制仪表盘
import akquant.plot as aqp
aqp.plot_dashboard(result)
```

<p align="center">
  <img src="assets/dashboard_preview.png" alt="Strategy Dashboard" width="800">
  <br>
  👉 <a href="https://akquant.akfamily.xyz/report_demo/">点击查看交互式报表示例 (Interactive Demo)</a>
</p>

## 文档索引

*   📖 **[核心特性与架构](docs/zh/index.md#核心特性)**: 了解 AKQuant 的设计理念与性能优势。
*   🛠️ **[安装指南](docs/zh/installation.md)**: 详细的安装步骤（含源码编译）。
*   🚀 **[快速入门](docs/zh/quickstart.md)**: 更多示例与基础用法。
*   🤖 **[机器学习指南](docs/zh/ml_guide.md)**: 如何使用内置的 ML 框架进行滚动训练。
*   📚 **[API 参考](docs/zh/api.md)**: 详细的类与函数文档。
*   💻 **[贡献指南](CONTRIBUTING.md)**: 如何参与项目开发。

## Citation

Please use this bibtex if you want to cite this repository in your publications:

```bibtex
@misc{akquant,
    author = {Albert King and Zhangyao Jie},
    title = {AKQuant},
    year = {2026},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/akfamily/akquant}},
}
```

## License

MIT License
