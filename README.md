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
</p>

# AKQuant

**AKQuant** 是一个基于 **Rust** 和 **Python** 构建的高性能量化投研框架。它结合了 Rust 的极致性能和 Python 的易用性，为量化交易者提供强大的回测、风控及机器学习支持。

相比传统框架（如 Backtrader），AKQuant 拥有 **20倍+** 的回测性能提升，并原生支持 **Walk-forward Validation**（滚动训练）和 **Zero-Copy** 数据访问。

👉 **[阅读完整文档](docs/zh/index.md)** | **[English Documentation](docs/en/index.md)**

## 安装说明

**AKQuant** 已发布至 PyPI，无需安装 Rust 环境即可直接使用。

```bash
pip install akquant
```

## 快速开始

以下是一个简单的策略示例：

```python
import akshare as ak
import akquant as aq
from akquant import Strategy

# 1. 准备数据
# 使用 akshare 获取 A 股历史数据 (需安装: pip install akshare)
df = ak.stock_zh_a_daily(symbol="sh600000", start_date="20230101", end_date="20231231")

class MyStrategy(Strategy):
    def on_bar(self, bar):
        # 简单策略示例:
        # 当收盘价 > 开盘价 (阳线) -> 买入
        # 当收盘价 < 开盘价 (阴线) -> 卖出

        # 获取当前持仓
        current_pos = self.get_position(bar.symbol)

        if current_pos == 0 and bar.close > bar.open:
            self.buy(bar.symbol, 100)
            print(f"[{bar.timestamp_str}] Buy 100 at {bar.close:.2f}")

        elif current_pos > 0 and bar.close < bar.open:
            self.close_position(bar.symbol)
            print(f"[{bar.timestamp_str}] Sell 100 at {bar.close:.2f}")

# 运行回测
result = aq.run_backtest(
    data=df,
    strategy=MyStrategy,
    symbol="sh600000"
)

# 打印回测结果
print("\n=== Backtest Result ===")
print(result.metrics_df)
```

**运行结果示例:**

```text
=== Backtest Result ===
                            Backtest
annualized_return          -0.000575
end_market_value       999433.064610
equity_r2                   0.981178
initial_market_value  1000000.000000
max_drawdown                0.000567
max_drawdown_pct            0.056694
sharpe_ratio               -6.331191
sortino_ratio              -6.845218
std_error                  22.986004
total_return               -0.000567
total_return_pct           -0.056694
ulcer_index                 0.000306
upi                        -1.878765
volatility                  0.000091
win_rate                    0.339286
```

## 文档索引

*   📖 **[核心特性与架构](docs/zh/index.md#核心特性)**: 了解 AKQuant 的设计理念与性能优势。
*   🛠️ **[安装指南](docs/zh/installation.md)**: 详细的安装步骤（含源码编译）。
*   🚀 **[快速入门](docs/zh/quickstart.md)**: 更多示例与基础用法。
*   🤖 **[机器学习指南](docs/zh/ml_guide.md)**: 如何使用内置的 ML 框架进行滚动训练。
*   📚 **[API 参考](docs/zh/api.md)**: 详细的类与函数文档。
*   💻 **[贡献指南](CONTRIBUTING.md)**: 如何参与项目开发。

## License

MIT License
