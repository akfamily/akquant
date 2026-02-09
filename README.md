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
from akquant import Strategy, run_backtest
from akquant.config import BacktestConfig

class MyStrategy(Strategy):
    def on_start(self):
        self.subscribe("600000")

    def on_bar(self, bar):
        # 简单的双均线逻辑示例
        if self.ctx.position.size == 0:
            self.buy(symbol=bar.symbol, quantity=100)
        elif bar.close > self.ctx.position.avg_price * 1.1:
            self.sell(symbol=bar.symbol, quantity=100)

# 运行回测
run_backtest(
    strategy=MyStrategy,
    symbol="600000",
    start_date="20230101",
    end_date="20231231"
)
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
