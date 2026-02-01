# 示例集合

## 1. 基础示例 (Basic Examples)

*   [快速开始 (Quickstart)](quickstart.md): 包含手动数据回测和 AKShare 数据回测的完整流程。
*   [简单的均线策略 (SMA Strategy)](strategy_guide.md#3-编写类风格策略-class-based): 展示了如何使用类风格编写策略，并在 `on_bar` 中进行简单的交易逻辑。

## 2. 进阶示例 (Advanced Examples)

*   **Zipline 风格策略**: 展示了如何使用函数式 API (`initialize`, `on_bar`) 编写策略，适合从 Zipline 迁移的用户。
    *   参考代码: `examples/zipline_style_backtest.py` (如果存在) 或参考 [策略指南](strategy_guide.md#4-编写函数风格策略-functional)。

*   **多品种回测 (Multi-Asset)**:
    *   **期货策略**: 展示期货回测配置（保证金、乘数）。参考 [快速开始](quickstart.md#4-多品种回测-期货期权)。
    *   **期权策略**: 展示期权回测配置（权利金、按张收费）。参考 [策略指南](strategy_guide.md#10-多品种策略示例)。

*   **向量化指标 (Vectorized Indicators)**:
    *   展示如何使用 `IndicatorSet` 预计算指标以提高回测速度。参考 [策略指南](strategy_guide.md#51-向量化预计算-推荐---indicatorset)。

## 3. 更多资源

*   查看 `examples/` 目录下的源代码获取更多实用示例。
*   阅读 [API 文档](api.md) 了解详细接口定义。
