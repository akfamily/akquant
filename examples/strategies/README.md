# 常用策略示例 (Common Strategies)

本目录包含一系列常用量化策略的实现代码，旨在帮助用户快速上手 AKQuant 策略开发。

这些示例展示了如何结合 [AKShare](https://github.com/akfamily/akshare) 获取数据并进行回测，但也适用于其他数据源。

## 核心提示 (Key Concepts)

在使用 AKQuant 编写策略时，请注意以下核心机制：

1.  **数据获取 (`get_history`)**:
    -   `self.get_history(count=N)` 返回的是**包含当前 Bar** 的最近 N 条数据。
    -   **计算均线 (MA)**: 直接使用 `get_history(N)` 即可（包含今日收盘价）。
    -   **计算突破信号 (Breakout)**: 如果需要基于*昨日*收盘价计算指标（避免未来函数），请获取 `N+1` 条数据并切片 `[:-1]` 剔除当前 Bar。

2.  **多标的回测**:
    -   `run_backtest` 的 `data` 参数支持传入字典 `{symbol: DataFrame}`。
    -   在策略中通过 `self.get_history(..., symbol=s)` 获取指定标的数据。

## 案例列表

### 1. [01_stock_dual_moving_average.py](./01_stock_dual_moving_average.py) - A股双均线策略
- **目标**: 演示如何获取单只股票（如平安银行 sz000001）的历史数据。
- **策略**: 双均线策略 (Golden Cross / Death Cross)。
- **核心点**:
    - 数据清洗与复权处理 (`adjust="qfq"`).
    - `get_history` 的基本使用（包含当前 Bar 数据用于计算当日 MA）。

### 2. [02_stock_grid_trading.py](./02_stock_grid_trading.py) - 股票网格交易
- **目标**: 演示股票网格交易策略逻辑。
- **策略**: 动态网格策略，价格下跌分批买入，上涨分批卖出。
- **核心点**:
    - `on_bar` 中的持仓状态管理。
    - 复杂交易逻辑的实现（基于上次成交价的网格）。

### 3. [03_stock_atr_breakout.py](./03_stock_atr_breakout.py) - 股票 ATR 通道策略
- **目标**: 演示基于波动率的通道突破策略。
- **策略**: ATR 通道突破策略。
- **核心点**:
    - **避免未来函数**: 演示如何剔除当前 Bar 数据来计算基于历史的指标（ATR/通道上下轨）。
    - 波动率指标计算。

### 4. [04_stock_momentum_rotation.py](./04_stock_momentum_rotation.py) - 多股票轮动
- **目标**: 演示多只股票（如 贵州茅台 vs 五粮液）的数据获取与组合管理。
- **策略**: 动量轮动策略 (Momentum Rotation)。
- **核心点**:
    - 多标的数据传入 (`Dict[str, DataFrame]`).
    - 跨标的动量比较与换仓逻辑.
    - `order_target_percent` 的正确使用。

### 5. [05_stock_momentum_rotation_timer.py](./05_stock_momentum_rotation_timer.py) - `on_daily_rebalance` 横截面轮动
- **目标**: 演示 `on_daily_rebalance` 的前一快照语义。
- **策略**: 交易日边界触发的动量轮动策略。
- **核心点**:
    - 每个交易日最多触发一次。
    - 回调内只看到前一交易日 / 前一账户快照。
    - 适合作为日边界准备和统一调仓入口。

### 6. [06_stock_momentum_rotation_bucket.py](./06_stock_momentum_rotation_bucket.py) - 收齐时间片后横截面轮动
- **目标**: 演示不使用定时器时，如何在 `on_bar` 中收齐同一 `timestamp` 后再执行一次横截面逻辑。
- **策略**: 时间片缓存 + 动量轮动策略。
- **核心点**:
    - 使用 `timestamp -> set(symbol)` 的缓存桶判断是否收齐。
    - 收齐后只执行一次跨标的打分与调仓。
    - 适用于无固定定时触发点的横截面策略。

### 7. [07_stock_momentum_rotation_on_timer.py](./07_stock_momentum_rotation_on_timer.py) - `on_timer` 风格的横截面轮动
- **目标**: 演示把横截面打分与调仓逻辑集中放到 `on_timer` 中。
- **策略**: 固定时点触发的动量轮动策略。
- **核心点**:
    - `add_daily_timer()` 与 `on_timer()` 的基本配合。
    - 将多标的横截面逻辑从 `on_bar` 中解耦出来。
    - 适合作为生产环境中更稳定的统一调仓入口。

### 8. [08_target_positions_long_short.py](./08_target_positions_long_short.py) - 高级目标仓位与多空切换
- **目标**: 演示 `rebalance_positions()` 如何在一个调用里同时表达多头与空头目标。
- **策略**: 先建立多头，再在同周期内切换为 `AAA` 空头和 `BBB` 多头。
- **核心点**:
    - `rebalance_positions()` 支持正负目标仓位。
    - `allow_short=True` 与 `RiskConfig(account_mode="margin", enable_short_sell=True)` 的配合。
    - `get_last_target_positions_plan()` 用于解释最近一次调仓计划。

### 9. [09_stock_momentum_rotation_after_bar.py](./09_stock_momentum_rotation_after_bar.py) - `on_daily_rebalance_after_bar` 横截面轮动
- **目标**: 演示 `on_daily_rebalance_after_bar` 的当日可见语义。
- **策略**: 首个跨标的完整 bar 切片后的动量轮动策略。
- **核心点**:
    - 框架会在当日首个完整切片后触发一次。
    - 回调内可见当日历史和当前账户快照。
    - 适合收盘价同周期调仓或依赖当日完整横截面的场景。

## 使用方法

直接运行对应的 Python 脚本即可：

```bash
python examples/strategies/01_stock_dual_moving_average.py
```
