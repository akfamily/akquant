# 热启动 (Warm Start) 指南

AKQuant 提供了强大的**热启动 (Warm Start)** 功能，允许你保存回测引擎的当前状态（快照），并在未来某个时间点恢复该状态继续运行。这对于长周期策略的分段回测、滚动训练（Rolling Window）以及模拟实盘环境非常有用。

## 1. 什么是热启动？

在传统的事件驱动回测中，每次运行都是从头开始（Cold Start），所有状态（持仓、资金、指标历史）都需要从零积累。

**热启动**允许你：
1.  **保存 (Snapshot)**: 将引擎的当前内存状态（包括持仓、未成交订单、策略变量、指标状态等）序列化保存到磁盘文件。
2.  **恢复 (Resume)**: 从磁盘文件加载状态，恢复引擎和策略，就像中间没有中断过一样，接着处理新的数据。

## 2. 基本用法

### 2.1 保存快照 (Phase 1)

在第一阶段回测结束时（或在策略逻辑中任意时刻），调用 `save_checkpoint` 保存状态。

```python
from akquant.checkpoint import save_checkpoint

# 运行第一阶段回测
result1 = run_backtest(data=data_phase1, strategy=MyStrategy, ...)

# 保存快照到文件
checkpoint_file = "checkpoint_phase1.pkl"
save_checkpoint(result1.engine, result1.strategy, checkpoint_file)
print(f"Snapshot saved to {checkpoint_file}")
```

### 2.2 恢复并继续运行 (Phase 2)

使用 `run_from_checkpoint` 函数从快照恢复，并传入第二阶段的数据。

**注意**：快照仅保存了动态状态（Portfolio, Orders, Strategy Attributes），**不包含** 静态配置（Instrument, MarketModel）。因此在恢复时，你**必须**重新配置这些静态信息。

```python
import akquant as aq

# 准备第二阶段数据
data_phase2 = ...

config = aq.BacktestConfig(
    strategy_config=aq.StrategyConfig(
        strategy_id="alpha",
        strategies_by_slot={"beta": BetaStrategy},
        strategy_max_order_size={"alpha": 10, "beta": 20},
    )
)

# 从快照恢复并运行
result2 = aq.run_from_checkpoint(
    checkpoint_path="checkpoint_phase1.pkl",
    data=data_phase2,
    symbols="AAPL",  # 你的主要标的
    # 重要：必须重新传入市场费用配置，因为 MarketModel 不会被保存
    commission_rate=0.0003,
    stamp_tax_rate=0.001,
    transfer_fee_rate=0.00001,
    config=config,
    t_plus_one=True  # 如果是 A 股
)
```

## 3. 策略适配 (关键)

为了支持热启动，你的策略代码需要正确处理**初始化 (Initialization)** 和 **恢复 (Restoration)** 的区别。

### 3.1 生命周期钩子

AKQuant 为策略提供了两个关键的启动钩子：

*   `on_start()`: 无论冷启动还是热启动，**都会调用**。用于通用初始化（如订阅行情）。
*   `on_resume()`: **仅在热启动时**调用（在 `on_start` 之前）。用于恢复特定的连接或资源。函数式入口对应 `on_resume(ctx)`。

### 3.2 避免覆盖状态

最常见的错误是在 `on_start` 中无条件地重新初始化指标，导致从快照恢复的指标状态被覆盖（重置为 0）。

**错误写法 ❌**：

```python
def on_start(self):
    # 错误！如果从快照恢复，self.sma 已经有值了，这里会将其覆盖为新对象
    self.sma = SMA(30)
    self.subscribe(self.symbol)
```

**正确写法 ✅**：

使用 `self.is_restored` 属性判断当前是否为恢复模式。

```python
def on_start(self):
    # 1. 初始化指标 (仅在冷启动时)
    if not self.is_restored:
        self.sma = SMA(30)
        # 初始化其他非持久化状态...
    else:
        self.log("Resumed from snapshot. Indicators retained.")

    # 2. 注册指标 (必须执行，以便 Engine 知道需要更新它)
    self.register_precomputed_indicator("sma", self.sma)

    # 3. 订阅行情 (必须执行，因为连接是临时的)
    self.subscribe(self.symbol)
```

### 3.3 指标持久化

AKQuant 内置的指标（如 `SMA`, `EMA`）已经支持 Pickle 序列化。如果你使用自定义指标或第三方库（如 TA-Lib），请确保它们支持 `pickle`，或者在 `__getstate__` 和 `__setstate__` 中手动处理状态保存。关于自定义指标的完整接入方式，可继续阅读：[自定义指标指南](../guide/custom_indicator.md)。

## 4. 注意事项

1.  **Instrument 需重新注册**：`run_from_checkpoint` 会尝试自动为新数据中的 Symbol 注册默认 Instrument。如果你的策略依赖特定的 `lot_size` 或 `multiplier`，建议在 `on_start` 中手动检查并调用 `self.ctx.engine.add_instrument(...)`。**注意**：若传了 `symbols` 白名单，白名单外的标的连数据都不会被加载，自然也就不会触发默认 Instrument 注册——`symbols` 表示「只跑这些标的」，不在其中的 Symbol 在本次恢复运行里完全不存在。
2.  **MarketModel 重置**：费用设置（佣金、印花税）和交易规则（T+1）不会保存在快照中。务必在 `run_from_checkpoint` 参数中重新传入正确配置（可通过显式参数或 `config.strategy_config`），包括 `commission_policy` / `commission_rate`、`stamp_tax_rate`、`transfer_fee_rate`（`stamp_tax`、`transfer_fee` 仍兼容）。
3.  **初始资金显示**：`result2.metrics.initial_cash` 会自动调整为恢复时的资金，确保收益率计算是基于第二阶段的实际起始资金，而不是账户的历史初始资金。
4.  **数据连续性**：确保 Phase 1 的结束时间与 Phase 2 的开始时间是连续的。如果中间有长时间中断，指标计算可能会出现跳跃。
5.  **`get_history()` 连续性**：新版本快照会一并保存并恢复历史缓冲，因此 `run_from_checkpoint` 恢复后，`get_history()`、`get_history_map()` 等接口会延续 Phase 1 的历史窗口；正常续跑时不再需要额外手工拼接 lookback 数据。
6.  **运行时配置注入**：可通过 `strategy_runtime_config` 在恢复阶段覆盖错误处理和快照阈值等运行时行为。
7.  **策略级风控状态可恢复**：策略限额、策略现金流、日损基线、回撤峰值、仅平仓激活态等会随快照保存并恢复，便于断点续跑后保持风控行为连续。
8.  **默认时区**：`run_from_checkpoint` 未显式传入 `timezone` 时，默认使用 `Asia/Shanghai`。

## 5. 完整示例

请参考项目中的 [21_warm_start_demo.py](https://github.com/akfamily/akquant/blob/main/examples/21_warm_start_demo.py) 获取类风格完整示例；如果你使用函数式入口，可直接参考 [56_functional_warm_start_demo.py](https://github.com/akfamily/akquant/blob/main/examples/56_functional_warm_start_demo.py)；如果你还需要验证 `strategies_by_slot` 下多 slot 一起恢复，可参考 [57_functional_multi_slot_warm_start_demo.py](https://github.com/akfamily/akquant/blob/main/examples/57_functional_multi_slot_warm_start_demo.py)。

```python
# 示例摘要
class MyStrategy(Strategy):
    def on_start(self):
        if not self.is_restored:
            self.sma = SMA(10)
        self.register_precomputed_indicator("sma", self.sma)

# ... 运行 Phase 1 ...
save_checkpoint(engine, strategy, "checkpoint.pkl")

# ... 运行 Phase 2 ...
run_from_checkpoint("checkpoint.pkl", data_new, ...)
```

## 5.x 多阶段结果合并 (`merge_results`)

分阶段续跑时，每段 `BacktestResult` 的曲线只覆盖本段。用 `merge_results` 把多段按
时间顺序拼成一条完整曲线，无需手写循环拼接：

```python
import akquant as aq

r1 = aq.run_backtest(data=phase1, strategy=MyStrategy, symbols="X", initial_cash=1e6)
aq.save_checkpoint(r1.engine, r1.strategy, "ckpt.pkl")
r2 = aq.run_from_checkpoint("ckpt.pkl", data=phase2, symbols="X")

merged = aq.merge_results(r1, r2)
print(merged.equity_curve)          # 跨两段的完整权益曲线
print(merged.metrics.total_return_pct)
returns = merged.to_quantstats()    # 交给 quantstats 出报告
```

要点：

- 各段须**时间递增、互不重叠**（允许 gap）；重叠会抛 `ValueError`。
- `dedupe_boundary=True`（默认）去除相邻段的边界重复时间戳。
- `drop_expired_instruments=True`（默认）依据 instrument snapshot 的 `expiry_date`
  清理已退市合约持仓，避免长区间资产爆炸。
- `merged.metrics` 为**核心指标子集**（total_return / max_drawdown / sharpe /
  sortino / calmar / win_rate / profit_factor 等），口径对齐单段回测；依赖引擎内部
  态的完整 60 项指标请在单段 `BacktestResult` 上读取。

## 6. 推荐阅读

- `run_from_checkpoint` 参数详情：[API 参考](../reference/api.md#akquantrun_from_checkpoint)
- 恢复阶段运行时覆盖：[Runtime Config 指南](runtime_config.md)
- 多 slot 连续性与策略级风控映射：[多策略指南](multi_strategy_guide.md)
