# Runtime Config 指南

本文介绍如何在回测与实盘的入口侧控制策略运行时行为，而无需修改策略类代码。

## 1. 解决什么问题

通过 `strategy_runtime_config`，可以在调用处注入运行时开关：

- 错误处理模式（`error_mode`）
- 账户更新阈值（`portfolio_update_eps`）
- 精确交易日边界钩子（`enable_precise_day_boundary_hooks`）
- 兼容模式开关（`re_raise_on_error`）
- 指标计算模式（`indicator_mode`）

该能力同时支持：

- `run_backtest(...)`
- `run_from_checkpoint(...)`
- `run_live(...)`

## 2. 基础用法

```python
from akquant import StrategyRuntimeConfig, run_backtest

result = run_backtest(
    data=data,
    strategy=MyStrategy,
    strategy_runtime_config=StrategyRuntimeConfig(
        error_mode="continue",
        portfolio_update_eps=1.0,
    ),
)
```

也可以直接传 `dict`：

```python
result = run_backtest(
    data=data,
    strategy=MyStrategy,
    strategy_runtime_config={"error_mode": "continue"},
)
```

## 3. 参数行为对照表

| 字段 | 类型 | 默认值 | 常见用途 | 非法输入行为 |
|---|---|---|---|---|
| `error_mode` | `"raise" \| "continue" \| "legacy"` | `"raise"` | 控制用户回调异常处理策略 | 抛出 `ValueError` |
| `portfolio_update_eps` | `float`（`>= 0`） | `0.0` | 过滤微小资产波动噪声 | 抛出 `ValueError` |
| `enable_precise_day_boundary_hooks` | `bool` | `False` | 启用基于边界定时器的精确日内钩子 | 按 `bool` 规则转换 |
| `re_raise_on_error` | `bool` | `True` | 在 `error_mode="legacy"` 下作为兼容兜底 | 按 `bool` 规则转换 |
| `indicator_mode` | `"incremental" \| "precompute"` | `"precompute"` | 选择指标走增量计算还是全量预计算 | 抛出 `ValueError` |

## 4. 冲突优先级

当策略侧配置与外部配置冲突时，由 `runtime_config_override` 决定：

- `runtime_config_override=True`（默认）：应用外部配置
- `runtime_config_override=False`：保留策略侧配置

同一策略实例、同一冲突内容的告警会自动去重。

## 5. 常见误用与排障

- `strategy_runtime_config` 传入未知字段会快速失败，并给出字段级错误。
- `portfolio_update_eps` 传负值会触发校验错误。
- 同一策略实例重复运行时，相同冲突告警可能只出现一次，这是去重行为。
- 当 `runtime_config_override=False` 时，即使传入外部配置也不会覆盖策略侧配置。

## 6. 热启动注入

从快照恢复时也可以覆盖运行时行为：

```python
result = run_from_checkpoint(
    checkpoint_path="snapshot.pkl",
    data=new_data,
    symbols="TEST",
    strategy_runtime_config={"error_mode": "continue"},
)
```

冲突处理规则与 `run_backtest` 完全一致。

## 7. 实盘下发（run_live）

`run_live(...)` 接受与 `run_backtest(...)` 同名的两个参数，配置会下发到主策略**与每个槽位子策略**：

```python
from akquant import StrategyRuntimeConfig
from akquant.live import run_live

run_live(
    strategy_cls=MyStrategy,
    instruments=instruments,
    broker="ctp",
    strategy_runtime_config=StrategyRuntimeConfig(error_mode="continue"),
)
```

典型用途是同一份策略代码在两侧用不同容错口径——回测用 `error_mode="raise"` 让问题尽早暴露，实盘用 `"continue"` 避免单个回调异常中断交易：

```python
run_backtest(data=data, strategy=MyStrategy,
             strategy_runtime_config={"error_mode": "raise"})
run_live(strategy_cls=MyStrategy, instruments=instruments,
         strategy_runtime_config={"error_mode": "continue"})
```

两点与回测的差异需要注意：

- **不传就完全不动**。策略内部自己写的 `self.runtime_config` 在实盘本来就生效（消费点直读该属性，与运行模式无关），不传该参数时它保持原样。
- **没有 `strategy_config` 兜底注入**。回测会从 `strategy_config.indicator_mode` 推一份默认值，实盘入口没有 config 对象参数，所以实盘想改 `indicator_mode` 必须显式传 `strategy_runtime_config`。

冲突检测、告警去重与 `runtime_config_override` 的语义与回测共用同一套实现，行为逐字一致。

## 8. 端到端示例

可直接运行：

- [22_strategy_runtime_config_demo.py](https://github.com/akfamily/akquant/blob/main/examples/22_strategy_runtime_config_demo.py)
- [44_strategy_source_loader_demo.py](https://github.com/akfamily/akquant/blob/main/examples/44_strategy_source_loader_demo.py)
- [69_live_runtime_config_dispatch_demo.py](https://github.com/akfamily/akquant/blob/main/examples/69_live_runtime_config_dispatch_demo.py)：实盘下发，含槽位子策略与 `runtime_config_override=False`（走 `broker="replay"`，无需连接柜台）

另见：[热启动指南](warm_start.md)。

预期输出标记（22 号）：

- `scenario1_done`
- `scenario2_exception=...`
- `scenario3_done`

预期输出标记（69 号）：

- `scenario1_done bars_seen=3` —— 实盘 `error_mode="continue"` 吞掉每根 bar 的异常，会话跑完
- `scenario2_exception=RuntimeError: ...` —— 同一策略在回测 `error_mode="raise"` 下第一根 bar 就抛出
- `slot on_start error_mode=continue` —— 配置确实下发到了槽位子策略
- `scenario4_kept=continue` —— `runtime_config_override=False` 保留了策略自设值

## 9. 故障速查清单

| 现象 / 错误信息 | 常见原因 | 快速修复 |
|---|---|---|
| `strategy_runtime_config contains unknown fields: ...` | 注入字典包含未知字段 | 删除未支持字段，仅保留文档中的字段名 |
| `invalid strategy_runtime_config: portfolio_update_eps must be >= 0` | `portfolio_update_eps` 传了负值 | 将 `portfolio_update_eps` 设置为 `0` 或正数 |
| 传了 runtime 配置但策略行为没变化 | 启用了 `runtime_config_override=False` | 改为 `runtime_config_override=True` 或移除该参数 |
| 冲突告警只出现一次 | 告警按“同一策略实例 + 同一冲突内容”去重 | 这是预期行为；如需重复观察可新建策略实例 |
| 热启动后仍抛出回调异常 | 恢复后的策略配置生效，外部覆盖未应用 | 传入 `strategy_runtime_config={"error_mode": "continue"}` 且确保 `runtime_config_override=True` |

## 10. 动态策略加载（strategy_source / strategy_loader）

`run_backtest(...)` 支持在调用时通过策略源码动态加载策略：

- `strategy_source`：策略输入，支持文件路径（`str` / `PathLike`）或 `bytes`
- `strategy_loader`：加载器名称，默认 `python_plain`
- `strategy_loader_options`：加载器参数字典

默认加载器：

- `python_plain`：从本地 Python 文件加载策略
- `encrypted_external`：通过外部回调解密并返回策略对象

### 10.1 python_plain 示例

```python
result = run_backtest(
    data=data,
    strategy=None,
    strategy_source="my_strategy.py",
    strategy_loader="python_plain",
    strategy_loader_options={"strategy_attr": "MyStrategy"},
)
```

### 10.2 encrypted_external 示例

```python
def decrypt_and_load(source, options):
    ...
    return MyStrategy

result = run_backtest(
    data=data,
    strategy=None,
    strategy_source=b"...encrypted-bytes...",
    strategy_loader="encrypted_external",
    strategy_loader_options={"decrypt_and_load": decrypt_and_load},
)
```

### 10.3 与 run_from_checkpoint 的关系

`run_from_checkpoint(...)` 当前从 checkpoint 恢复策略实例，不会通过
`strategy_source` / `strategy_loader` 重新加载策略实现。

## 11. broker_profile 选择建议

`run_backtest(..., broker_profile=...)` 可快速注入一组费率/滑点/手数默认值，适合在“参数还未完全定稿”阶段快速对齐不同执行风格。

优先级规则：

- 显式参数优先于 `broker_profile` 模板值
- 模板值优先于系统默认值

| 模板名 | 推荐场景 | 主要特征 | 典型风险 |
|---|---|---|---|
| `cn_stock_miniqmt` | A 股常规仿真、对齐 MiniQMT 基础口径 | 默认佣金 + 印花税 + 过户费 + 最小佣金 + 百股一手 | 对极端冲击成本刻画偏保守 |
| `cn_stock_t1_low_fee` | 低费率账户压力测试、策略净值敏感性分析 | 更低佣金/过户费、较低最小佣金 | 可能高估高换手策略净收益 |
| `cn_stock_sim_high_slippage` | 盘中冲击/流动性压力场景、稳健性回归 | 较高滑点、较保守成交约束 | 可能低估低冲击策略表现 |

模板参数明细（当前内置值）：

| 模板名 | commission_rate | stamp_tax_rate | transfer_fee_rate | min_commission | slippage | volume_limit_pct | lot_size |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cn_stock_miniqmt` | 0.0003 | 0.001 | 0.00001 | 5.0 | 0.0002 | 0.2 | 100 |
| `cn_stock_t1_low_fee` | 0.0002 | 0.001 | 0.000005 | 3.0 | 0.0001 | 0.25 | 100 |
| `cn_stock_sim_high_slippage` | 0.0003 | 0.001 | 0.00001 | 5.0 | 0.001 | 0.1 | 100 |

快速示例：

```python
result = run_backtest(
    data=data,
    strategy=MyStrategy,
    symbols="000001.SZ",
    broker_profile="cn_stock_t1_low_fee",
    show_progress=False,
)
```

如果你已有明确的券商实盘参数，建议直接显式传入 `commission_rate`、`slippage`、`lot_size` 等字段，作为最终基线。
