# 参数优化指南

参数优化是量化策略开发中至关重要的一环。AKQuant 提供了强大的优化工具，帮助你探索参数对策略表现的影响，并评估策略的稳健性。

目前支持两种主要的优化模式：

1.  **网格搜索 (Grid Search)**: `run_grid_search`
2.  **滚动优化 (Walk-Forward Optimization)**: `run_walk_forward`

---

## 1. 网格搜索 (Grid Search)

网格搜索是一种穷举式的参数优化方法。它通过遍历给定的参数组合，在同一段历史数据上运行回测，从而找到表现最优的参数组合。

### 适用场景
*   探索策略对参数的敏感性。
*   确定参数的大致合理范围。
*   寻找策略在历史数据上的"理论上限"。

### 使用方法

使用 `akquant.run_grid_search` 函数：

```python
from akquant import FloatParam, IntParam, run_grid_search, Strategy

# 1. 定义策略：参数用内联字段声明，经 self.params.<name> 访问
class MyStrategy(Strategy):
    ma_period = IntParam(10, ge=2, le=200)
    stop_loss = FloatParam(0.02, ge=0.0, le=1.0)

    # ... on_start / on_bar 中用 self.params.ma_period / self.params.stop_loss

# 2. 定义参数网格（键名须与内联字段名一致）
param_grid = {
    "ma_period": [10, 20, 30],
    "stop_loss": [0.01, 0.02, 0.05]
}

# 3. 运行网格搜索
results = run_grid_search(
    strategy=MyStrategy,
    param_grid=param_grid,
    data=df,
    sort_by="sharpe_ratio",  # 按夏普比率排序
    ascending=False          # 降序
)

print(results.head())
```

### 参数模型驱动优化（推荐）

当你需要把策略接入页面配置（例如 Web UI / API）时，AKQuant 的内联参数字段
天然就是**单一事实来源**：同一套 `IntParam` / `FloatParam` / ... 声明，既用于
`self.params.<name>` 运行期访问，也可以直接导出 schema、单独校验参数，或喂给
`param_grid` 做离散组合搜索——不需要再单独维护一个 `ParamModel` 子类。

```python
from akquant import (
    IntParam,
    Strategy,
    get_strategy_param_schema,
    validate_strategy_params,
    run_grid_search,
)


class SmaStrategy(Strategy):
    fast_period = IntParam(10, ge=2, le=200, title="快线")
    slow_period = IntParam(30, ge=3, le=500, title="慢线")

    def on_start(self):
        # 派生初始化统一放在 on_start，此时 self.params 已就绪
        ...


schema = get_strategy_param_schema(SmaStrategy)
runtime_params = validate_strategy_params(
    SmaStrategy,
    {"fast_period": 12, "slow_period": 36},
)

results = run_grid_search(
    strategy=SmaStrategy,
    data=df,
    param_grid={"fast_period": [5, 10, 15], "slow_period": [20, 30, 60]},
)
```

`param_grid` 的键名必须和内联字段名完全一致；`strict_strategy_params=True`
（默认）下，未知键或越界取值（超出 `ge`/`le`、不在 `choices` 内）都会在校验阶段
直接报错，而不是静默回退到默认值。

### 多目标优化 (Multi-Objective Optimization)

在实际交易中，单一指标往往存在局限性（例如仅看夏普比率可能选出只交易一次的幸存者）。AKQuant 支持基于多个指标进行排序：

```python
results = run_grid_search(
    strategy=MyStrategy,
    param_grid=param_grid,
    data=df,
    sort_by=["sharpe_ratio", "total_return"], # 优先按夏普，其次按总收益
    ascending=[False, False]                  # 均为降序
)
```

### 结果筛选 (Result Filtering)

为了避免“幸存者偏差”或过滤掉不符合风控要求的参数组合（例如交易次数过少、最大回撤过大），你可以使用 `result_filter` 回调函数：

```python
def result_filter(metrics):
    # 筛选条件：
    # 1. 交易次数 >= 50
    # 2. 夏普比率 > 1.0
    # 3. 最大回撤 < 20%
    return (
        metrics.get("closed_trade_count", 0) >= 50 and
        metrics.get("sharpe_ratio", 0) > 1.0 and
        metrics.get("max_drawdown_pct", 1.0) < 0.2
    )

results = run_grid_search(
    ...,
    result_filter=result_filter
)
```

### 核心参数

*   `strategy`: 策略类。
*   `param_grid`: 参数字典，key 为参数名，value 为参数值列表。
*   `data`: 回测数据。
*   `sort_by`: 排序指标，支持字符串或字符串列表。
*   `ascending`: 排序方向，支持布尔值或布尔值列表。
*   `result_filter`: (可选) 结果筛选回调函数，用于过滤不符合条件的组合。
*   `warmup_calc`: (可选) 动态计算预热期的回调函数。
*   `constraint`: (可选) 参数约束回调函数，用于过滤无效组合。
*   `forward_worker_logs`: (可选) 并行优化时是否将子进程 `self.log()` 回传到主进程日志。
    *   `False` (默认): 性能更优，日志可能在主进程不可见。
    *   `True`: 主进程可聚合子进程日志，适合排障与教学演示。
*   `strict_strategy_params`: (默认 `True`，由 `run_grid_search` 注入到 `run_backtest`)。
    *   启用后会严格校验 `param_grid` 是否与策略内联参数字段（`IntParam`/`FloatParam`/... 声明）匹配；
    *   发现未知参数或越界取值时立即抛错，避免“静默回退”导致优化结果失真；
    *   关闭时 (`False`) 未知键会回退到字段默认值构造，而不是报错。

### 资源控制与异常处理 (Resource Control & Error Handling)

当参数组合数量庞大或某些组合可能导致死循环/内存溢出时，可以通过以下参数进行控制：

```python
results = run_grid_search(
    ...,
    timeout=60.0,           # 单次任务最长运行 60 秒，超时则跳过
    max_tasks_per_child=1   # 强制每个 Worker 进程只执行 1 个任务后重启
)
```

*   `timeout`: 单个回测任务的超时时间（秒）。如果任务在指定时间内未完成，将被标记为失败并跳过。这对于防止某些参数导致的死循环非常有用。
*   `max_tasks_per_child`: Worker 进程重启频率。设置为 `1` 可以强制每次任务都使用新的进程，有效防止内存泄漏（OOM）或清理超时残留的线程资源。

### Windows 并行执行注意事项（`max_workers > 1`）

在 Windows 上，`run_grid_search` / `run_walk_forward` 的多进程并行使用 `spawn` 启动方式。此时：

*   策略类必须定义在**可导入模块**中，不能定义在 `__main__`（例如脚本内临时类）。
*   脚本入口需要使用 `if __name__ == "__main__":` 保护。
*   这属于 Python 多进程机制限制，**不是成交策略语义本身的问题**。

推荐写法：

```python
from my_strategy_module import TailTradingStrategy
from akquant import run_grid_search


def main() -> None:
    results = run_grid_search(
        strategy=TailTradingStrategy,
        param_grid=param_grid,
        data=all_data,
        max_workers=4,
    )
    print(results.head())


if __name__ == "__main__":
    main()
```

### 并行日志可见性与提示规则

在 `max_workers > 1` 场景下，日志提示与 `forward_worker_logs` 的关系如下：

*   `forward_worker_logs=False`：会提示“子进程日志可能在主进程不可见”。
*   `forward_worker_logs=True` 且主进程存在有效 logger handler：启用日志回传，不再显示“不可见”提示。
*   `forward_worker_logs=True` 但主进程无有效 logger handler：会提示“请求了日志回传但主进程无可用 handler”。

推荐在运行优化前显式配置一次日志，尤其是你希望查看主进程聚合后的 worker 日志时：

```python
import akquant

akquant.configure_logging(
    akquant.LogConfig(
        profile="optimize",
        level="INFO",
        console=True,
        filename="logs/optimize.log",
        file_level="DEBUG",
        file_json=True,
        file_max_bytes=50_000_000,
        file_backup_count=3,
    )
)
```

说明：

*   `profile="optimize"` 会在默认格式中带上进程名，便于区分不同 worker。
*   `forward_worker_logs=True` 只负责“把子进程日志送回主进程”，不负责自动创建 handler。
*   如果你只想快速打开控制台日志，也可以继续使用 `akquant.register_logger(level="INFO")`。
*   如果你希望文件日志更方便被日志平台消费，可额外打开 `file_json=True`。
*   开启日志聚合后，来自 Rust 执行链路的 warning 也会回传到主进程。例如保证金不足拒单、收盘过期、未知取消订单、`same-cycle` 延后等执行 warning，会按统一结构化字段输出，便于批量筛查 `phase=execution`、`order_id`、`strategy_id` 等信息。

### 持久化与断点续传 (Persistence & Resume)

对于参数组合极多（如 > 10,000 组）的场景，单机运行可能需要数小时甚至数天。如果中途断电或程序崩溃，重新运行将非常耗时。AKQuant 支持将优化结果实时写入 SQLite 数据库，并支持断点续传。

```python
results = run_grid_search(
    ...,
    db_path="optimization_results.db"  # 指定数据库路径
)
```

*   `db_path`: SQLite 数据库文件路径。
    *   **实时保存**: 每完成一个参数组合的回测，结果（参数、指标、耗时、错误信息）都会立即写入数据库。
    *   **断点续传**: 程序启动时会自动检查数据库。如果发现某个参数组合已经运行过（基于参数的 JSON 匹配），将直接跳过，只运行剩余的任务。
    *   **结果复用**: 你可以编写脚本从数据库中读取结果进行后续分析，而无需重新运行回测。

### ⚠️ 风险提示
网格搜索容易导致 **过拟合 (Overfitting)**。选出的"最优参数"可能只是恰好适应了这段历史数据的噪声，在未来实盘中往往失效。切勿直接使用网格搜索的最优结果进行实盘。

---

## 2. 滚动优化 (Walk-Forward Optimization)

滚动优化（Walk-Forward Analysis/Optimization, WFO）是一种更接近实战的验证方法。它模拟了真实的时间流逝，将数据切分为多个 `[训练集 | 测试集]` 窗口，不断滚动进行"样本内优化"和"样本外验证"。

### 适用场景
*   验证策略的真实稳健性。
*   评估策略在"未知"数据上的表现。
*   生成随时间动态调整的参数路径。

### 原理
WFO 的核心思想是：**永远只用过去的数据来决定现在的参数**。

1.  **窗口 1**:
    *   **训练 (In-Sample)**: 使用 1-3月数据进行 Grid Search，找到最优参数 A。
    *   **测试 (Out-of-Sample)**: 使用参数 A 在 4月数据上进行回测。
2.  **窗口 2**:
    *   **训练**: 滚动到 2-4月，重新 Grid Search，找到最优参数 B。
    *   **测试**: 使用参数 B 在 5月数据上进行回测。
3.  **拼接**: 将所有测试段的结果拼接，形成最终的资金曲线。

### 使用方法

使用 `akquant.run_walk_forward` 函数：

```python
from akquant import run_walk_forward

# 运行 WFO
wfo_results = run_walk_forward(
    strategy=MyStrategy,
    param_grid=param_grid,
    data=df,
    train_period=250,      # 训练窗口长度 (例如250个Bar)
    test_period=60,        # 测试/滚动步长 (例如60个Bar)
    metric="sharpe_ratio", # 优化目标
    initial_cash=100_000.0,
    warmup_calc=warmup_calc, # 支持动态预热
    constraint=param_constraint, # 支持参数约束
    max_workers=4,
    forward_worker_logs=True,    # 样本内并行优化日志回传
    strict_strategy_params=True, # 严格参数校验（推荐保持开启）
)

# wfo_results 包含拼接后的资金曲线和每段使用的参数
print(wfo_results)
```

### 核心参数

*   `train_period`: 训练窗口长度 (Bar数量)。窗口越长，参数越稳定；窗口越短，适应变化越快。
*   `test_period`: 测试窗口长度 (Bar数量)。通常也是滚动的步长。
*   `metric`: 在训练集上选择最优参数的依据指标。支持单个字符串 (如 `"sharpe_ratio"`) 或字符串列表 (如 `["sharpe_ratio", "total_return"]`)。
*   `ascending`: 排序方向，与 `metric` 对应。支持布尔值或布尔值列表 (默认 `False`，即降序)。
*   `result_filter`: (可选) 结果筛选回调函数，在每个训练窗口中过滤不符合条件的参数组合。
*   `kwargs` 透传规则：
    *   样本内优化阶段透传给 `run_grid_search`；
    *   样本外验证阶段透传给 `run_backtest`；
    *   因此可在 WFO 中继续使用 `forward_worker_logs` 与 `strict_strategy_params`。

---

## 3. 高级功能

### 动态预热期 (Dynamic Warmup)

不同的参数组合可能需要不同的预热期（例如长均线策略需要更多历史数据）。你可以通过 `warmup_calc` 回调函数动态指定：

```python
def warmup_calc(params):
    # 预热期 = 长期均线窗口 + 1
    return params["long_window"] + 1

run_grid_search(..., warmup_calc=warmup_calc)
```

### 参数约束 (Constraint)

有些参数组合在逻辑上是无效的（例如短期均线 > 长期均线）。通过 `constraint` 回调函数可以提前过滤这些组合，节省计算资源：

```python
def param_constraint(params):
    # 只保留短期窗口 < 长期窗口的组合
    return params["short_window"] < params["long_window"]

run_grid_search(..., constraint=param_constraint)
```

---

## 4. Grid Search vs Walk-Forward 对比

| 特性 | Grid Search (网格搜索) | Walk-Forward (滚动优化) |
| :--- | :--- | :--- |
| **数据使用** | 使用全部数据一次性优化 | 数据滚动切分，训练集/测试集严格分离 |
| **参数结果** | **1 组** 全局静态参数 | **多组** 动态变化的参数 |
| **过拟合风险** | **极高** (看着答案找最优解) | **低** (模拟真实未知环境) |
| **核心目的** | 探索参数敏感性，找“理论上限” | 验证策略稳健性，评估“实战预期” |
| **代码对应** | `akquant.run_grid_search` | `akquant.run_walk_forward` |
