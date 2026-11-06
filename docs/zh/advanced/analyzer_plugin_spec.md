# Analyzer 插件

Analyzer 插件在回测过程中旁听 bar 和成交，结束时产出一份自定义分析结果，放进 `result.analyzer_outputs`。适合做内置指标没有覆盖的统计，例如按时段统计成交、记录特定条件出现的次数，而不必改策略代码。

## 最小例子

```python
import akquant as aq

class TradeCounter:
    name = "trade_counter"

    def __init__(self):
        self.bars = 0
        self.trades = 0

    def on_start(self, context):
        pass

    def on_bar(self, context):
        self.bars += 1

    def on_trade(self, context):
        self.trades += 1

    def on_finish(self, context):
        return {"bars": self.bars, "trades": self.trades}

result = aq.run_backtest(
    data=df,
    strategy=MyStrategy,
    symbols="600000",
    analyzer_plugins=[TradeCounter()],
)
print(result.analyzer_outputs)   # {"trade_counter": {"bars": ..., "trades": ...}}
```

`analyzer_plugins` 必须是列表或元组。每次回测请传入新的插件实例，插件对象本身会累积状态。

## 插件协议

```python
class AnalyzerPlugin(Protocol):
    name: str
    def on_start(self, context: dict[str, Any]) -> None: ...
    def on_bar(self, context: dict[str, Any]) -> None: ...
    def on_trade(self, context: dict[str, Any]) -> None: ...
    def on_finish(self, context: dict[str, Any]) -> dict[str, Any]: ...
```

- 四个方法都必须实现，`name` 必须存在，否则 `run_backtest` 在开始前抛 `TypeError`。
- `on_finish` 的返回值以 `name` 为键收进 `result.analyzer_outputs`；同名插件后者覆盖前者。
- `akquant.analyzer_plugin.AnalyzerTemplate` 是一个可以直接复制改写的模板（统计成交笔数 `seen_trades`）。

## 回调时机与上下文

| 回调 | 时机 | `context` 的键 |
|---|---|---|
| `on_start` | 数据循环开始前，调用一次 | `engine`、`strategy`、`strategies`、`slot_strategy_map`、`symbols` |
| `on_bar` | 每根 bar 的策略回调之后（预热期内不调用，见下） | `strategy`、`bar`、`engine`、`ctx`、`owner_strategy_id` |
| `on_trade` | 每笔成交回报 | `strategy`、`trade`、`engine`、`ctx`、`owner_strategy_id` |
| `on_finish` | 回测结果生成之后，调用一次 | `engine`、`strategy`、`strategies`、`slot_strategy_map`、`result` |

- `strategy` 是主策略实例；多策略时 `strategies` 是全部实例，`slot_strategy_map` 是策略 id 到实例的映射，`owner_strategy_id` 是当前回调所属的策略 id。
- `on_finish` 拿到的 `result` 就是即将返回的 `BacktestResult`，可以读取它的成交、权益曲线等数据。
- 多策略时所有策略共用同一组插件：每根 bar、每笔成交按所属策略各触发一次，用 `owner_strategy_id` 区分。
- 只有 bar 会触发 `on_bar`，tick 不会。

## 插件出错时

每个插件单独隔离：某个插件的回调抛异常，只跳过它的这一次回调，其他插件照常收到同一次回调，回测也不中断。

- 每个插件的每个回调只记第一次错误（error 级别，带 traceback），之后的同类错误只计数，回测结束时汇总一条“另有 N 次异常已抑制”。
- `on_finish` 出错的插件不出现在 `result.analyzer_outputs` 里，其他插件的输出照常保留。

## 已知限制

- **只用于回测**：`run_live` 不接收 `analyzer_plugins`。
- **没有插件自动发现**：插件需要显式传进 `analyzer_plugins`，不支持通过包的 entry points 自动注册。
- **预热期内 `on_bar` 不到达 analyzer，但同期 `on_trade` 会**。`Strategy.warmup_period` 按标的独立计数，某标的累计 bar 数不足 `warmup_period` 时，引擎会在调用 `on_bar` 之前返回；而 `on_trade` 走独立的订单事件路径，不受这个门槛约束。因此按“收到的 `on_bar` 次数”做分母或索引的插件，在多标的加预热的场景下计数会偏小或错位。请按标的分别计数（在 `on_bar` 与 `on_trade` 里分别按 `context["bar"].symbol` / `context["trade"].symbol` 累加），不要依赖两种回调的先后与次数对应。
