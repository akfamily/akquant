# Analyzer 插件接口草案

## 目标

- 让第三方分析器无需修改内核即可接入回测流程。
- 将固定分析输出扩展为“内置 + 插件”双轨体系。

## 最小生命周期

```python
class AnalyzerPlugin(Protocol):
    name: str
    def on_start(self, context: dict[str, Any]) -> None: ...
    def on_bar(self, context: dict[str, Any]) -> None: ...
    def on_trade(self, context: dict[str, Any]) -> None: ...
    def on_finish(self, context: dict[str, Any]) -> dict[str, Any]: ...
```

实现草案：
- [analyzer_plugin.py](https://github.com/akfamily/akquant/blob/main/python/akquant/analyzer_plugin.py)

## 插件管理

- `AnalyzerManager.register(plugin)`
- `AnalyzerManager.on_start/on_bar/on_trade/on_finish`
- 输出结构：`{plugin_name: plugin_result}`

## 上下文约定（v0）

- `engine`
- `strategy`
- `bar`（在 `on_bar` 时存在）
- `trade`（在 `on_trade` 时存在）
- `result`（在 `on_finish` 时存在）

## 模板插件

- `AnalyzerTemplate`
- 示例输出：`seen_trades`

## 分发建议

- 包发现机制采用 `entry_points`。
- 每个插件包声明：
  - 支持 AKQuant 版本范围
  - 插件名称
  - 入口类

## 验收

- 插件异常隔离策略可配置（继续/中断）。
- 报告输出可附加插件 section。
- 提供至少 2 个官方示例插件。

## 已知限制

- **预热期内 `on_bar` 不到达 analyzer，但同期 `on_trade` 会**。`Strategy.warmup_period`
  按标的独立计数（见 CHANGELOG「修复 `warmup_period` 在多标的下每个标的只预热约
  `N / 标的数` 根的问题」条目），某标的累计 bar 数不足 `warmup_period` 时，引擎在
  `on_bar_event` 里会在调用 `analyzer_manager.on_bar` 之前 `return`——但 `on_trade`
  走的是独立的订单事件路径，不受这个门槛约束，预热期内的成交回报仍会正常派发到
  `analyzer_manager.on_trade`。
  这是既有的结构性不一致（修复前同样存在，只是预热期短、不易暴露；per-symbol 化后
  某个标的的预热期可能覆盖它的全部历史，不一致更容易暴露）。
  **影响**：若自定义 `AnalyzerPlugin` 按"收到的 `on_bar` 调用次数"做分母或索引，在
  多标的 + 预热场景下这个计数会比真实成交序列偏小/错位。请勿假设 bar 计数与成交
  计数同步递增；需要按标的维护自己的计数（例如在 `on_bar` 与 `on_trade` 里分别对
  `context["bar"].symbol` / `context["trade"].symbol` 累加，不要依赖调用次数本身
  的先后关系）。
  修复该不一致需要改动 analyzer 的调度架构（让 `on_trade` 也感知 warmup 门槛，或
  反过来让预热期的 bar 也送达 analyzer），超出本文档当前范围，暂不改行为。
