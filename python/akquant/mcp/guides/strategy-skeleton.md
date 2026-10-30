# 类式策略骨架

MCP 一期只支持类式策略：在一个 `.py` 文件里定义一个继承 `akquant.Strategy` 的类，再用 `check_strategy` 和 `run_backtest` 检查、回测。

## 四条规则

1. **不要重写 `__init__`。** 参数在类体里用 `IntParam`、`FloatParam`、`BoolParam`、`ChoiceParam`、`ListParam` 声明，读取时用 `self.params.<名>`。`run_backtest` 的 `strategy_params` 只能传这里声明过的参数；重写了 `__init__` 又不接收 `**kwargs` 时，传参会报 `unexpected keyword argument`。
2. **运行状态在 `on_start` 里初始化。** 这时 `self.params` 已经就绪。指标也在 `on_start` 里用 `self.I(...)` 声明（见 `get_guide("indicators")`）。
3. **核心逻辑写在 `on_bar(self, bar)` 里。** 每根 K 线闭合时调用一次；有多个标的时，每个标的各调用一次，用 `bar.symbol` 区分。
4. **一个文件只放一个 `Strategy` 子类。** 有多个时，`check_strategy` 和 `run_backtest` 都要传 `strategy_attr` 指定用哪个。

## 回调顺序

`on_start` → 每根 bar：先触发有回报的 `on_order` / `on_trade`，再触发 `on_bar` → 结束时触发 `on_stop`。完整的回调清单可以用 `search_api("on_")` 查。

## 骨架

```python
from akquant import Bar, IntParam, Strategy


class MomentumStrategy(Strategy):
    """N 日动量: 收盘价高于 N 日前就持有, 否则空仓."""

    lookback = IntParam(20, ge=2, title="回看天数")

    def on_start(self) -> None:
        self.closes: dict[str, list[float]] = {}

    def on_bar(self, bar: Bar) -> None:
        history = self.closes.setdefault(bar.symbol, [])
        history.append(bar.close)
        n = self.params.lookback
        if len(history) <= n:
            return
        position = self.get_position(bar.symbol)
        if bar.close > history[-n - 1] and position == 0:
            self.order_target_percent(target_percent=0.9, symbol=bar.symbol)
        elif bar.close < history[-n - 1] and position > 0:
            self.order_target_percent(target_percent=0.0, symbol=bar.symbol)
```

外部改参数时这样传：`run_backtest(..., strategy_params={"lookback": 10})`。
