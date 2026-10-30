# 指标：self.I() 声明式指标

指标统一用 `self.I(...)` 在 `on_start` 里声明一次，框架负责每根 bar 自动推进；取值时 `ind[0]` 是当前值，`ind[1]` 是上一根的值。

## 规则

- **在 `on_start` 里声明**，不要放进 `on_bar`。`check_strategy` 遇到 `on_start`/`__init__` 以外的 `self.I` 调用会报 `i-call-outside-init`。参数化的指标（比如周期来自 `self.params`）只能写在 `on_start` 里，因为 `__init__` 执行时参数还没注入。
- **用 `name=` 命名**。同一个策略里有多个同类指标时，名字不能重复。
- **取值**：`ind[0]` 是当前 bar 的值（`ind.value` 是它的别名），`ind[n]` 是前 n 根的值。预热期或越界时返回 `None`，不抛异常，所以判空统一写 `if value is None`。
- **内置指标**：`EMA`、`SMA`、`RSI`、`MACD`、`ATR`、`BollingerBands` 等 100 多个，都是 Rust 实现的增量指标。用 `search_api("EMA")` 查构造参数。
- **自定义向量化指标**：`self.I(PrecomputedIndicator("名", lambda df: ...), name="名")`，函数接收整段行情 DataFrame。
- 回溯深度由 `lookback` 决定，默认 128。

## 示例：双均线交叉

```python
from akquant import EMA, Bar, IntParam, Strategy


class EmaCrossStrategy(Strategy):
    """快线上穿慢线买入, 下穿卖出."""

    fast = IntParam(10, ge=2)
    slow = IntParam(30, ge=3)

    def on_start(self) -> None:
        self.ema_fast = self.I(EMA(self.params.fast), name="ema_fast")
        self.ema_slow = self.I(EMA(self.params.slow), name="ema_slow")

    def on_bar(self, bar: Bar) -> None:
        values = (
            self.ema_fast[0],
            self.ema_slow[0],
            self.ema_fast[1],
            self.ema_slow[1],
        )
        if any(v is None for v in values):
            return  # 预热期
        fast_now, slow_now, fast_prev, slow_prev = values
        position = self.get_position(bar.symbol)
        if fast_prev <= slow_prev and fast_now > slow_now and position == 0:
            self.order_target_percent(target_percent=0.9, symbol=bar.symbol)
        elif fast_prev >= slow_prev and fast_now < slow_now and position > 0:
            self.order_target_percent(target_percent=0.0, symbol=bar.symbol)
```
