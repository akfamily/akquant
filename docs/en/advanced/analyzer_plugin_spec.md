# Analyzer Plugins

An analyzer plugin listens to bars and fills during a backtest and produces a custom analysis at the end, stored in `result.analyzer_outputs`. Use it for statistics the built-in metrics don't cover — fills by time of day, how often a condition occurred — without touching strategy code.

## Minimal example

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

`analyzer_plugins` must be a list or tuple. Pass fresh plugin instances to each backtest, since plugins accumulate state.

## Plugin protocol

```python
class AnalyzerPlugin(Protocol):
    name: str
    def on_start(self, context: dict[str, Any]) -> None: ...
    def on_bar(self, context: dict[str, Any]) -> None: ...
    def on_trade(self, context: dict[str, Any]) -> None: ...
    def on_finish(self, context: dict[str, Any]) -> dict[str, Any]: ...
```

- All four methods and `name` are required; otherwise `run_backtest` raises `TypeError` before starting.
- The return value of `on_finish` is stored in `result.analyzer_outputs` under `name`; a later plugin with the same name overwrites an earlier one.
- `akquant.analyzer_plugin.AnalyzerTemplate` is a ready-to-copy template (it counts fills as `seen_trades`).

## Callback timing and context

| Callback | When | `context` keys |
|---|---|---|
| `on_start` | Once, before the data loop | `engine`, `strategy`, `strategies`, `slot_strategy_map`, `symbols` |
| `on_bar` | After the strategy callback for each bar (not during warmup, see below) | `strategy`, `bar`, `engine`, `ctx`, `owner_strategy_id` |
| `on_trade` | For each fill | `strategy`, `trade`, `engine`, `ctx`, `owner_strategy_id` |
| `on_finish` | Once, after the backtest result is built | `engine`, `strategy`, `strategies`, `slot_strategy_map`, `result` |

- `strategy` is the main strategy instance; with multiple strategies, `strategies` holds all instances, `slot_strategy_map` maps strategy id to instance, and `owner_strategy_id` is the id the callback belongs to.
- The `result` passed to `on_finish` is the `BacktestResult` about to be returned; you can read its trades, equity curve and so on.
- With multiple strategies, all strategies share the same plugins: each bar and each fill triggers a callback once per owning strategy; use `owner_strategy_id` to tell them apart.
- Only bars trigger `on_bar`; ticks do not.

## When a plugin fails

Plugins are isolated from each other: if one plugin's callback raises, only that call is skipped; other plugins still receive the same callback and the backtest continues.

- The first error per plugin and callback is logged at error level with a traceback; later errors of the same kind are only counted, and one summary line ("N more exceptions suppressed") is logged when the backtest ends.
- A plugin whose `on_finish` fails is left out of `result.analyzer_outputs`; other plugins' outputs are kept.

## Known limitations

- **Backtest only**: `run_live` does not accept `analyzer_plugins`.
- **No automatic discovery**: plugins must be passed in `analyzer_plugins` explicitly; registration through package entry points is not supported.
- **During warmup `on_bar` does not reach analyzers, but `on_trade` does.** `Strategy.warmup_period` is counted per symbol; while a symbol has fewer bars than `warmup_period`, the engine returns before calling `on_bar`, whereas `on_trade` goes through the separate order-event path and is not gated. Plugins that use the number of `on_bar` calls as a denominator or index will therefore under-count or misalign with multiple symbols and warmup. Count per symbol instead (accumulate by `context["bar"].symbol` in `on_bar` and `context["trade"].symbol` in `on_trade`), and don't rely on the two callbacks staying in step.
