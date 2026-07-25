# Chapter 4: Event-Driven Backtesting Principles

This chapter is currently maintained in Chinese first.

- Chinese chapter: [第 4 章：事件驱动回测原理 (Event-Driven Architecture)](../../zh/textbook/04_backtest_engine.md)
- Fill mode note: `fill_policy=<FillMode>` (one of `CurrentClose()`, `NextOpen()`, `NextClose()`, `NextAverage()`, `NextHighLowMid()`) is now the recommended public style.
- Configuration layering note (highest to lowest): order-level > strategy-map level (`strategy_*`) > run-level > market defaults.
- Commission layering note: `commission_policy` is now the recommended run-level entry for `percent`, `fixed`, and `per_unit`; legacy `commission_rate` remains as a percent-only shorthand.
- T+1 scope note: `t_plus_one` currently remains a run/market-level switch, not a per-strategy layered setting.
- Time ordering note:
  - The event loop orders and advances time in UTC.
  - Structured fields such as `event_time_iso`, `bar.timestamp_iso`, and `trade.timestamp_iso` are intentionally UTC ISO 8601 strings.
  - Local timezone conversion is a display concern, handled by helpers like `self.now`, `self.to_local_time(...)`, and `self.format_time(...)`.
  - In short: UTC keeps the engine correct; local time keeps the output readable.
- Textbook home: [Chinese textbook index](../../zh/textbook/index.md)
- Practice links:
  - Primary example: [examples/textbook/ch04_comparison.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch04_comparison.py)
  - Extended example: [examples/25_streaming_backtest_demo.py](https://github.com/akfamily/akquant/blob/main/examples/25_streaming_backtest_demo.py)
  - Guide: [Data Guide](../guide/data.md)
