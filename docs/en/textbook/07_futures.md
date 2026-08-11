# Chapter 7: Futures Market and Derivatives Strategies

This chapter is currently maintained in Chinese first.

- Chinese chapter: [第 7 章：期货市场与衍生品策略](../../zh/textbook/07_futures.md)
- Textbook home: [Chinese textbook index](../../zh/textbook/index.md)
- Practice links:
  - Primary example: [examples/textbook/ch07_futures.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch07_futures.py)
  - Extended example: [examples/04_mixed_assets.py](https://github.com/akfamily/akquant/blob/main/examples/04_mixed_assets.py)
  - Expiry callback example: [examples/49_on_expiry_demo.py](https://github.com/akfamily/akquant/blob/main/examples/49_on_expiry_demo.py)
  - Functional twin example: [examples/textbook/ch07_futures_functional.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch07_futures_functional.py)
  - Guide: [Strategy Guide](../guide/strategy.md)

## China futures config highlights

`BacktestConfig.china_futures` supports prefix-level futures templates and validation:

- Prefix template fields: `multiplier`, `margin_ratio`, `tick_size`, `lot_size`
- Prefix commission override: `commission_rate`
- Prefix validation switches: `enforce_tick_size`, `enforce_lot_size`
- Session behavior switch: `enforce_sessions`

The config now validates early during object construction (fail-fast):

- Empty `symbol_prefix` is rejected
- Invalid numeric ranges (e.g. non-positive `multiplier`) are rejected
- Duplicate prefixes in the same list are rejected with index-aware messages

For complete examples and explanations, refer to the Chinese chapter section
“AKQuant 中国期货配置速览”.

### Priority matrix

| Scenario | Highest priority | Secondary priority | Fallback |
|---|---|---|---|
| Contract fields (`multiplier`, `margin_ratio`, `tick_size`, `lot_size`) | Explicit `InstrumentConfig` fields | `instrument_templates_by_symbol_prefix` | `run_backtest` defaults |
| Prefix commission | `fee_by_symbol_prefix` | Template `commission_rate` | `StrategyConfig.commission_rate` |
| Prefix validation switches | `validation_by_symbol_prefix` | Template `enforce_tick_size` / `enforce_lot_size` | Global `ChinaFuturesConfig.enforce_*` |
| Market selection | `use_china_futures_market=False` or mixed-asset fallback | `use_china_futures_market=True` with futures-only set | `use_simple_market` |

Notes:

- Within the same layer, explicit prefix rules override template defaults.
- In order validation matching, a more specific prefix (longer match) wins.

Low-level Engine API naming:

- Futures fee APIs are standardized as `set_futures_fee_rules` and
  `set_futures_fee_rules_by_prefix`.
- Legacy singular names `set_future_fee_rules*` are removed.

## Futures Margin Account Snapshot Semantics

Under AKQuant's futures margin-account semantics, read `get_account()` and
`equity` like this:

- `cash`: cash balance. Opening a futures position does not deduct full notional
  the way a spot buy does; cash mainly reflects fees and realized cash flows.
- `equity`: account equity. This is the `equity` property.
- `used_margin` / `margin`: margin currently in use.
- `free_margin`: available margin (`equity - used_margin`), i.e. the amount
  actually usable to open new positions. When an order is rejected for
  insufficient margin, the `Available` value in the log refers to this field —
  do not compare it against `cash`.
- `notional_value`: current futures notional exposure.
- `unrealized_pnl`: floating PnL marked with the latest price.
- `market_value`: mainly a spot-style marked-value field and should not be read
  as "futures notional".

For futures strategies, the safer mental model is:

1. use `equity` for total account value,
2. use `used_margin` for margin usage,
3. use `notional_value` for leverage exposure.
