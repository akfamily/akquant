# Chapter 7: Futures Market and Derivatives Strategies

This chapter is currently maintained in Chinese first.

- Chinese chapter: [第 7 章：期货市场与衍生品策略](../../zh/textbook/07_futures.md)
- Session note: the primary example passes `sessions={"rb": "CN_FUTURES_NIGHT_23"}`. Night sessions belong to the next trading day (Friday night belongs to the following Monday) and settlement runs at the 15:00 close. In custom schedules `"prev"` anchors to the previous trading day (not calendar day), so a CME-style Sunday-evening open cannot be expressed; use `ALWAYS_OPEN` for round-the-clock trading.
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
- Trading sessions are no longer configured here (`enforce_sessions` / `session_profile` / `sessions` were removed and raise a migration `TypeError`); use `run_backtest(sessions=...)`, e.g. `sessions={"rb": "CN_FUTURES_NIGHT_23"}`, or `sessions={"default": "ALWAYS_OPEN"}` for no session constraint
  - Keys match by symbol prefix (longest prefix wins; exchange-qualified keys such as `shfe.au` work), `"default"` is the global default; keys must be non-empty and unique after trimming and lower-casing (`"RB"` together with `"rb"` is rejected)
  - Besides a template name, a value can be a custom schedule: a sequence of `(start, end, session[, anchor])`, e.g. `sessions={"xx": [("21:00", "02:30", "continuous"), ("09:00", "11:30", "continuous"), ("13:30", "15:00", "continuous")]}`. `start` / `end` must be strict two-digit `HH:MM` (`"9:30"` is rejected, write `"09:30"`); `session` is one of `continuous` / `opening_auction` / `auction_hold` / `closing_auction` / `break` / `post_close`; anything outside the segments is closed, so never write `closed`
  - Anchoring: without a 4th item, segments with `start >= "18:00"` are the previous trading day's night session and the rest belong to the trading day itself; override with `"prev"` / `"day"`. `"prev"` means the previous trading day, not the previous calendar day; use `ALWAYS_OPEN` for round-the-clock trading. `end <= start` means crossing midnight and is only allowed for night segments; segments must not overlap within one anchor, nor once night segments are laid onto the trading day
  - Without `trading_calendar`, a custom schedule defaults to a weekday calendar (holidays are not known; pass a trading-day list when it matters)
  - An option not matched by any `sessions` prefix follows the schedule of its `underlying_symbol` when that is a registered future (a night template for `rb` also applies to RB options); otherwise it uses `CN_OPTION`

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
