# Tick Size Alignment

Order prices must be an integer multiple of the instrument's tick size. AKQuant **validates and rejects**
misaligned prices; it never rounds them for you. The full rationale (including the QuantConnect Lean and
CCXT incidents behind this choice) is in the [Chinese version](../../zh/advanced/tick_size_alignment.md).

## Rounding explicitly

Use `Strategy.round_to_tick(symbol, price, direction)`:

| `direction` | Rounding | Use for |
|---|---|---|
| `"down"` | floor | buy orders (never pay more than intended) |
| `"up"` | ceiling | sell orders (never receive less than intended) |
| `"nearest"` (default) | half-up | display / logging only |

It raises `KeyError` for an unregistered symbol instead of guessing a tick. Misaligned prices are rejected
locally before submission in live trading and by the matcher in backtests.

## Default tick sizes

| Instrument | Tick |
|---|---|
| A-share stocks | 0.01 |
| ETFs / funds | 0.001 |
| Bonds incl. convertible bonds (SSE and SZSE) | 0.001 |
| SSE/SZSE ETF options | 0.0001 |

Without an explicit `tick_size`, `InstrumentConfig` defaults to `0.001` for `FUND` and `0.01` for everything
else. Two product presets carry the right tick (explicitly passed fields still win):

| `asset_type` | Expands to | Tick | Other defaults |
|---|---|---|---|
| `"CONVERTIBLE_BOND"` | FUND | 0.001 | lot of 10, T+0 |
| `"ETF_OPTION"` | OPTION | 0.0001 | multiplier 10000, T+0, China single-leg margin |

Pitfalls: the generic option default of 0.01 is wrong for ETF options — use `"ETF_OPTION"` or pass
`tick_size=0.0001`. ETFs or convertible bonds configured as `"STOCK"` get 0.01; use `"FUND"` /
`"CONVERTIBLE_BOND"` instead.

## Convertible bonds

```python
from akquant import InstrumentConfig

InstrumentConfig(symbol="113050.SH", asset_type="CONVERTIBLE_BOND")
```

This models trading and fees only (T+0, lot of 10, no stamp duty). Conversion, put, call and first-day
suspension are not modeled, and AKQuant has no price-limit (limit up/down) mechanism.

## Disabling validation

`ChinaStockConfig(enforce_tick_size=False)` for stocks/funds and
`ChinaFuturesValidationConfig(symbol_prefix=..., enforce_tick_size=False)` for futures. Not recommended:
the broker or exchange will reject the order later, where it is more expensive to diagnose.
