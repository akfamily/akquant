# Runtime Config Guide

This page explains how to control strategy runtime behavior from backtest entry points, without editing strategy class code.

## 1. What It Solves

`strategy_runtime_config` lets you inject runtime switches at call-site:

- Error handling mode (`error_mode`)
- Portfolio update threshold (`portfolio_update_eps`)
- Precise day-boundary hooks (`enable_precise_day_boundary_hooks`)
- Legacy fallback flag (`re_raise_on_error`)

It works for both:

- `run_backtest(...)`
- `run_from_checkpoint(...)`

## 2. Basic Usage

```python
from akquant import StrategyRuntimeConfig, run_backtest

result = run_backtest(
    data=data,
    strategy=MyStrategy,
    strategy_runtime_config=StrategyRuntimeConfig(
        error_mode="continue",
        portfolio_update_eps=1.0,
    ),
)
```

You can also pass a `dict`:

```python
result = run_backtest(
    data=data,
    strategy=MyStrategy,
    strategy_runtime_config={"error_mode": "continue"},
)
```

## 3. Parameter Matrix

| Field | Type | Default | Typical Use | Invalid Input Behavior |
|---|---|---|---|---|
| `error_mode` | `"raise" \| "continue" \| "legacy"` | `"raise"` | Control callback exception policy | Raises `ValueError` |
| `portfolio_update_eps` | `float` (`>= 0`) | `0.0` | Skip tiny equity/cash update noise | Raises `ValueError` |
| `enable_precise_day_boundary_hooks` | `bool` | `False` | Enable strict before/after trading by boundary timers | Coerced by bool conversion |
| `re_raise_on_error` | `bool` | `True` | Legacy fallback when `error_mode="legacy"` | Coerced by bool conversion |

## 4. Conflict Priority

When strategy-side config and external config conflict, behavior is controlled by `runtime_config_override`:

- `runtime_config_override=True` (default): apply external config
- `runtime_config_override=False`: keep strategy-side config

Identical conflict warnings are deduplicated per strategy instance.

## 5. Common Pitfalls

- Passing unknown keys in `strategy_runtime_config` fails fast with field-level error.
- Passing negative `portfolio_update_eps` fails fast with validation error.
- Repeated runs on the same strategy instance may not repeat identical warnings due to deduplication.
- `runtime_config_override=False` preserves strategy-side config even if external values are provided.

## 6. Warm Start Injection

You can override runtime behavior when resuming from snapshot:

```python
result = run_from_checkpoint(
    checkpoint_path="snapshot.pkl",
    data=new_data,
    symbols="TEST",
    strategy_runtime_config={"error_mode": "continue"},
)
```

Conflict rules are exactly the same as `run_backtest`.

See also: [Warm Start Guide](warm_start.md).

## 7. End-to-End Demo

See runnable demo:

- [22_strategy_runtime_config_demo.py](https://github.com/akfamily/akquant/blob/main/examples/22_strategy_runtime_config_demo.py)
- [44_strategy_source_loader_demo.py](https://github.com/akfamily/akquant/blob/main/examples/44_strategy_source_loader_demo.py)

Expected output markers:

- `scenario1_done`
- `scenario2_exception=...`
- `scenario3_done`

## 8. Troubleshooting Cheat Sheet

| Symptom / Error | Likely Cause | Fast Fix |
|---|---|---|
| `strategy_runtime_config contains unknown fields: ...` | Unknown key in injected dict | Remove unsupported keys and keep only documented fields |
| `invalid strategy_runtime_config: portfolio_update_eps must be >= 0` | Negative `portfolio_update_eps` | Set `portfolio_update_eps` to `0` or a positive float |
| Runtime config passed but strategy behavior unchanged | `runtime_config_override=False` is enabled | Set `runtime_config_override=True` or remove the flag |
| Conflict warning appears only once | Warning dedup is per strategy instance and conflict payload | This is expected; create a new strategy instance if you need repeated warning output |
| Warm-start resume still raises callback exceptions | Restored strategy config remains active and override not applied | Pass `strategy_runtime_config={"error_mode": "continue"}` with `runtime_config_override=True` |

## 9. Dynamic Strategy Loading (`strategy_source` / `strategy_loader`)

`run_backtest(...)` supports loading strategy implementation dynamically at call time:

- `strategy_source`: strategy input, supports file path (`str` / `PathLike`) or `bytes`
- `strategy_loader`: loader name, defaults to `python_plain`
- `strategy_loader_options`: loader options dict

Built-in loaders:

- `python_plain`: load strategy from a local Python source file
- `encrypted_external`: delegate decryption and loading to external callback

### 9.1 `python_plain` example

```python
result = run_backtest(
    data=data,
    strategy=None,
    strategy_source="my_strategy.py",
    strategy_loader="python_plain",
    strategy_loader_options={"strategy_attr": "MyStrategy"},
)
```

### 9.2 `encrypted_external` example

```python
def decrypt_and_load(source, options):
    ...
    return MyStrategy

result = run_backtest(
    data=data,
    strategy=None,
    strategy_source=b"...encrypted-bytes...",
    strategy_loader="encrypted_external",
    strategy_loader_options={"decrypt_and_load": decrypt_and_load},
)
```

### 9.3 Relation to `run_from_checkpoint`

`run_from_checkpoint(...)` currently restores strategy instance from checkpoint and does not
reload strategy implementation via `strategy_source` / `strategy_loader`.

## 10. broker_profile Selection Guide

`run_backtest(..., broker_profile=...)` injects a preset of fee/slippage/lot defaults, useful when you want to align execution assumptions quickly before finalizing broker-specific params.

Priority rule:

- Explicit arguments override `broker_profile` template values
- Template values override system defaults

| Template | Recommended Scenario | Main Characteristics | Typical Risk |
|---|---|---|---|
| `cn_stock_miniqmt` | General A-share simulation, baseline MiniQMT alignment | Default commission + stamp duty + transfer fee + minimum commission + 100-share lot | Can be optimistic under extreme impact conditions |
| `cn_stock_t1_low_fee` | Low-fee account sensitivity tests, net-return stress checks | Lower commission/transfer fee and lower minimum commission | Can overestimate high-turnover strategy performance |
| `cn_stock_sim_high_slippage` | Intraday impact/liquidity stress scenarios, robustness replay | Higher slippage and more conservative turnover assumptions | Can underestimate low-impact strategy performance |

Template parameter details (current built-in values):

| Template | commission_rate | stamp_tax_rate | transfer_fee_rate | min_commission | slippage | volume_limit_pct | lot_size |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cn_stock_miniqmt` | 0.0003 | 0.001 | 0.00001 | 5.0 | 0.0002 | 0.2 | 100 |
| `cn_stock_t1_low_fee` | 0.0002 | 0.001 | 0.000005 | 3.0 | 0.0001 | 0.25 | 100 |
| `cn_stock_sim_high_slippage` | 0.0003 | 0.001 | 0.00001 | 5.0 | 0.001 | 0.1 | 100 |

Quick example:

```python
result = run_backtest(
    data=data,
    strategy=MyStrategy,
    symbols="000001.SZ",
    broker_profile="cn_stock_t1_low_fee",
    show_progress=False,
)
```

If you already have broker-confirmed live parameters, prefer explicit values such as `commission_rate`, `slippage`, and `lot_size` as your final baseline.
