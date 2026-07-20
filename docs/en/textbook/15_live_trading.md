# Chapter 15: Live Trading Systems and Operations

This chapter is currently maintained in Chinese first.

- Chinese chapter: [第 15 章：实盘交易系统与运维](../../zh/textbook/15_live_trading.md)
- Textbook home: [Chinese textbook index](../../zh/textbook/index.md)
- Live execution semantics note:
  - CTP supports `execution_semantics_mode` with `strict` (default) and `compatible`.
  - In `strict`, terminal order states are confirmed by `OnRtnOrder` callbacks.
- Practice links:
  - Primary example: [examples/textbook/ch15_live_trading.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch15_live_trading.py)
  - Extended example: [examples/textbook/ch15_strategy_loader.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch15_strategy_loader.py)
  - Supplementary example: [examples/44_strategy_source_loader_demo.py](https://github.com/akfamily/akquant/blob/main/examples/44_strategy_source_loader_demo.py)
  - Guide: [Live Functional Quickstart Guide](../advanced/live_functional_quickstart.md)

## Operational Logging Note

Before live or paper troubleshooting, explicitly configure logging instead of relying on defaults:

```python
import akquant

akquant.configure_logging(
    akquant.LogConfig(
        profile="live",
        level="INFO",
        console=True,
        filename="logs/live.log",
        file_level="DEBUG",
        file_json=True,
        file_max_bytes=50_000_000,
        file_backup_count=5,
    )
)
```

This keeps strategy-side `on_order` / `on_trade` logs and gateway/execution warnings in the same pipeline. It is especially useful when tracing rejects, unknown cancel requests, session-close expiry, or strict-semantics cases where terminal state is not confirmed until broker callbacks arrive.

**Order auditing (an offline-reconstructable record).** Under `broker_live`, every order submit / update / fill / cancel / reject is emitted as a structured INFO record under the `akquant.audit.order` namespace. Set `order_audit_file` to also persist a dedicated audit JSON stream — **the full order lifecycle can be reconstructed from that file alone, even after the process stops**:

```python
akquant.configure_logging(
    akquant.LogConfig(
        profile="live",
        console=True,
        console_level="WARNING",          # console only keeps rejects/disconnects that need a human
        filename="logs/live.log",
        file_level="INFO",
        order_audit_file="logs/orders_audit.log",  # per-order audit JSON stream
        order_audit_level="INFO",
    )
)
```

For a high-frequency live strategy this split is strongly recommended: a clean `WARNING` console, with the per-order INFO audit persisted separately.

**Sensitive masking (on by default).** Logs mask credential-class fields (`password`/`token`/`api_key`, …) fully and account-class fields (`user_id`/`account`, …) keeping the last 4 chars, at the handler layer — so even a newly added log statement cannot leak secrets in cleartext. Disable with `mask_sensitive=False`.

**Log language.** Messages are english by default (a searchable, collaboration-friendly contract), and structured fields (`event`/`side`/`price`) are always english. If you prefer a Chinese console, set `language="zh"`: it only re-renders the **console order-audit line**, while **files and JSON audit streams stay english**, so grep/reconciliation/alerting never fork by language. Reserve `CRITICAL` for system-level fatal events (trader front disconnect, runner crash) and route it to a dedicated alert channel.
