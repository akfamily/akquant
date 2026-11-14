# Custom Broker Registry

This page explains how to register and use a custom broker in `akquant` without editing built-in factory branches.

## 1. When to Use

- Integrate an internal broker API or third-party trading system
- Implement dedicated Market/Trader gateways per broker
- Inject mock broker implementations for integration tests

## 2. Core APIs

Import these APIs from `akquant.gateway`:

- `register_broker(name, builder)`
- `unregister_broker(name)`
- `get_broker_builder(name)`
- `list_registered_brokers()`
- `create_gateway_bundle(...)`

Registered brokers are resolved first by `create_gateway_bundle`; built-in `ctp` is used as fallback.

## 3. Builder Signature

```python
def builder(
    feed: DataFeed,
    symbols: Sequence[str],
    use_aggregator: bool,
    **kwargs: Any,
) -> GatewayBundle:
    ...
```

`GatewayBundle` should include:

- `market_gateway` (required)
- `trader_gateway` (optional)
- `metadata` (optional)

## 4. Minimal Integration Flow

```python
from akquant import DataFeed
from akquant.gateway import create_gateway_bundle, register_broker

register_broker("demo", demo_builder)

bundle = create_gateway_bundle(
    broker="demo",
    feed=DataFeed(),
    symbols=["000001.SZ"],
    label="demo",
)

print(bundle.metadata)
```

See runnable example:

- [35_custom_broker_registry_demo.py](https://github.com/akfamily/akquant/blob/main/examples/35_custom_broker_registry_demo.py)

## 5. Recommended Constraints

- Keep `name` lowercase to avoid accidental key collisions
- Validate builder input and fail fast
- Keep `trader_gateway.place_order` idempotent for `client_order_id`
- Align callbacks to unified models: `UnifiedOrderSnapshot` / `UnifiedTrade` / `UnifiedExecutionReport`

## 6. Troubleshooting

- `broker must be one of ...`: broker not registered or name mismatch
- Registered but not resolved: check registration happens before `create_gateway_bundle`
- No strategy callbacks: ensure trader gateway implements `on_order/on_trade/on_execution_report`

## 7. Pre-Go-Live Checklist

- [Custom Broker Production Checklist](./custom_broker_production_checklist.md)
