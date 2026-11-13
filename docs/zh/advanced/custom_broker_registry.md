# 自定义 Broker 注册

本页说明如何在不修改内置工厂分支的前提下，为 `akquant` 注册并使用自定义 broker。

## 1. 适用场景

- 接入内部柜台或第三方交易系统
- 为不同券商实现独立的 Market/Trader 网关
- 在测试环境下注入 mock broker 做联调

## 2. 核心 API

可直接从 `akquant.gateway` 导入以下接口：

- `register_broker(name, builder)`
- `unregister_broker(name)`
- `get_broker_builder(name)`
- `list_registered_brokers()`
- `create_gateway_bundle(...)`

注册后的 broker 会被 `create_gateway_bundle` 优先解析；未命中时才走内置 `ctp`。

## 3. Builder 签名

```python
def builder(
    feed: DataFeed,
    symbols: Sequence[str],
    use_aggregator: bool,
    **kwargs: Any,
) -> GatewayBundle:
    ...
```

其中 `GatewayBundle` 需要返回：

- `market_gateway`（必填）
- `trader_gateway`（可选）
- `metadata`（可选）

## 4. 最小接入流程

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

完整可运行示例见：

- [35_custom_broker_registry_demo.py](https://github.com/akfamily/akquant/blob/main/examples/35_custom_broker_registry_demo.py)

## 5. 建议约束

- `name` 建议小写，避免重复键覆盖
- `builder` 内部应做参数校验并快速失败
- `trader_gateway.place_order` 建议保证 `client_order_id` 幂等
- 回调字段建议对齐统一模型：`UnifiedOrderSnapshot` / `UnifiedTrade` / `UnifiedExecutionReport`

## 6. 故障排查

- 报错 `broker must be one of ...`：未注册或注册名称不一致
- 已注册但未生效：检查是否在调用 `create_gateway_bundle` 前完成注册
- 事件回调无触发：确认 trader gateway 已实现 `on_order/on_trade/on_execution_report`

## 7. 上线前检查

- [自定义 Broker 生产接入清单](./custom_broker_production_checklist.md)
