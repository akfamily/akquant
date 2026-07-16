# 如何新增一个 Broker

本页说明 akquant broker 网关层的插件契约：新增一个 broker 需要实现哪些接口、
声明哪些能力、以及如何把下单请求路由到具体柜台。目标是让接入者不修改
`akquant.gateway` 核心代码即可插入一个新的交易通道。

如果只是想在不改内置工厂分支的前提下注册一个 broker，先看
[自定义 Broker 注册](custom_broker_registry.md)；本页补充的是"网关本身怎么写"。

## 总览：五步契约

1. 写 **builder** 函数，通过 `register_broker(name, builder)` 注册（或者内置
   broker 直接放进 `python/akquant/gateway/brokers/builtins.py`，由
   `register_builtin_brokers()` 统一注册）。
2. 声明 **`BrokerCapability`**：`broker_extra_fields` 列出本 broker 允许的
   订单专属字段，`features` 声明任意能力标志。
3. 继承 **`TraderGatewayBase`** 实现必需方法。
4. 在 `place_order` 里按 `req.asset_type` 路由品种、从 `req.extra` 取专属
   字段，并维护 broker 订单号与 `client_order_id` 的映射。
5. 纯交易 broker（不接行情）令 `GatewayBundle(market_gateway=None, ...)`，
   行情继续走 akquant 现有 `DataFeed`。

## 1. Builder：注册入口

Builder 是一个可调用对象，签名固定为：

```python
def builder(
    feed: DataFeed,
    symbols: Sequence[str],
    use_aggregator: bool,
    **kwargs: Any,
) -> GatewayBundle:
    ...
```

- 第三方/内部 broker：调用 `register_broker(name, builder)`（来自
  `akquant.gateway`），注册后 `create_gateway_bundle(broker=name, ...)` 会
  优先解析到它。
- 想合并进内置分支的 broker：把 builder 函数加进
  `python/akquant/gateway/brokers/builtins.py`，并在
  `register_builtin_brokers()` 里追加一行 `register_broker("xxx", _build_xxx)`。

Builder 内部通常做三件事：校验必填 `kwargs`、构造 `TraderGateway`（以及可选的
`MarketGateway`）、把 `trader_gateway.get_capabilities()` 的结果写入
`GatewayBundle.trader_capabilities`，供上层校验 `extra` 字段用。

## 2. 声明 `BrokerCapability`

`BrokerCapability`（`from akquant.gateway.broker_models import BrokerCapability`）
是一个 frozen dataclass，描述这个 broker 的执行语义边界：

- `broker_extra_fields: tuple[str, ...]`：策略下单时通过
  `submit_order(..., extra={...})` 传入的柜台专属字段，必须在这里声明；
  未声明的 key 会在校验时被拒绝（`validate_broker_extra` 会抛
  `RuntimeError`，列出未声明字段与已声明集合）。
- `features: frozenset[str]`：任意能力标志的开放集合，用于策略侧按需探测
  "这个 broker 支不支持某个特性"，不做强类型约束。
- 其余字段（`position_effect`、`reduce_only`、`supports_short_sell`、
  `supported_position_effects` 等）描述开平仓/做空等语义是否可用，按 broker
  实际能力如实填写即可，不确定的保持默认值（保守）。

`TraderGateway` 协议要求实现 `get_capabilities() -> BrokerCapability`，
通常返回一个模块级的 `default_capability()` 单例：

```python
def default_capability() -> BrokerCapability:
    return BrokerCapability(
        broker_name="mybroker",
        broker_extra_fields=("account_id", "order_style"),
        features=frozenset({"supports_stop_limit"}),
    )
```

## 3. 继承 `TraderGatewayBase`

`from akquant.gateway.trader_base import TraderGatewayBase` 提供了所有
broker 共享的管件：回调注册、id 反查表、以及默认的
`heartbeat`/`sync_open_orders`/`sync_today_trades` 实现。子类只需要实现
`TraderGateway` 协议中还缺的部分：

必须实现：

- `connect()` / `disconnect()` / `start()` —— 生命周期
- `place_order(req) -> str` / `cancel_order(broker_order_id)`
- `query_order(broker_order_id)` / `query_trades(since=None)`
- `query_account()` / `query_positions()`
- `get_capabilities() -> BrokerCapability`

基类已提供、通常不需要重写：

- 回调注册：`on_order` / `on_trade` / `on_execution_report`
- id 反查：`record_broker_order(broker_order_id, client_order_id)` /
  `client_order_id_for(broker_order_id)`
- 分发：`_emit_order` / `_emit_trade` / `_emit_exec_from_order`
- 默认实现：`heartbeat()`（恒真）、`sync_open_orders()` /
  `sync_today_trades()`（空列表）—— 如果 broker 支持断线补齐，覆盖这两个
  方法即可。

## 4. `place_order` 里的路由与 id 映射

`UnifiedOrderRequest` 里的 `asset_type` 用来在 `place_order` 内部路由到不同
品种的下单通道（例如证券 vs 期货）；`extra: dict[str, Any]` 携带的是
`BrokerCapability.broker_extra_fields` 里声明过的柜台专属字段，直接从
`req.extra` 取值即可（上层已按声明集合校验过，未声明的 key 不会出现）。

下单成功后，用 `self.record_broker_order(broker_order_id, req.client_order_id)`
记录柜台委托号到策略 `client_order_id` 的映射；收到委托/成交回报时，用
`self.client_order_id_for(broker_order_id)` 反查回 `client_order_id`，再拼成
统一模型对象经 `self._emit_order(...)` / `self._emit_trade(...)` /
`self._emit_exec_from_order(...)` 分发给策略层回调。

## 5. 纯交易 broker：行情走现有 feed

如果新 broker 只做交易、不提供行情（多数国内前置机/柜台都是这种情况），
builder 返回的 `GatewayBundle` 里 `market_gateway=None` 即可，行情继续由
akquant 现有的 `DataFeed` 提供：

```python
return GatewayBundle(
    market_gateway=None,
    trader_gateway=trader_gateway,
    trader_capabilities=trader_gateway.get_capabilities(),
    metadata={"broker": "mybroker"},
)
```

## 最小骨架

```python
from __future__ import annotations

from typing import Any, Sequence

from akquant import DataFeed
from akquant.gateway import register_broker
from akquant.gateway.broker_models import (
    BrokerCapability,
    UnifiedAccount,
    UnifiedOrderRequest,
    UnifiedOrderSnapshot,
    UnifiedPosition,
    UnifiedTrade,
)
from akquant.gateway.protocols import GatewayBundle
from akquant.gateway.trader_base import TraderGatewayBase


def default_capability() -> BrokerCapability:
    return BrokerCapability(
        broker_name="mybroker",
        broker_extra_fields=("account_id",),
        features=frozenset(),
    )


class MyTraderGateway(TraderGatewayBase):
    """最小可运行的 TraderGateway 骨架。"""

    def __init__(self, capability: BrokerCapability | None = None) -> None:
        super().__init__()
        self._capability = capability or default_capability()

    def connect(self) -> None:
        ...  # 登录/建立会话

    def disconnect(self) -> None:
        ...  # 释放连接

    def start(self) -> None:
        ...  # 建立推送长连、开始分发回报

    def get_capabilities(self) -> BrokerCapability:
        return self._capability

    def place_order(self, req: UnifiedOrderRequest) -> str:
        # 按 req.asset_type 路由品种；req.extra 取柜台专属字段。
        account_id = req.extra.get("account_id")
        broker_order_id = self._send_order_to_broker(req, account_id)
        if broker_order_id:
            self.record_broker_order(broker_order_id, req.client_order_id)
        return broker_order_id

    def cancel_order(self, broker_order_id: str) -> None:
        ...  # 调用柜台撤单接口

    def query_order(self, broker_order_id: str) -> UnifiedOrderSnapshot | None:
        ...  # 查询单笔委托并转换为 UnifiedOrderSnapshot

    def query_trades(self, since: int | None = None) -> list[UnifiedTrade]:
        ...  # 查询成交并转换为 UnifiedTrade 列表

    def query_account(self) -> UnifiedAccount | None:
        ...  # 查询资金账户

    def query_positions(self) -> list[UnifiedPosition]:
        ...  # 查询持仓

    def _on_broker_push(self, event: str, data: dict[str, Any]) -> None:
        # 收到推送后：反查 client_order_id，再分发给策略层回调。
        broker_order_id = str(data.get("order_id", ""))
        client_order_id = self.client_order_id_for(broker_order_id)
        if event == "order_update":
            snapshot = self._parse_order(data, client_order_id)
            self._emit_order(snapshot)
            self._emit_exec_from_order(snapshot)
        elif event == "trade_update":
            self._emit_trade(self._parse_trade(data, client_order_id))

    def _send_order_to_broker(
        self, req: UnifiedOrderRequest, account_id: Any
    ) -> str:
        raise NotImplementedError

    def _parse_order(
        self, data: dict[str, Any], client_order_id: str
    ) -> UnifiedOrderSnapshot:
        raise NotImplementedError

    def _parse_trade(self, data: dict[str, Any], client_order_id: str) -> UnifiedTrade:
        raise NotImplementedError


def build_mybroker(
    feed: DataFeed, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
) -> GatewayBundle:
    _ = (feed, symbols, use_aggregator)
    trader_gateway = MyTraderGateway()
    return GatewayBundle(
        market_gateway=None,  # 纯交易 broker：行情走 akquant 现有 feed
        trader_gateway=trader_gateway,
        trader_capabilities=trader_gateway.get_capabilities(),
        metadata={"broker": "mybroker"},
    )


register_broker("mybroker", build_mybroker)
```

## 参考实现与相关文档

- [自定义 Broker 注册](custom_broker_registry.md) —— `register_broker` /
  `create_gateway_bundle` 等注册 API 的详细说明。
- [Broker Capability Matrix](broker_capability_matrix.md) —— 各内置 broker
  的能力矩阵与统一错误规范，声明 `BrokerCapability` 前建议先对照。
