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

可选实现：

- `classify_order_error(exc: BaseException) -> UnifiedErrorType` —— 把
  `place_order`/`cancel_order` 抛出的异常分类成「柜台明确回绝」还是「订单状态
  不可知」，供核心决定该回吐拒单事件（`on_reject` + `record_reject`）还是只报
  「状态未知」（`on_error(exc, "order_submit", request)` + CRITICAL 日志，绝不
  伪造拒单）。这是唯一懂本 broker 错误码语义的地方——核心仓不 import 插件，
  分类知识只能留在这里。

  **保守缺省：不实现 = 一律按状态未知处理。** `classify_order_error` 是可选
  方法，核心用 `getattr` 探测；未实现、返回值无法识别为
  `UnifiedErrorType`、或该方法自身抛错，都会被
  `python/akquant/gateway/order_errors.py::classify_gateway_error` 归入
  `RETRYABLE`（状态未知），不会二次崩溃。这个缺省是刻意保守的：宁可让策略多
  等一轮 `sync_open_orders` 对账，也不谎报「这单没报出去」。

  **不实现的实际代价不是「行为不变」。** 未实现该方法时，本 broker 的**每一笔
  普通拒单**（如「资金不足」「委托价不符合最小变动单位」这类柜台明确回绝、
  订单确定不存在的场景）都会被当成状态未知处理——永远不会触发 `on_reject`，
  而是每次都触发 `on_error` + CRITICAL 日志。这正是本次改动要修的症状，插件
  作者不实现它就会在自己的 broker 上原样复现。

  以 `akquant-middleware` 的实现为例（`src/akquant_middleware/adapter.py`）：

  ```python
  def classify_order_error(self, exc: BaseException) -> UnifiedErrorType:
      """400 业务拒绝 / 409 幂等冲突 / 422 参数校验失败：中间件在柜台侧已把
      这笔单判掉了，订单确定不存在 → NON_RETRYABLE。

      5xx 与 httpx 传输异常（超时、连接断开）：报文可能已经转给柜台，是否
      成单不可知 → RETRYABLE，核心据此不会谎报拒单。
      """
      if isinstance(exc, MiddlewareApiError) and exc.status_code in (400, 409, 422):
          return UnifiedErrorType.NON_RETRYABLE
      return UnifiedErrorType.RETRYABLE
  ```

  `UnifiedErrorType` 的取值与「哪些分类会被视为明确拒单」定义在
  `python/akquant/gateway/order_errors.py`（`_DEFINITE_REJECT_TYPES` 目前只收
  `RISK_REJECTED` 与 `NON_RETRYABLE`；其余含未知一律按状态未知处理）。

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

## 委托状态映射：终态与非终态

`UnifiedOrderStatus` 分两类，映射错一条的代价不对称，所以单列一节：

| 类别 | 取值 | 含义 |
| --- | --- | --- |
| 非终态 | `New` / `Submitted` / `PartiallyFilled` | 还可能继续成交，`sync_open_orders()` **只应返回这三种** |
| 终态 | `Filled` / `Cancelled` / `Rejected` / `Expired` | 不会再变，核心据此关闭 id 映射、停止跟踪 |

终态集在核心侧的落点是 `python/akquant/live/_payload_utils.py`
的 `_TERMINAL_STATUSES`，三个内置 broker 的 `_is_terminal_status()` 与之同口径；
回测侧对应 `strategy_order_events._TERMINAL_ORDER_STATUSES`。`Expired` 是
日内单收盘作废、柜台把单判废这类场景，**别漏**：漏掉它等于把终态单当活单。

三条容易踩的坑：

1. **未识别的柜台状态请兜底成非终态，但必须记 `warning`。** 兜底成非终态是保守
   的（宁可多查一次挂单，也不要把活单当终态丢掉），可一旦静默，柜台新增的终态
   码值就会永远留在挂单表里，`cancel_all_orders` 每轮对它撤一次，柜台回
   「委托状态错误不能撤单」。有日志才有人发现。
2. **不要用 `status == Filled` 判断成交。** IOC / 最优五档即时成交剩余撤销这类
   委托，**正常收尾就是 `Cancelled`**（部分成交时柜台状态是「部撤」），成交量由
   `filled_quantity` 承载。策略与 broker 内部都应以 `filled_quantity` 为准。
3. **不要复用 `create_default_mapper()` 里的单字符码值。** 那张表是 CTP 口径
   （`0`=全部成交、`5`=已撤单），与恒生《数据词典》1203「委托状态」的数字码
   （`0`=未报、`5`=部撤、`8`=已成）**完全冲突**。接非 CTP 柜台请自带映射表——
   `akquant-middleware` 就是把整张表收在自己的 `mapper.py` 里的。

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
