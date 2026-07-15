# middleware broker

akquant 的 `middleware` broker：作为 **TradeTools2.0 标准化交易中间件**
（`/api/v1/accounts/{id}` 标准 API）的 HTTP/WS 客户端，映射到 akquant 的
`TraderGateway` / `Unified*` 模型。与直连前置机的 `qmf` broker 并存，按部署选择。

- 不改 TradeTools2.0（纯消费其暴露的标准 API）。
- 零 akquant-core 改动：仅本目录 + `brokers/builtins.py` 一行注册。

## 分层

| 文件 | 职责 |
|---|---|
| `mapper.py` | 纯函数：`symbol<->instrument_id`、status 映射、`build_order_body`、`parse_*` |
| `client.py` | `MiddlewareHttpClient`(httpx)：统一信封、`/sessions`、`/accounts/{id}/orders\|cancel\|positions\|trades\|orders\|summary` |
| `ws.py` | `MiddlewarePushClient`：`/ws?accounts=`，`book.order`/`book.trade` → `on_push` |
| `adapter.py` | `MiddlewareTraderGateway(TraderGatewayBase)`：实现 `TraderGateway` 协议 |

## 用法

```python
from akquant.gateway.brokers.builtins import register_builtin_brokers
from akquant.gateway.registry import create_registered_gateway_bundle

register_builtin_brokers()
bundle = create_registered_gateway_bundle(
    "middleware",
    feed=None,
    symbols=[],
    use_aggregator=False,
    base_url="http://gw.host/api/v1",
    broker_id="hengsheng",
    fund_account="20432166",
    password="******",
    account_type="security",   # 默认 security
    ws_url="ws://gw.host/api/v1/ws?accounts=...",
    token="",                  # 可选 service JWT（Authorization: Bearer）
    enable_options=False,
)
trader = bundle.trader_gateway
trader.connect()   # POST /sessions，存 account_id
trader.start()     # WS 长连，book.* → on_order/on_trade/on_execution_report
```

## 端点映射

| TraderGateway 方法 | 中间件端点 |
|---|---|
| `connect()` | `POST /sessions`（存 `data.account.account_id`） |
| `disconnect()` | 停 WS + 关连接 |
| `start()` | `WS /ws?accounts={id}` |
| `place_order(req)` | `POST /accounts/{id}/orders` → `broker_order_id` |
| `cancel_order(bid)` | `POST /accounts/{id}/cancel` |
| `query_order(bid)` / `sync_open_orders()` | `GET /accounts/{id}/orders` |
| `query_trades()` / `sync_today_trades()` | `GET /accounts/{id}/trades` |
| `query_positions()` | `GET /accounts/{id}/positions` |
| `query_account()` | `GET /accounts/{id}/summary` |
| `heartbeat()` | `GET /sessions`（判断该 account 在线） |
| `on_order` / `on_execution_report` ← | WS `book.order` |
| `on_trade` ← | WS `book.trade` |

## 开放项（与中间件团队对齐点，隔离在 `mapper.py` 一处）

1. `instrument_id` 格式：`600000.SH`↔`SSE:600000`、`x.SZ`↔`SZSE:x`；期权用 `SSE_OPT:`/`SZSE_OPT:`。
2. status 取值集：`pending→NEW`、`submitted→SUBMITTED`、`partially_filled→PARTIALLY_FILLED`、
   `filled→FILLED`、`cancelled|partially_cancelled→CANCELLED`、`rejected→REJECTED`，其它→`SUBMITTED`。
3. WS 对 API 单必推 `book.order`/`book.trade`，帧含 `client_order_id`/`broker_order_id`。
4. 鉴权：可选 `Authorization: Bearer <token>`（config `token`）。
5. `query_account` 用 `/summary`（`data` 直接是 summary 对象）。

## 测试

`tests/test_gateway_middleware_{mapper,client,ws,adapter,factory}.py`，
用 httpx `MockTransport` + 假 WS 帧，无需活的 TradeTools2.0。
