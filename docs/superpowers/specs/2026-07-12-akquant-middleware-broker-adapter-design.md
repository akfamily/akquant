# akquant middleware broker adapter 设计

日期：2026-07-12 · 分支：dev · 状态：设计（用户已指示直接实现）

## 目标

在 akquant 新增一个 broker：`middleware`，作为 **TradeTools2.0 标准化交易中间件**（`/api/v1/accounts/{id}` 标准 API）的 HTTP/WS 客户端，映射到 akquant 的 `TraderGateway`/`Unified*`。与既有 `qmf`（直连 chibi_quant 前置机）并存、按部署选择。

- **不改 TradeTools2.0**（纯消费其已暴露 API）。
- **零 akquant-core 改动**（仅新增 `gateway/brokers/middleware/` + `register_broker("middleware", ...)`，与 ctp/qmf 同构）。
- 只覆盖《akquant 策略侧最小契约》的 broker 不变面：会话/下单/撤单/查询/回报。不碰 business-applications/lifecycle/exit-triggers/performance/risk。
- 验证：httpx `MockTransport` 单测 + 假 WS 帧，不需要活的 TradeTools2.0。

## 分层（新增文件）

```
python/akquant/gateway/brokers/middleware/
  __init__.py
  mapper.py    # 纯函数：symbol<->instrument_id、status 映射、build_order_body、parse_*
  client.py    # MiddlewareClient(httpx)：统一信封、/sessions、/accounts/{id}/orders|cancel|positions|trades|orders|summary
  ws.py        # MiddlewarePushClient：/ws?accounts=，book.order/book.trade -> on_push
  adapter.py   # MiddlewareTraderGateway(TraderGatewayBase)：实现 TraderGateway 协议
tests/
  test_gateway_middleware_mapper.py
  test_gateway_middleware_client.py
  test_gateway_middleware_ws.py
  test_gateway_middleware_adapter.py
  test_gateway_middleware_factory.py
```

## 契约要点（依据前端对接文档 + 策略侧最小契约）

- 统一信封：`{success, code, msg, data, trace_id}`；`success==true` 取 `data`，否则抛 `MiddlewareApiError(code,msg)`。
- 鉴权：可选 `Authorization: Bearer <token>`（service JWT），配置项，无则不带。
- `account_id = "{broker}:{fund}:{type}"`，路径里 `encodeURIComponent`（Python 用 `urllib.parse.quote(account_id, safe="")`）。

### 端点映射（TraderGateway 方法 → 中间件）
| 方法 | 端点 |
|---|---|
| `connect()` | `POST /sessions`（存 `data.account.account_id`） |
| `disconnect()` | `DELETE /sessions/{id}` |
| `start()` | `WS /ws?accounts={id}` |
| `place_order(req)` | `POST /accounts/{id}/orders` → 返回 `broker_order_id` |
| `cancel_order(bid)` | `POST /accounts/{id}/cancel` |
| `query_positions()` | `GET /accounts/{id}/positions` |
| `query_account()` | `GET /accounts/{id}/summary` |
| `query_trades(since)`/`sync_today_trades()` | `GET /accounts/{id}/trades` |
| `query_order(bid)`/`sync_open_orders()` | `GET /accounts/{id}/orders` |
| `heartbeat()` | `GET /sessions`（判断该 account 在线） |
| `get_capabilities()` | 静态 `BrokerCapability`（options 由 kwargs 决定） |
| `on_order/on_execution_report` ← | WS `book.order` |
| `on_trade` ← | WS `book.trade` |

### 字段映射（mapper.py）
- **下单** `UnifiedOrderRequest` → body：`client_order_id`、`instrument_id=symbol_to_instrument(symbol, asset_type)`、`side=Buy/Sell→buy/sell`、`offset=position_effect`（auto→open）、`order_type=Limit/Market→limit/market`、`price`、`quantity`、`legs=[]`、`extra` 透传。
- **symbol↔instrument_id**（**§开放项 1，隔离在此一处**）：`600000.SH`↔`SSE:600000`、`x.SZ`↔`SZSE:x`；asset_type=option 时用 `SSE_OPT:`/`SZSE_OPT:`；反向按 `SSE/SZSE(+_OPT)` → `.SH/.SZ`。
- **status 映射**（**§开放项**）：`submitted→SUBMITTED`、`partially_filled→PARTIALLY_FILLED`、`filled→FILLED`、`cancelled|partially_cancelled→CANCELLED`、`rejected→REJECTED`、`pending→NEW`、其它/`unknown→SUBMITTED`。
- **回执/回报** 标准 `Order`→`UnifiedOrderSnapshot`；`book.trade`→`UnifiedTrade`；`positions`→`UnifiedPosition`；`summary`→`UnifiedAccount`（`net_asset→equity`、`available→available_cash`、`cash_balance/cash→cash`）。
- `client_order_id` 反查：`place_order` 后 `record_broker_order(broker_order_id, client_order_id)`；WS 回报按 `client_order_id` 反查（依赖中间件回显，见契约 §3）。

## 明确假设（开放项，隔离可调）

以下取默认值实现，并集中在 mapper/config 一处，待中间件团队确认后改一处即可：
1. instrument_id 格式（上）。
2. status 取值集（上）。
3. WS 对 API 单必推 `book.order`/`book.trade`，含 `client_order_id`。
4. 鉴权 Bearer JWT（config 可选）。
5. `query_account` 用 `/summary`（`data` 直接是 summary 对象）。

## 范围外

business/lifecycle/exit-triggers/performance/risk/operations；端到端真连（需活的 TradeTools2.0，另行）。
