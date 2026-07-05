# QMF Broker Gateway 对接设计

日期：2026-07-03
状态：待评审
范围：仅实盘交易对接（backtest_server3.0 不在本次范围内）

## 1. 背景与目标

akquant 已有成熟的实盘 gateway 抽象层（`python/akquant/gateway/`）：以 `typing.Protocol`
定义 `TraderGateway` / `MarketGateway`，统一数据模型 `broker_models.py`，并有
`register_broker` / `create_gateway_bundle` 的注册-工厂插件机制。目前只有 CTP 是真实网关，
MiniQMT / PTrade 均为 stub。

`py_jinrongjie_gateway_server-develop`（Python 包 `chibi_quant`）是一个独立部署在**服务器**上的
FastAPI 服务，南向经恒生 T2 SDK 连接期魔方（QMF）券商前置机/柜台，北向对客户端暴露
**HTTP REST + 一个 WebSocket 推送**。

目标：在 akquant（部署在**客户端**）中新增一个 **QMF broker gateway 适配器**，实现
`TraderGateway` 协议，通过 HTTP/WS 跨机对接 chibi_quant，使 akquant 策略能经期魔方前置机
实盘下单/撤单/查询并接收委托与成交回报。

## 2. 已确认的约束（来自需求方）

1. **范围**：仅实盘交易对接；backtest_server3.0 暂不联动。
2. **品种**：证券（股票）与期权**都要**（分阶段，见 §8）。
3. **行情**：继续用 akquant 现有 feed（AKShare 等）。QMF 网关**不提供行情**，故本适配器
   **只实现 TraderGateway，不实现 MarketGateway**。
4. **部署形态**：akquant 在客户端，chibi_quant 在服务器 —— 跨机通信，需处理
   base_url（支持 https）、超时、断线重连、密钥安全下发。

## 3. chibi_quant 对外接口（客户端视角，已核实）

- 协议：`POST /api/v1/*`，JSON in/out，统一信封 `{"result","msg","data"}`，`result=="0"` 为成功；
  列表查询 `data` 为数组，单记录为对象。
- 鉴权：除 `/auth/login`、`/auth/status` 外均需 `Authorization: Bearer <gateway_token>`。
- 登录：`POST /api/v1/auth/login`，密码字段须 **AES-256-GCM 加密**，格式
  `base64(nonce(12B) || ciphertext || tag(16B))`，共享密钥 `CHIBI_PASSWORD_KEY`；
  参考实现 `chibi_quant/common/password_crypto.py`。返回 `data.user_token`（`gw-...`）作为 Bearer。
  证券登录 `asset_prop="0"`，期权登录 `asset_prop="B"`（会话按角色桶隔离，二者独立）。
- 证券交易/查询：
  - 下单 `POST /api/v1/trading/order`（req `OrderRequest`；关键返回 `entrust_no`、`error_no`）
  - 撤单 `POST /api/v1/trading/cancel`（req `CancelRequest`，需 `entrust_no`）
  - 资金 `POST /api/v1/account/funds`、持仓 `/account/positions`、委托 `/account/orders`、成交 `/account/trades`
- 期权交易/查询：`/api/v1/option/*`（下单 `/option/order` 带 `entrust_oc` 开平、`covered_flag` 备兑；
  查询 `/option/{orders,trades,assets,positions}`）。
- 推送：`WebSocket /api/v1/stream`（Bearer 或 `?token=`）。连接后先收 `ready`，之后 `push` 帧
  `event ∈ {order_update, trade_update}`。**网关不缓存推送历史**——断线重连后必须用 HTTP 查询补齐。
- 联调：`chibi_quant --mock`（或 `CHIBI_MOCK_MODE=1`）无需真实柜台，响应取自 `mock_data/*.json`。

gateway 会自动注入会话字段（`user_token/client_id/branch_no/fund_account/...`）与操作字段
（`op_branch_no/op_entrust_way/op_station`），**客户端不发送这些**。

## 4. akquant 侧接口与既有事实（已核实）

- 协议：`python/akquant/gateway/protocols.py` 的 `TraderGateway`（`connect/disconnect/place_order/
  get_capabilities/cancel_order/query_order/query_trades/query_account/query_positions/
  on_order/on_trade/on_execution_report/sync_open_orders/sync_today_trades/heartbeat/start`）。
- 统一模型：`broker_models.py` —— `UnifiedOrderRequest{client_order_id, symbol, side, quantity,
  price, order_type, time_in_force, position_effect, reduce_only}`、`UnifiedOrderSnapshot`、
  `UnifiedTrade`、`UnifiedExecutionReport`、`UnifiedAccount`、`UnifiedPosition`、
  `UnifiedOrderStatus{New,Submitted,PartiallyFilled,Filled,Cancelled,Rejected}`、
  `UnifiedErrorType{retryable,non_retryable,risk_rejected}`、`BrokerCapability`。
- 注册-工厂：`registry.register_broker(name, builder)` + `factory.create_gateway_bundle(...)`；
  builder 签名 `(feed, symbols, use_aggregator, **kwargs) -> GatewayBundle`；
  参考 `examples/35_custom_broker_registry_demo.py`。
- 运行时：`LiveRunner`（`live.py`）在 `broker_live` 模式下，`BrokerOrderSubmitter`
  （`gateway/order_submitter.py`）把 `strategy.submit_order` 路由到 `trader_gateway.place_order`，
  并注册 `on_order/on_trade/on_execution_report`；`BrokerEventBridge` 去重并分发回策略；
  `BrokerRecovery` 周期性调用 `sync_open_orders/sync_today_trades` 做重连补齐。
- **关键约束**：实盘 `submit_order`（`order_submitter.py:214`）与策略层
  （`strategy_trading_api.py:765`）**均硬拒绝 `extra` 字段**；`UnifiedOrderRequest` 无品种字段、
  无期权开平/备兑/行权字段。这直接决定了期权语义的承载方式（见 §8 Phase 2）。

## 5. 组件设计（新增包 `python/akquant/gateway/brokers/qmf/`）

每个文件单一职责、可独立测试：

- `crypto.py`：AES-256-GCM 密码加密，输出 `base64(nonce||ciphertext||tag)`；与
  chibi_quant `password_crypto.py` 逐字节对齐（单测用固定 nonce 做已知向量比对）。
  依赖 `cryptography`。输入：明文密码 + base64 密钥；输出：密文串。
- `client.py`：`QMFHttpClient`。职责：会话（登录拿 Bearer、`auth/status` 保活、登出）、
  下单/撤单、各类查询；统一解 `{result,msg,data}` 信封，`result!="0"` 抛带错误码的异常；
  支持 `base_url`(http/https)、超时、有限重试。依赖 `httpx`（或 `requests`）。不含任何映射逻辑。
- `ws.py`：`QMFPushClient`。职责：连 `/api/v1/stream`，处理 `ready/push/heartbeat/ack/error/pong`，
  上行 `ping/resubscribe`，断线指数退避重连；把原始 `push` 帧交给回调。不含归一逻辑。
- `mapper.py`：纯函数集合，无 IO。职责：
  - `symbol` ↔ (`exchange_type`, `stock_code`)：如 `"600000.SH"→(1,"600000")`、`.SZ→2`。
  - `UnifiedOrderRequest → OrderRequest`：`side` Buy/Sell→`entrust_bs` 1/2；`order_type`→`entrust_prop`；
    `price/quantity`→`entrust_price/entrust_amount`。
  - 柜台 `entrust_status` → `UnifiedOrderStatus`；`error_no/error_info` → `UnifiedErrorType`
    （匹配中文关键字如"风控"→risk_rejected，"连接/超时"→retryable，其余→non_retryable）。
  - 查询响应/推送帧 → `UnifiedOrderSnapshot/UnifiedTrade/UnifiedExecutionReport/
    UnifiedAccount/UnifiedPosition`。
- `adapter.py`：`QMFTraderGateway`，实现 `TraderGateway` 协议。持有 `QMFHttpClient` +
  `QMFPushClient` + mapper。`start()` 起 WS，收到 `push` 帧→mapper 归一→触发已注册的
  `on_order/on_trade/on_execution_report`。`sync_open_orders/sync_today_trades` 走 HTTP 查询
  （供 `BrokerRecovery` 断线补齐）。`get_capabilities()` 返回证券能力矩阵。
- `builder.py`：`register_broker("qmf", builder)`；从 `gateway_options`/kwargs 读取
  `base_url`、`ws_url`、登录参数、`password_key` 等，构造 `GatewayBundle(trader_gateway=...,
  market_gateway=<沿用现有 feed 侧>, trader_capabilities=...)`。

## 6. 数据流

1. `LiveRunner(broker="qmf", trading_mode="broker_live", gateway_options={...})` →
   `create_gateway_bundle` → builder → `QMFTraderGateway`。
2. `connect()`：AES-GCM 加密密码 → `auth/login` → 存 Bearer。`start()`：起 WS 长连。
3. 策略 `submit_order(...)` → `BrokerOrderSubmitter` 构造 `UnifiedOrderRequest` →
   `QMFTraderGateway.place_order` → mapper → `client` POST `/trading/order` →
   返回 `entrust_no` 作为 `broker_order_id`（akquant 侧维护 client↔broker id 映射）。
4. 回报：WS `push`(order_update/trade_update) → mapper 归一 → `on_order/on_trade`。
5. 重连补齐：`BrokerRecovery` 周期调用 `sync_open_orders/sync_today_trades`（HTTP 查询）。

## 7. 错误处理

- HTTP 层：非 2xx / 连接错误 → 归类 retryable，有限重试后上抛。
- 业务层：`result!="0"` 或 `error_no!="0"` → 按 `msg/error_info` 关键字归类
  `UnifiedErrorType`，下单失败反映为 `UnifiedOrderStatus.REJECTED` + `reject_reason`。
- WS：断线 → 指数退避重连；重连后依赖 `BrokerRecovery` 的 HTTP 补齐弥补丢失推送。
- 会话过期：`auth/status` 保活；失效则重登录并重连 WS。

## 8. 分阶段实施

### Phase 1 — 证券/股票（本设计完整覆盖）
`crypto → client → mapper → ws → adapter → builder`，先打 `chibi_quant --mock` 全链路跑通
（登录→下单→撤单→查询→WS 回报），再切真实柜台。能力矩阵：`position_effect=False,
supports_short_sell=False`（A 股 T+1、无开平概念）。

### Phase 2 — 期权（待决策后再动手）
`/api/v1/option/*`，独立 `asset_prop="B"` 会话。核心难点：期权需 `entrust_oc`（开O/平C/行权X）
与 `covered_flag`（备兑），而 `UnifiedOrderRequest` 无此字段、`extra` 被禁。三个候选方案：

- **A（推荐）扩展 akquant 核心层**：为 `UnifiedOrderRequest` 增可选期权字段，并为声明能力的
  broker 放开 `extra` 透传（改 `order_submitter.py` / `strategy_trading_api.py`）。开平/备兑/行权
  全支持，长期干净；代价是改核心并需相应测试。
- **B 约定推断，不改核心**：QMF 网关从 symbol 格式判别证券/期权，用 `position_effect`(open/close)
  映 `entrust_oc`；备兑/行权暂不支持。改动最小但能力不完整。
- **C 两个注册 broker**：`qmf`(证券) 与 `qmf_option`(期权) 各自会话；期权侧仍需解决字段承载。

> **决策待确认**：Phase 2 采用哪个方案。默认倾向 A，但因其触及核心，实现前需求方需明确同意。
> Phase 1 不依赖此决策，可先行。

## 9. 测试策略（TDD）

- `mapper.py`：纯函数，用 `mock_data/*.json` 为基准写单测（symbol 拆分、状态/错误码映射、
  各响应→Unified* 模型）。
- `crypto.py`：已知向量（固定 nonce）比对，确保与 chibi_quant 可互解。
- `client.py` / `adapter.py`：对 `chibi_quant --mock` 起的服务做集成测试，覆盖
  登录→下单→撤单→查询→WS 回报全链路。
- 遵循 CLAUDE.md：`uvx ruff check` + `uvx ruff format --check` 通过；示例须实跑 exit 0。

## 10. 不做（YAGNI）

- 不实现 QMF MarketGateway（行情用现有 feed）。
- 不联动 backtest_server3.0。
- 不在 Phase 1 引入期权。
- 不实现 audit 端点（服务器间用途）。

## 11. 待确认问题清单

1. Phase 2 期权字段方案（A/B/C，默认 A）。
2. `CHIBI_PASSWORD_KEY` 如何安全下发到客户端 akquant 侧（配置文件 / 环境变量 / 密钥管理）。
3. `order_type` Market/Limit → 柜台 `entrust_prop` 的确切取值（mock 仅示 `0`，需真实 T2 语义确认）。
4. 跨机是否有 TLS/反向代理（base_url 用 https 与否）。
