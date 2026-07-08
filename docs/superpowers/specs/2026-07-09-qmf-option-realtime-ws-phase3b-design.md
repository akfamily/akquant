# QMF 期权实时推送 WS（Phase 3b）设计

日期：2026-07-09 · 分支：dev · 状态：设计已批准，待写实现计划

## 目标

为 akquant 的 QMF broker（`python/akquant/gateway/brokers/qmf/`）补全**期权委托/成交的实时推送**：启用期权会话（`enable_options`）时，除既有的证券推送 WS 外，再建立**第二路 WS 绑定期权 token**，把期权柜台的实时回报（issue_type 33011 成交 / 33012 委托）分发到策略的 `on_trade` / `on_order`。补全期权 live 交易闭环（此前期权回报仅能轮询查询）。

对应此前推迟的 Phase 3b。证券侧推送、期权 HTTP 下单/撤单/查询（Phase 2/3a）已完成，本阶段仅补期权推送。

## 背景与协议（已核实）

- 服务端 `chibi_quant` 推送按路由键 **`(account_type, fund_account)`** 扇出（`push/hub.py`）。WS 连接的 `account_type` 由其 token 所属会话决定：证券登录 `asset_prop="0"` → `securities`；期权登录 `asset_prop="B"` → `options`。
- 推送帧（`push/protocol.py::push_frame`）：`{type:"push", event, account_type, issue_type, ...data}`。`event` 为 `order_update` / `trade_update`，**证券与期权共用 event 名**，靠 `account_type` 区分。issue_type 映射（`t2/push_codec.py`，已确认）：12 成交/23 委托 = 证券；**33011 成交 / 33012 委托 = 期权**。
- 结论：一条 WS 只能收到其 token 对应 `account_type` 的推送。**必须为期权 token 单独建第二路 WS**。

akquant 侧现状：`QMFPushClient`（`ws.py`）已通用（`ws_url` + `token` + `on_push`，`run_forever(reconnect=5)` 自动重连）；`mapper.parse_option_order/parse_option_trade`、`_emit_order/_emit_trade/_emit_exec_from_order`、按 `trade_id` 去重的 `broker_event_bridge` 均已就绪；`adapter.query_trades()` 已合并证券+期权成交，故通用 `broker_recovery` 冷启动 seed 已自动覆盖期权。

## 方案

采用**第二个 `QMFPushClient` 实例 + 独立 `_dispatch_option_push`**（与证券 `start()` 对称）。

被否方案：把 `QMFPushClient` 泛化为多路 `[(token, dispatch)]`（仅 2 路，抽象收益低，YAGNI）；单连接多路复用（服务端按 token 路由，技术上不可行）。

## 改动面（仅 `brokers/qmf/`，零 akquant-core 改动）

- `adapter.py`
  - `start()`：证券推送不变；`self._option_client is not None` 时，额外建 `self._option_push = QMFPushClient(ws_url, self._option_client.token, on_push=self._dispatch_option_push)` 并 `start()`。
  - `disconnect()`：同时 `stop()` 期权推送路（存在才停）。
  - 新增 `_dispatch_option_push(event, data)`：对称于 `_dispatch_push`：
    - `trade_update` → `mapper.parse_option_trade(data, client_order_id)` → `_emit_trade`
    - `order_update` → `mapper.parse_option_order(data, client_order_id)` → `_emit_order` + `_emit_exec_from_order`
    - `client_order_id = self.client_order_id_for(str(data.get("entrust_no","")))`
- `ws.py`：`QMFPushClient` **不改**。加固项（可选，计划阶段决定是否纳入）：`_handle_message` 把帧顶层 `account_type` 透传给 `on_push`，`_dispatch_option_push` 断言其为 `"options"`，防止路由异常时误把证券帧当期权解析。
- `mapper.py` / 冷启动 seed / bridge 去重：**不改**（解析器已存在；`query_trades()` 合并 + 通用 recovery 已 seed 期权成交；去重按 `trade_id` 通用）。

## 数据流

期权柜台 33011/33012 → 服务端 `push_frame(event, account_type="options")` → 期权 token WS → `QMFPushClient._handle_message` → `_dispatch_option_push` → `parse_option_*` → `_emit_*` → `broker_event_bridge`（按 `trade_id` 去重）→ 策略 `on_trade` / `on_order`。

## 正确性

- **去重**：继承 `broker_event_bridge` 的 `trade_id` 去重（WS 与查询共享）。
- **冷启动 seed**：继承 —— 通用 `broker_recovery.sync_today_trades` → `adapter.query_trades()`（已合并期权）→ `mark_trades_seen`，避免把连线前的历史期权成交当新 `on_trade` 重放。
- **重连**：沿用 `run_forever(reconnect=5)`。

## 已知假设 / 风险

- **`entrust_no` 跨会话唯一性**：反查表 `record_broker_order/client_order_id_for`（`TraderGatewayBase`）为证券+期权**共享**、以 `entrust_no` 为键；`_option_broker_ids` 仅用于撤单路由。若证券与期权柜台分配出相同 `entrust_no` 会互相覆盖 —— 此为**既有假设**（期权查询路径已依赖同一 `client_order_id_for`），3b 不新增此风险，仅记录。若未来需硬隔离，可给反查表加 `account_type` 维度（超出 3b 范围）。

## 测试与验证

- **单元（主）**：仿 `test_gateway_qmf_ws.py`，构造 `account_type="options"` 的 33011/33012 帧喂 `_dispatch_option_push`，断言 emit 出期权 `trade`/`order`（含 `entrust_no→client_order_id` 反查、`_emit_exec_from_order`）。
- **集成**：扩 `test_gateway_qmf_option_integration.py`，覆盖启用期权时第二路 WS 的建立/停止、与证券路并存。
- **e2e vs `chibi_quant --mock`**：**mock 模式不主推业务事件**（`_mock_lifespan` 建了 `PushHub` 但无 T2 回调 publish）。故 e2e 只能确认：期权 token WS 成功连上并收到 `ready` 帧且 `account_type="options"`（验证 token→路由绑定端到端正确），**无法**验证真实成交推送 —— 真实推送逻辑由单元测试覆盖。
- ruff clean、mypy、全量 pytest 通过。

## 范围外（后续阶段）

3c（行权+备兑）、3d（组合多腿）、A（期权合约/额度只读）、F（可转债交易）；期权推送的重连补偿/断线期间事件回补（依赖通用 recovery，已够用）。
