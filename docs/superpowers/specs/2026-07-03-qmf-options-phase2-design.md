# QMF 期权对接设计（Phase 2）

日期：2026-07-03
状态：待评审
前置：Phase 1（证券）+ Phase 1.5（插件化核心）已完成并合入 dev。本阶段基于新接缝
（`asset_type` + 能力约束的 `extra` + `features` + `TraderGatewayBase` + 可选行情 + registry）实现。

## 1. 目标

让 QMF broker 支持**期权**交易（沪深 ETF/个股期权），与证券共存于同一个 `broker="qmf"`：
策略按 `asset_type="option"` 下单，经 chibi_quant `/api/v1/option/*` 路由到期权柜台会话
（`asset_prop="B"`）。证券路径保持不变。

**不改 akquant 核心**——本阶段全部落在 `python/akquant/gateway/brokers/qmf/` 与 `builtins.py`，
正是 Phase 1.5 追求的「插件表达自身语义」的验证。

## 2. 已确认决策

- **会话模型 = A：单 qmf 双会话**。一个 `QMFTraderGateway` 内部持两个 `QMFHttpClient`：
  证券会话（`asset_prop="0"`）与期权会话（`asset_prop="B"`），按 `req.asset_type` 路由。
- **期权语义走 `extra`（不复用 position_effect）**：期权开平/备兑/委托属性经
  `extra={"entrust_oc","covered_flag","entrust_prop"}` 传入,由 `broker_extra_fields` 声明并校验。
  避免与 futures 风味的 `position_effect` 语义重叠。
- **期权开关 `enable_options`**（gateway_options，默认 False）：为 True 时 `connect()` 额外登录
  期权会话（fail-fast）、能力矩阵声明期权字段/`features`、接受期权下单；为 False 时
  证券行为完全不变，期权下单报清晰错误。证券-only 用户零影响。

## 3. chibi_quant 期权接口（已核实）

- 登录：同 `/api/v1/auth/login`，`asset_prop="B"`（网关按角色桶隔离证券/期权会话）。
- 下单 `POST /api/v1/option/order`（`OptOrderRequest`）：`exchange_type`、`option_code`、
  `entrust_amount`、`opt_entrust_price`、`entrust_bs`(1买/2卖)、`entrust_oc`(O开/C平/X行权)、
  `covered_flag`(1备兑/0非)、`entrust_prop`(F 系列)、`batch_no?`。返回含 `entrust_no`。
- 撤单 `POST /api/v1/option/cancel`：`entrust_no`、`exchange_type?`。
- 查询：`/option/orders`(338020)、`/option/trades`(338021)、`/option/positions`(338023)、
  `/option/assets`(338022，需 `money_type`，默认 "0" 人民币)。
- 会话字段（user_token/asset_prop/option_account 等）由网关注入,客户端不发。

## 4. 组件设计（均在 qmf 包内 + builtins）

### 4.1 `mapper.py`（追加期权纯函数）
- `build_option_order_payload(req) -> dict`：
  - `exchange_type, option_code = split_symbol(req.symbol)`（复用；`option_code`=数字段，如
    `"10004321.SH"→(1,"10004321")`）。
  - `entrust_bs`(复用 side 映射)、`entrust_amount`(quantity)、`opt_entrust_price`(price)。
  - `entrust_oc = req.extra["entrust_oc"]`（必填 O/C/X；缺失 → ValueError 清晰提示）。
  - `covered_flag = req.extra.get("covered_flag", "0")`。
  - `entrust_prop = req.extra.get("entrust_prop")`（必填；缺失 → ValueError，或用 builder 配置的
    默认 `option_entrust_prop`）。
- `parse_option_order(row, client_order_id="") -> UnifiedOrderSnapshot`、
  `parse_option_trade(row, client_order_id="") -> UnifiedTrade`、
  `parse_option_position(row) -> UnifiedPosition`：字段名以期权返回为准
  （`option_code`→symbol via join_symbol、`opthold_type`/`enable_amount`/`hold_amount` 等）。
  状态映射先复用 `map_order_status`（若期权 `entrust_status` 集不同,列入 §9 待确认，先按证券集映射）。

### 4.2 `client.py`（追加期权 HTTP 方法，仍单会话类；期权会话是另一个实例）
- `place_option_order(fields) -> dict` → `/api/v1/option/order`
- `cancel_option_order(entrust_no, exchange_type=None) -> dict` → `/api/v1/option/cancel`
- `query_option_orders()/query_option_trades()/query_option_positions() -> list[dict]`
- `query_option_assets(money_type="0") -> dict` → `/api/v1/option/assets`
- 期权会话是一个 `asset_prop="B"` 的 `QMFHttpClient` 实例（复用现有登录/信封逻辑，无需改会话代码）。

### 4.3 `adapter.py`（`QMFTraderGateway` 增期权路由）
- 构造增 `option_client: QMFHttpClient | None`（builder 在 `enable_options` 时注入）。
- `connect()`：证券 `login()`（不变）；若有期权会话则 `option_client.login()`（fail-fast）。
- `place_order(req)`：`req.asset_type == "option"` →（无期权会话则 RuntimeError“需 enable_options”）
  → `option_client.place_option_order(mapper.build_option_order_payload(req))` → `entrust_no`；
  `self.record_broker_order(entrust_no, req.client_order_id)` 且记 `self._option_broker_ids.add(entrust_no)`
  以便撤单/回报判源。否则走现有证券路径。
- `cancel_order(broker_order_id)`：若在 `_option_broker_ids` → `option_client.cancel_option_order`；否则证券。
- `query_positions()`：证券持仓 +（启用则）期权持仓，合并返回。
- `query_orders`/`sync_open_orders`/`query_trades`/`sync_today_trades`：证券 +（启用则）期权,合并。
- `query_account()`：**Phase 2 返回证券资金**；期权资产（`/option/assets`）本阶段不并入统一账户
  （列入 §9，后续再定聚合口径）。
- 能力：`default_capability(enable_options)` → 启用时
  `broker_extra_fields=("entrust_oc","covered_flag","entrust_prop")`、`features=frozenset({"options"})`；
  未启用维持 Phase 1 证券能力。
- 推送 `_dispatch_push`：证券会话 WS 已有；期权回报若走同一 MC 推送则复用;若期权需独立 WS 订阅,
  本阶段以 `BrokerRecovery` 的 HTTP 轮询（`sync_*` 已合并期权）补齐,期权实时 WS 列入 §9。

### 4.4 `builtins.py`（`_build_qmf` 增期权装配）
- 读 `enable_options`(默认 False)；启用时用基础配置派生 `asset_prop="B"` 的 `option_client`
  （账号同证券,可用 `option_*` gateway_options 覆盖），构造 `QMFTraderGateway(..., option_client=...)`,
  能力按启用声明。未启用时与现状完全一致。

## 5. 数据流（期权下单）

`submit_order(symbol="10004321.SH", side="Buy", quantity=1, price=0.05, asset_type="option",
extra={"entrust_oc":"O","covered_flag":"0","entrust_prop":"F0"})`
→ order_submitter：`validate_broker_extra`(对照 `broker_extra_fields`) + `normalize_asset_type`
→ `UnifiedOrderRequest(asset_type="option", extra={...})`
→ `QMFTraderGateway.place_order` 判 `asset_type=="option"` → `option_client.place_option_order`
（mapper 组包）→ `entrust_no` 记入反查表 + 期权 id 集 → 回报经 `sync_*`/推送归一分发。

## 6. 错误处理

- 期权下单但 `enable_options=False`/无期权会话 → RuntimeError“期权交易需 enable_options 并配置期权会话”。
- `extra` 缺 `entrust_oc`/`entrust_prop` → mapper ValueError 清晰提示缺失字段。
- 期权会话登录失败(connect 时) → 抛出（fail-fast），不静默降级。
- 其余沿用 Phase 1（信封 `result!="0"`、错误码归类）。

## 7. 测试策略（TDD）

- mapper：`build_option_order_payload`（symbol 拆分、extra 取值、缺字段报错）、
  `parse_option_{order,trade,position}`（用 `mock_data/opt_*.json` 为基准）。
- client：`httpx.MockTransport` 打桩 `/option/*`，验证 Bearer/信封/列表解析。
- adapter：fake 双 client，验证 `asset_type` 路由（stock→证券、option→期权）、撤单判源、
  查询合并、`enable_options=False` 时期权下单报错、能力矩阵随启用变化。
- 集成/端到端：`chibi_quant --mock` 起服务，`enable_options=True` 跑期权下单/查询链路 exit 0
  （mock 有 `opt_*.json`）；示例 `examples/41_qmf_option_live_demo.py` 实跑。
- 全程：现有 gateway/qmf 测试保持全绿；ruff `E,F,I,D`；提交 `--no-verify`,末尾完整门禁。

## 8. 不做（YAGNI / 冻结）

- 组合策略（338013/14）、行权指派/交割管理、备兑划转、历史查询、可交易数量(338010) —— 全部后续。
- 期权行权作为独立业务流程（`entrust_oc="X"` 作为下单方向可传，但行权指派/交割接口不接）。
- 期权资产并入统一 `query_account`（本阶段 query_account=证券）。
- 期权独立实时 WS 订阅（本阶段靠 HTTP `sync_*` 合并补齐）。
- 不改 akquant 核心（本阶段应零核心改动——若发现必须改核心,停下来评审）。

## 9. 待确认

1. 会话模型 A（默认已选；B 见问题）。
2. 期权 `entrust_prop` 取值（F0/F1/… 的确切语义与默认值）——先要求 `extra` 传入或 builder 配默认。
3. 期权 `entrust_status` 状态集是否与证券一致（先复用证券映射，真实柜台核对）。
4. 期权会话账号是否与证券同一账号（默认同账号、仅 `asset_prop="B"`；可 `option_*` 覆盖）。
5. 期权资产聚合进 `query_account` 的口径（本阶段不并入）。
6. 期权回报是否走同一 MC 推送（否则本阶段靠 HTTP 轮询,实时 WS 后补）。
