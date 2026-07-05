# P2：on_order/on_trade 事件模型统一（回测/实盘同形状）设计

日期：2026-07-04
状态：待评审
动机：策略 `on_order`/`on_trade` 回调在回测收到 Rust `Order`/`Trade`，在 broker_live 收到 `UnifiedOrderSnapshot`/`UnifiedTrade`——字段名/类型/枚举都不同（`avg_fill_price` vs `average_filled_price`、`trade_id` vs `id`、`status` 是 `UnifiedOrderStatus` vs `OrderStatus`、`side`/`position_effect` 是 str vs 枚举），且 `UnifiedOrderSnapshot` 缺 `side`/`quantity`/`price`/`order_type`。一个按回测写法的策略在实盘 `on_order/on_trade` 里会 `AttributeError` 或读到不同语义。P2 让两模式回调看到**同一属性形状**（backtrader/vnpy 的一致数据模型教训）。

延续既有路线：P1（ExecutionBackend seam）与 Strategy API v2 已合入 dev。本阶段 P2 只碰事件派发层。

## 0. 范围与边界

- **纯 Python**：Rust 引擎（`Order`/`Trade`/`OrderStatus` 等 pyo3 类型、`_on_*_event` 派发、`strategy_order_events` 回测派发）**不改**，无 `.rs`/`.pyi` 编辑、无 `cargo build`。
- **回测零回归**：回测继续投递原生 Rust `Order`/`Trade`（它们**原生满足**目标属性契约）。只改 broker_live 派发，使其投递**形状兼容**对象。
- 采用 **duck-typed 同形状**（同属性名 + 同枚举类型），不强求同一个类（Rust `Order` 无法从 Python 任意构造）。
- `on_execution_report` 是 broker-live/FIX 概念，回测无对应；**不强行统一**（仍投 `UnifiedExecutionReport`），只补基类 no-op 避免崩溃。

## 1. 目标属性契约（策略回调可读的字段）

`on_order(order)` 两模式均可读（回测 Rust `Order` 已具备；broker_live 适配器补齐）：
`id, symbol, side, order_type, time_in_force, status, quantity, price, trigger_price, filled_quantity, average_filled_price, commission, position_effect, reduce_only, tag, reject_reason, created_at, updated_at`。

`on_trade(trade)` 两模式均可读：
`id, order_id, symbol, side, timestamp, quantity, price, commission, position_effect`。

枚举类型对齐（broker_live 适配器产出与回测同款 Rust 枚举实例）：
- `status` → `akquant.OrderStatus`
- `side` → `akquant.OrderSide`
- `order_type` → `akquant.OrderType`
- `position_effect` → `akquant.PositionEffect`
- `time_in_force` → `akquant.TimeInForce`

这样 `order.status == OrderStatus.Filled`、`trade.side == OrderSide.Buy` 在两模式都成立。

## 2. 组件设计

### 2.1 新增 `gateway/broker_event_adapter.py`
- `@dataclass StrategyOrder`：上面 on_order 契约的全部字段 + 附加 broker 字段 `client_order_id`、`broker_order_id`（附加无害）。
- `@dataclass StrategyTrade`：on_trade 契约字段 + 附加 `client_order_id`、`broker_order_id`。
- `_STATUS_MAP: dict[UnifiedOrderStatus, OrderStatus]`：`NEW→New, SUBMITTED→Submitted, PARTIALLY_FILLED→PartiallyFilled, FILLED→Filled, CANCELLED→Cancelled, REJECTED→Rejected`。（broker 无 `Accepted`/`Expired`；不反向映射。）
- 枚举转换 helper（容错，大小写/别名）：`_to_order_side(str)->OrderSide`、`_to_position_effect(str)->PositionEffect`、`_to_order_type(str)->OrderType`、`_to_time_in_force(str)->TimeInForce`；无法解析给合理默认（如 `position_effect` 默认 `PositionEffect.Auto`）。
- `map_order_snapshot(snapshot, request=None, owner_strategy_id=None) -> StrategyOrder`：
  - `id = broker_order_id`；`symbol/filled_quantity/reject_reason/position_effect` 直取；`average_filled_price = avg_fill_price`；`updated_at = timestamp_ns`；`status = _STATUS_MAP[snapshot.status]`。
  - `side/quantity/price/order_type/time_in_force/trigger_price/reduce_only/tag` 从 `request`（`UnifiedOrderRequest`）回填；`request=None` 时给默认（side=None、quantity=filled_quantity、price=None、order_type=None…）。
  - `commission=0.0`（broker 快照无）；`created_at=None`。
- `map_trade(trade, request=None, owner_strategy_id=None) -> StrategyTrade`：
  - `id = trade_id`；`order_id = broker_order_id`；`side = _to_order_side(trade.side)`；`timestamp = timestamp_ns`；`symbol/quantity/price/position_effect` 直取；`commission=0.0`。
- 三个 map 函数容错 dict 与 dataclass 两种输入（沿用 `payload_field` 兼容），并对未知/缺字段安全降级，不抛异常打断派发。

### 2.2 提交请求缓存（`LiveRunner`）
- 新增 `self._order_requests: dict[str, UnifiedOrderRequest]`（键 `client_order_id`）。
- `BrokerOrderSubmitter.submit_order` 现构造 `UnifiedOrderRequest` 后调 `_sync_order_id_mapping`/`_bind_order_owner`；**新增回调** `record_order_request(client_order_id, request)`，`LiveRunner` 实现为写入 `_order_requests`。（`BrokerRuntime`/submitter 已有回调注入模式，加一个。）
- 清理：终态（Filled/Cancelled/Rejected）时，在现有 id-map 清理路径（`test_live_runner_cleans_mapping_on_terminal_status` 覆盖处）一并 `pop` 掉 `_order_requests`，防无界增长。

### 2.3 派发接线（`BrokerEventBridge` + `LiveRunner`）
- `LiveRunner` 提供 `adapt_strategy_payload(event_name, payload) -> Any`：对 `order`→`map_order_snapshot(payload, request=_lookup_request(payload), owner=_resolve_owner(payload))`；`trade`→`map_trade(...)`；其余（execution_report/account）原样返回。`_lookup_request` 用 `client_order_id`（或经 `_broker_to_client_order_ids` 解）查 `_order_requests`。
- `BrokerEventBridge._dispatch_strategy_event` 在调 `on_order`/`on_trade` 前用注入的 `adapt_strategy_payload` 转换 payload（bridge 保持瘦；转换器由 LiveRunner 注入，同 `update_broker_state` 的注入方式）。`on_execution_report`/`on_portfolio_update` 不转换。

### 2.4 基类 no-op `on_execution_report`
- `Strategy` 现有 `on_order`/`on_trade` no-op（`strategy.py:1406-1422`），**无** `on_execution_report` → broker_live 下 `_safe_strategy_callback` 的 `getattr` 会 `AttributeError`。新增 `def on_execution_report(self, report): pass` 基类 no-op（与 on_order/on_trade 一致）。

## 3. 数据流

- broker 成交推送 → `UnifiedTrade` → `BrokerEventBridge.drain_events` → `adapt_strategy_payload("trade", ...)` → `StrategyTrade`（`side`=OrderSide 枚举、`id`=trade_id、`timestamp`=ns）→ `strategy.on_trade(StrategyTrade)`。策略读 `trade.side/price/quantity/symbol/commission` —— 与回测 `Trade` 同形状。
- broker 委托推送 → `UnifiedOrderSnapshot` → `adapt_strategy_payload("order", ...)`（经请求缓存回填 side/qty/price/order_type，status 映射为 `OrderStatus`）→ `strategy.on_order(StrategyOrder)`。

## 4. 错误处理

- 适配器对缺字段/未知枚举安全降级（默认值），绝不因映射失败打断事件派发（外层 `_safe_strategy_callback` 已 try/except → on_error）。
- 请求缓存未命中（如重启后收到旧单推送）：回填字段给默认（side=None 等），status/filled 等仍正确；不报错。
- Rust 枚举导入：适配器 `from .. import`（或经 akquant 顶层）拿 `OrderStatus/OrderSide/OrderType/PositionEffect/TimeInForce`；纯读枚举成员，不改 Rust。

## 5. 测试策略（TDD）

- 适配器单测：`map_order_snapshot`/`map_trade` 名对齐、status 枚举映射（全 6 变体）、side/position_effect 枚举转换、请求缓存回填 vs 缺失默认、dict 与 dataclass 输入。
- 请求缓存：submit 记录、终态清理（不泄漏）。
- 派发接线：`BrokerEventBridge` 派发前转换（用 fake，断言 `on_order` 收到的对象 `.status is OrderStatus.X`、`.average_filled_price` 存在、`.side` 来自请求）。
- **对照/parity 测试**：一个 on_order/on_trade 读 `order.side`、`order.status == OrderStatus.Filled`、`order.average_filled_price`、`trade.side == OrderSide.Buy`、`trade.price` 的策略回调，喂 (a) 回测风格 Rust-like 对象 与 (b) broker 适配对象，两者都不 AttributeError 且断言一致。
- 基类 no-op：`Strategy().on_execution_report(obj)` 不抛；broker_live 未定义该方法的策略收到 execution_report 不崩。
- 回测零回归：`strategy_order_events` 路径不改，回测 on_order/on_trade 仍收 Rust 原生对象；full `tests/` 全绿。
- ruff `E,F,I,D`；Rust 零改动断言（`git diff -- '*.rs' *.pyi` 空）。

## 6. 影响面

- 新增 `python/akquant/gateway/broker_event_adapter.py`（+ 测试）。
- 改 `python/akquant/gateway/broker_event_bridge.py`（派发前经注入的适配器转换）。
- 改 `python/akquant/live.py`（`_order_requests` 缓存 + `adapt_strategy_payload` + 注入 bridge + 终态清理 + record_order_request 回调实现）。
- 改 `python/akquant/gateway/order_submitter.py` / `broker_runtime.py`（新增 `record_order_request` 回调注入）。
- 改 `python/akquant/strategy.py`（基类 no-op `on_execution_report`）。
- 文档：`docs/zh/advanced/qmf_broker_gateway.md` 增「事件模型统一」小节（on_order/on_trade 两模式同形状；on_execution_report broker-only）。

## 7. 不做（YAGNI / 后续）

- 不把 `on_execution_report` 塞进统一形状（broker-only 概念）。
- 不改回测 Rust `Order`/`Trade`（无必要，已满足契约）。
- 不追 `Accepted`/`Expired`（broker 无此状态）。
- P3 本地条件单、BarGenerator、流式指标库等仍在后续路线图。

## 8. 待确认

1. 完整 parity（含请求缓存回填 on_order 的 side/qty/price/order_type）—— 默认已选（用户授权按最佳策略执行）。
2. 适配输出为**新 dataclass**（`StrategyOrder`/`StrategyTrade`，duck-typed 同形状）而非 Rust 类型 —— 默认（Rust 类型不可从 Python 构造）。
3. 请求缓存生命周期：提交写入 + 终态清理 —— 默认。
