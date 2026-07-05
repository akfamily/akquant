# broker_live 撤单与状态读路由到柜台（③b + ③c）设计

日期：2026-07-03
状态：待评审
动机：`broker_live` 下只有 `submit_order` 被路由到柜台;`cancel_order` 与 `get_position/get_account/...` 仍走 sim 引擎 `ctx`（成交从不回灌引擎）。目标：把撤单与状态读也路由到真实柜台。

## 1. 背景（已核实）

- `BrokerOrderSubmitter.install()` 目前仅覆盖 `submit_order`/`can_submit_client_order`/`get_execution_capabilities`;其余方法读写 `strategy.ctx`。
- **成交不回灌引擎**:`BrokerEventBridge` 只触发回调 + 累积 `_broker_order_states`(按 broker_order_id 存委托 payload);LiveRunner 维护 `_client_to_broker_order_ids`(client→broker id 映射)。
- **引擎无注入 API**:`akquant.pyi` 只有持仓/资金 getter,无 set/apply_fill——故"把成交回灌 ctx 让 ctx 自然正确"(方案 A)不可行(需改 Rust 核心)。
- `trader_gateway` 已有 `query_positions()->list[UnifiedPosition]`、`query_account()->UnifiedAccount|None`、`sync_open_orders()->list[UnifiedOrderSnapshot]`、`cancel_order(broker_order_id)`。

## 2. 已确认决策

- **状态源 = B：查柜台 + 事件失效缓存**。`broker_live` 重写公共读方法转发 `trader_gateway.query_*`;应答进一个短生命缓存,`on_trade`/`on_order` 推送到达时**失效**(下次读重新查),兼顾正确与调用量。柜台是唯一真相。
- **撤单(③b)**:`broker_live` 重写 `cancel_order`/`cancel_all_orders`,用现有 client→broker id 映射解出 broker_order_id,调 `trader_gateway.cancel_order`。
- **组合目标类下单(`buy_all`/`order_target_*`)**:它们直接读 `strategy.ctx.cash`/`ctx.get_position`(不经公共方法)。本阶段在 `broker_live` **明确报错**(避免按 sim 状态错误 sizing);让其读柜台状态作为后续(需改这些 impl 走公共读方法)。

## 3. 组件设计

### 3.1 新增 `BrokerStateCache`（`gateway/broker_state_cache.py`）
- 持 `trader_gateway`;方法 `positions() -> dict[str,float]`、`available_positions() -> dict[str,float]`、`account() -> UnifiedAccount|None`、`open_orders() -> list[UnifiedOrderSnapshot]`。
- 内部按类型缓存最近一次 `query_*` 结果;`invalidate()` 清缓存;读时若缓存空则查柜台并填充。
- `invalidate()` 由 broker 事件桥在 `on_trade`/`on_order` 分发时调用(委托/成交变动 → 下次读刷新)。
- 柜台查询异常 → 记日志,返回上次缓存(或空),不抛断策略。

### 3.2 `broker_runtime` / `install()` 扩展
`BrokerRuntime.install_submitter`（或新增 `install_broker_strategy_api`）在 `broker_live` 额外 `setattr` 到策略：
- `cancel_order(order_id)` → 解 `order_id`(策略侧 client_order_id) → broker_order_id(经 LiveRunner 的 `_client_to_broker_order_ids`;或 order_id 本就是 broker id 时直接用) → `trader_gateway.cancel_order(broker_order_id)`;未知 id → 清晰错误。
- `cancel_all_orders(symbol=None)` → 遍历重写后的 `get_open_orders(symbol)` → 逐个 `cancel_order`。
- `get_position(symbol) -> float` / `get_available_position(symbol) -> float` → `BrokerStateCache.positions()/available_positions().get(symbol, 0.0)`。
- `get_account() -> dict` → 由 `BrokerStateCache.account()` 的 `UnifiedAccount` 映射成与回测 `get_account` 一致的 dict 形状(cash/available/equity 等键;缺的键给 0/合理默认)。
- `get_portfolio_value() -> float` → `UnifiedAccount.equity`。
- `get_open_orders(symbol=None) -> list` → `BrokerStateCache.open_orders()`(按 symbol 过滤);元素形状与回测 `get_open_orders` 尽量一致(至少含 id/symbol/side/quantity/status)。
- `buy_all`/`order_target*`：`broker_live` 下这些**保持报错**(在其 impl 入口检测 broker_live 能力 → 清晰 `RuntimeError`,提示改用 `submit_order`/`buy` + 自行按 `get_account`/`get_position` sizing)。

### 3.3 事件失效接线
在 broker 事件分发路径（`BrokerEventBridge`/LiveRunner `_broker_dispatch`）里,`on_trade`/`on_order` 处理后调用 `BrokerStateCache.invalidate()`。

## 4. 数据流

- 读:`strategy.get_position("600000.SH")` →（broker_live 重写）→ `BrokerStateCache.positions()`（命中缓存或查 `trader_gateway.query_positions()`）→ float。
- 撤单:`strategy.cancel_order("c1")` →（重写）→ `_client_to_broker_order_ids["c1"]="9000000001"` → `trader_gateway.cancel_order("9000000001")`。
- 失效:柜台成交推送 → `on_trade` → `cache.invalidate()` → 下次读刷新。

## 5. 错误处理

- 柜台查询失败 → 日志 + 返回上次缓存/空,不断策略。
- 撤单未知 id → 清晰 `RuntimeError`。
- `buy_all`/`order_target_*` in broker_live → 清晰 `RuntimeError`（暂不支持,提示替代）。

## 6. 测试策略（TDD）

- `BrokerStateCache`：命中/失效/异常回退（用 fake gateway 的 query_* 计数验证缓存与失效）。
- 撤单路由:重写后的 `cancel_order` 用 id 映射调 `trader_gateway.cancel_order`(fake 记录);未知 id 报错。
- 状态读:重写后的 `get_position/get_account/get_portfolio_value/get_open_orders` 返回柜台值(fake query_* 数据),且经缓存(第二次不再查,直到 invalidate)。
- `buy_all`/`order_target_*` in broker_live → `RuntimeError`。
- `paper`/回测零回归(不安装这些重写)。
- 端到端:`chibi_quant --mock` 下 LiveRunner(或直接构造)验证 `get_position`/`cancel` 走柜台。
- 现有 gateway/live_runner 全绿;ruff `E,F,I,D`。

## 7. 不做（YAGNI / 冻结）

- 方案 A（回灌引擎 ctx，需 Rust 注入 API）与方案 C（事件源本地账本）——本阶段用 B。
- 让 `buy_all/order_target_*` 在 broker_live 真正按柜台 sizing（本阶段报错;后续改这些 impl 走公共读）。
- 期权资产已在 ③(Phase 3a query) 并入 `query_account`;此处 `get_account` 复用之。
- ③d 条件单不在此。

## 8. 影响面

- 新增 `python/akquant/gateway/broker_state_cache.py`
- `python/akquant/gateway/order_submitter.py` 或 `broker_runtime.py`（install 扩展 + 重写方法）
- `python/akquant/gateway/broker_event_bridge.py` 或 `live.py`（on_trade/on_order → cache.invalidate）
- `python/akquant/strategy_trading_api.py`（`buy_all`/`order_target*` broker_live 报错）
- Tests + 文档小节

## 9. 待确认

1. 状态源 B（默认已选;C/A 见问题）。
2. `buy_all/order_target_*`：本阶段**报错**（默认）vs 立刻改走柜台读。默认报错。
3. `get_account` dict 形状:以回测 `get_account` 的键为准（cash/available_cash/equity/... 逐一对齐）。
4. 缓存生命:纯事件失效（默认）vs 叠加短 TTL 兜底。默认纯事件失效 + 首次懒查。
