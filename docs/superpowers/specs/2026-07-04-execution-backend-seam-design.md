# ExecutionBackend 执行接口 seam（回测/实盘一套化 P1）设计

日期：2026-07-04
状态：待评审
动机：让回测与实盘尽量用**同一套**策略代码。当前 akquant 是「回测读 Rust `ctx`；broker_live 用 `setattr` 猴补把 `submit_order`/状态读/撤单替换成柜台版」——两套 + 打补丁。学 backtrader 的 `self.broker` 单一 broker 接口 + vnpy 的共享数据模型，把执行面收敛成一个策略持有的后端对象 `self.execution`。

本设计是三阶段路线的 **P1（地基）**：
- **P1（本文）** 执行接口 seam：读/下单/撤单/目标类下单全走 `ExecutionBackend`，删猴补与「目标类下单 broker_live 报错」。
- P2 统一 order/trade 事件模型（`on_order/on_trade` 两模式同形状对象）。
- P3 本地条件/止损单层（`LocalStopOrder` + `on_stop_order`，两模式共用状态机）。

## 1. 背景（已核实）

- `Strategy` 持类属性 `ctx: Optional[StrategyContext]`（`strategy.py:294`）与 `execution_mode`（已存在，`strategy.py:295`）。公共方法（`get_position`/`submit_order`/…）都 delegate 到 `strategy_trading_api.py` 的自由函数。
- **回测**：所有状态读走 `strategy.ctx.*`（如 `strategy_trading_api.py:28-41`）；下单最终经 `_submit_buy_side_orders/_submit_sell_side_orders` → `ctx.buy()/ctx.sell()`；撤单 `ctx.cancel_order`（`:178-202`）。
- **Rust 引擎边界**：`StrategyContext`（`akquant.pyi`）只暴露 position/cash getter + `buy/sell/cancel_order/schedule`，**无任何 set/fill 注入 API**。故「把成交回灌引擎让 ctx 自然正确」（方案 A）不可行；broker_live 走查柜台（方案 B，已在 ③c 落地）。
- **broker_live 现状（猴补）**：`BrokerOrderSubmitter.install()`（`gateway/order_submitter.py:185-195`）`setattr` 覆盖 `submit_order`/`can_submit_client_order`/`get_execution_capabilities`；`broker_strategy_api.install_broker_state_reads/install_broker_cancel`（③c）覆盖状态读与撤单。安装点在 `live.py`（仅 `trading_mode=="broker_live"`）。
- **隐式 seam 已存在**：`buy()`（`:229`）用 `getattr(strategy,"submit_order")` 调用可替换的方法；`order_target*` 调 `buy()/sell()`——所以它们的**下单侧在 broker_live 已经路由到柜台**。它们被禁用的**唯一原因**是**读侧**直接读 `strategy.ctx.get_position()`（如 `_order_target_core` `:1316`），而不是已被覆盖的公共读方法。`_reject_target_orders_in_broker_live`（`:682-693`）在每个目标类函数入口拦截。
- **模式探测**：`get_execution_capabilities(strategy)`（`:658-679`）先看 `strategy.__dict__["get_execution_capabilities"]`（broker_live 注入），否则默认 `broker_live=False`。

## 2. 目标 / 非目标

**目标**
- 引入 `ExecutionBackend` 协议 + `SimExecution`（回测）/`BrokerExecution`（实盘）两实现；策略持 `self.execution`。
- 全部**原子**执行操作（读 + submit + cancel）走 `self.execution`。
- **组合**函数（`order_target*`/`buy_all`/`short`/`cover`/`close_position`/`stop_buy/sell`）改为只调原子操作，两模式零分支；删除 `_reject_target_orders_in_broker_live`。
- 用 `ExecutionBackend` 取代 broker_live 的 3 处 `setattr` 猴补。
- `order_target*/buy_all` 在 broker_live 按真实柜台状态 sizing 并真正下单（不再报错）。

**非目标（本阶段不做，后续）**
- P2 统一 order/trade 事件对象形状（本阶段 `on_order/on_trade` 仍各自形状）。
- P3 本地条件/止损单（broker_live 仍拒绝 trigger_price/StopTrail，语义门平移不改行为）。
- 事件溯源 `self.pos` 账本（与柜台真相二源分叉，B 方案已够，不引入）。
- CTP 平今/平昨 `OffsetConverter`（独立需求，单列后续）。

## 3. 组件设计

### 3.1 `execution/base.py` — `ExecutionBackend` 协议
`typing.Protocol`，仅原子操作（12 个）：

读：
- `get_position(symbol) -> float`
- `get_available_position(symbol) -> float`
- `get_positions() -> dict[str, float]`
- `get_open_orders(symbol=None) -> list`
- `get_order(order_id) -> Any | None`
- `get_account() -> dict[str, Any]`
- `get_portfolio_value() -> float`
- `get_cash() -> float`
- `hold_bar(symbol=None) -> int`

写：
- `submit_order(**kwargs) -> str`（签名同现 `submit_order` 自由函数）
- `cancel_order(order_id) -> None`
- `cancel_all_orders(symbol=None) -> None`

能力：
- `capabilities() -> dict[str, Any]`（取代 `get_execution_capabilities` 的注入探测）

约定：symbol 省略时由后端按当前 bar/tick 解析（`resolve_symbol` 语义，两后端一致）。

### 3.2 `execution/sim.py` — `SimExecution(strategy)`
把现有回测自由函数中读/写 `ctx` 的逻辑搬入（或委托到保留的 helper）：读走 `strategy.ctx.*`；`submit_order` 走现 `_submit_*_orders` → `ctx.buy/sell`（含 position_effect 解析、多腿 close_today/yesterday 拆分、order_type 解析等**共享规范化逻辑**）；`cancel_order/cancel_all_orders` 走 `ctx.cancel_order` + active_orders 状态置位。`capabilities()` 返回现默认 dict（`broker_live=False`，含 account_mode/supports_short_sell 从 `ctx.risk_config` 读）。

### 3.3 `gateway/broker_execution.py` — `BrokerExecution(...)`
合并现 `broker_strategy_api`（`BrokerStateCache` 支撑的状态读、`_account_to_dict` 15 键对齐、撤单路由）+ `order_submitter`（submit 语义门：sim 旋钮 warn-once、trigger_price/StopTrail 报错、broker 未就绪 guard、下单经 gateway、返回 broker_order_id）。构造参数：`trader_gateway`、`state_cache`、以及现 `BrokerOrderSubmitter` 需要的回调集合。`capabilities()` 返回 broker_live 那套（`broker_live=True`/`client_order_id`/`broker_extra_fields` 等）。缓存失效仍由事件桥在 order/trade 推送时触发（`wrap_state_invalidation` 保留，指向 `BrokerExecution` 内的 cache）。

**规范化逻辑归属**：`submit_order` 的**规范化/校验**（asset_type、client_order_id 支持性、order_type 解析、trailing 参数校验、position_effect 归一）保留在共享自由函数 `submit_order` 里；两后端只负责**放置**这份已规范化的请求（Sim→ctx.buy/sell；Broker→gateway + 语义门）。避免重复实现。

### 3.4 组合自由函数改写（`strategy_trading_api.py`）
`order_target`/`_order_target_core`/`order_target_value`/`order_target_percent`/`order_target_weights`/`order_target_positions`/`buy_all`/`short`/`cover`/`close_position`/`stop_buy`/`stop_sell`：把 `strategy.ctx.get_position(...)`/`ctx.cash`/`ctx.positions`/`get_portfolio_value` 等改为 `strategy.execution.get_position(...)`/`.get_cash()`/`.get_positions()`/`.get_portfolio_value()`；submit 改走 `strategy.execution.submit_order(...)`（或保留经 `buy()/sell()`，二者最终等价）。**删除** `_reject_target_orders_in_broker_live` 及其全部调用点。`order_target_positions` 现有的 broker_live-aware 做空逻辑改为依据 `execution.capabilities()` 判定（行为不变）。

### 3.5 绑定
- **默认（回测/paper）**：策略构造/reset 处（`ctx` 绑定同一位置）`self.execution = SimExecution(self)`。**急切绑定**——`self.execution` 始终非 None（backtrader `self.broker` 语义）。
- **broker_live**：`LiveRunner` 现安装点用 `self.execution = BrokerExecution(...)` **取代** `order_submitter.install()` 的 setattr + `install_broker_state_reads` + `install_broker_cancel`（三处猴补删除）。多 slot 时每 target 一个 `BrokerExecution`（各自 cache，事件失效全部——沿用现列表失效）。
- `get_execution_capabilities(strategy)` → `strategy.execution.capabilities()`；删掉 `__dict__` 注入探测与 `order_submitter` 对 `get_execution_capabilities` 的 setattr。
- 兼容：`strategy.submit_order()/get_position()/...` 公共方法签名与行为不变（仍 delegate → 自由函数 → `self.execution`）。`self.buy/sell` 内 `getattr(strategy,"submit_order")` 仍可用（指向类方法 → 自由函数 → execution）。

### 3.6 绑定失败处理
急切绑定要求策略能拿到自身引用即可（`SimExecution(self)` 不依赖 ctx 就绪；读 ctx 时才需 ctx，ctx 为 None 时返回 0.0/空，与现状一致）。若极端边缘用例在 `self.execution` 未绑定前就调用执行方法 → 抛清晰 `RuntimeError("execution backend 未绑定")`，不静默回退到旧 ctx 路径（保持「一套」不留分叉）。

## 4. 数据流

- 读：`strategy.get_position("600000.SH")` → 自由函数 → `strategy.execution.get_position(...)` →（Sim→`ctx.get_position`；Broker→`BrokerStateCache`）。
- 目标类：`strategy.order_target(sym, 1000)` → `_order_target_core` 读 `strategy.execution.get_position(sym)` 算 delta → `strategy.execution.submit_order(...)`。回测经 ctx 撮合，broker_live 经柜台——**同一函数体**。
- 撤单：`strategy.cancel_order(oid)` → `strategy.execution.cancel_order(oid)`（Sim→ctx；Broker→gateway）。
- 能力：`get_execution_capabilities(strategy)` → `strategy.execution.capabilities()`。

## 5. 错误处理

- broker 语义门（trigger_price/StopTrail 报错、sim 旋钮 warn-once、broker 未就绪 guard）从 `order_submitter` 平移进 `BrokerExecution.submit_order`，**行为逐字不变**。
- 柜台查询异常 → 记日志 + 返回上次缓存（`BrokerStateCache` 现状）。
- `order_target*/buy_all` 在 broker_live 从「报错」改为「按柜台 sizing 真下单」——**唯一有意的行为变化点**。
- execution 未绑定 → 清晰 `RuntimeError`（见 3.6）。

## 6. 测试策略（TDD）

- `SimExecution`：各原子 op 读/写 ctx 正确（用 fake ctx）。
- `BrokerExecution`：状态读经 cache、submit 语义门、撤单路由、capabilities——迁移/复用现 `broker_strategy_api`/`order_submitter` 用例。
- **组合函数对照测试**：同一 `order_target*`/`buy_all`/`short`/`cover` 在 (a) `SimExecution`+fake ctx 与 (b) `BrokerExecution`+fake gateway 下都给出正确 sizing 与下单调用——证「一套」。
- 迁移现有测试：依赖 setattr 猴补 / 直接注入 `get_execution_capabilities` 的用例改为 `strategy.execution = Fake/Sim/BrokerExecution`。
- 回测/paper 零回归；broker_live 新行为（目标类下单从报错→成交）更新对应断言。
- ruff `E,F,I,D` 全绿；full `tests/` 通过（golden 收尾 `git checkout`）。

## 7. 影响面

- 新增：`python/akquant/execution/__init__.py`、`base.py`、`sim.py`；`python/akquant/gateway/broker_execution.py`。
- 改：`strategy.py`（构造绑定 `self.execution`；delegator 不变）、`strategy_trading_api.py`（原子函数改走 execution、组合函数改读 execution、删守卫、`get_execution_capabilities` 改读 capabilities）、`live.py`（安装 `BrokerExecution` 取代猴补）、`gateway/broker_runtime.py`（构造 `BrokerExecution`；失效接线指向其 cache）。
- 删/退役：`gateway/order_submitter.py` 的 setattr 安装（逻辑迁入 `BrokerExecution`）、`gateway/broker_strategy_api.py` 的 `install_broker_state_reads/install_broker_cancel`（逻辑迁入 `BrokerExecution`；`_account_to_dict`/`_resolve_symbol`/`wrap_state_invalidation` 复用或平移）。
- 文档：`docs/zh/advanced/qmf_broker_gateway.md` 更新「实盘状态与撤单」——目标类下单实盘现已支持；新增执行接口说明。

## 8. 破坏性变更清单（已获授权）

1. broker_live 不再 `setattr` 猴补策略方法；改持 `self.execution`。依赖猴补断言的测试需迁移。
2. `order_target*/buy_all` 在 broker_live 不再报错——语义变化。
3. `get_execution_capabilities` 数据源从 `__dict__` 注入改为 `execution.capabilities()`。
4. 直接 new 且不进引擎的策略实例：现在也带 `SimExecution`；未绑定即调用执行方法会清晰报错而非静默走 ctx。

## 9. 待确认

1. 绑定方式：**急切绑定 SimExecution**（默认已定；理由:最「一套」、契合 backtrader、破坏性已授权）。
2. `BrokerExecution` 落位：`gateway/`（依赖 gateway 内部）——默认。
3. P1 后是否立刻进 P2/P3，还是先只交付 P1 验证一段（默认:先交付 P1）。
