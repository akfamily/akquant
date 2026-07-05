# P3：broker_live 本地条件/止损单设计

日期：2026-07-04
状态：待评审
动机：broker_live 目前对 `trigger_price`/`trail_offset`/StopTrail 直接报错（`order_submitter.py:226-232`）——条件/止损单在实盘不可用。回测里止损是 Rust 引擎原生撮合（`submit_order(order_type="StopMarket", trigger_price=X)` → `ctx.buy/sell` → 引擎按 bar high/low 或 tick price 判触发）。但 broker_live 下订单经 `BrokerExecution`→柜台、**不进 Rust 引擎**，故引擎原生止损用不上。P3 加一个**客户端本地止损簿**在 broker_live 自己盯价触发，使同一策略 API 两模式都能用条件单（vnpy 本地止损单教训）。

延续路线：①/③a/③b③c/ExecutionBackend seam(P1)/Strategy API v2/事件模型统一(P2) 均已合入 dev；本阶段 P3。

## 0. 范围与边界

- **纯 Python**：Rust 引擎（撮合、`_on_*_event` 派发、止损原生实现、talib）不改，无 `.rs`/`.pyi` 编辑、无 `cargo build`。
- **回测零回归**：回测继续用 Rust 原生止损；本地止损簿只在 broker_live 生效（`strategy_events` 钩子按 `execution` 是否有 `check_stop_triggers` 判定，SimExecution 无此方法）。
- **策略 API 不变**：`submit_order(order_type="StopMarket"/"StopLimit"/"StopTrail"/"StopTrailLimit", trigger_price=, trail_offset=, price=)` 两模式同写法；实现按模式分流。
- 覆盖条件单类型：`StopMarket`、`StopLimit`、`StopTrail`、`StopTrailLimit`（与回测一致）。

## 1. 触发语义（对齐 Rust，已核实 `src/execution/common.rs`）

- 买入止损：`price >= trigger_price` 触发；卖出止损：`price <= trigger_price` 触发。
- **Bar 模式**：买用 `bar.high >= trigger`、卖用 `bar.low <= trigger`（穿越判定）。
- **Tick 模式**：买/卖均用 `tick.price` 比较。
- **追踪（StopTrail/StopTrailLimit）**：每次价格更新先更新参考价——卖出止损跟踪 running **high**，`trigger = high - trail_offset`；买入止损跟踪 running **low**，`trigger = low + trail_offset`；再按上面判触发。首次参考价用 `trail_reference_price`（若给）否则用首个观察价。
- **触发后**：`StopMarket`/`StopTrail` → 提交 `Market` 底层单；`StopLimit`/`StopTrailLimit` → 提交 `Limit`（`price` 为限价）底层单。底层单 `trigger_price=None`（普通单）。

## 2. 组件设计

### 2.1 新增 `gateway/local_stop_book.py`
- `@dataclass LocalStopOrder`：`local_id, symbol, side(str "Buy"/"Sell"), quantity, order_type(str), trigger_price(float|None), price(float|None), trail_offset(float|None), trail_reference_price(float|None), status(str), tag, position_effect, reduce_only, kwargs(dict 透传其余下单参数)`。
- `LocalStopBook`（纯逻辑，无 IO）：
  - `register(order: LocalStopOrder) -> None`（按 local_id 存）。
  - `cancel(local_id) -> bool`（存在则删返 True）。
  - `open_orders(symbol: str | None = None) -> list[LocalStopOrder]`。
  - `check(symbol, last, high=None, low=None) -> list[LocalStopOrder]`：对该 symbol 的每个挂单：若 trailing 先更新 `trail_reference_price`/`trigger_price`（用 high/low 或 last）；再判触发（bar 用 high/low、tick high/low=None 时用 last）；触发的从簿中移除并加入返回列表。
  - 触发方向：`_is_triggered(side, trigger, last, high, low)`——buy: `(high if high is not None else last) >= trigger`；sell: `(low if low is not None else last) <= trigger`。
  - 追踪更新：sell `ref = max(ref or first, high or last); trigger = ref - offset`；buy `ref = min(ref or first, low or last); trigger = ref + offset`。

### 2.2 `BrokerExecution`（`gateway/broker_execution.py`）扩展
- 持 `self._stop_book = LocalStopBook()` 与本地 id 计数（`_next_local_stop_id() -> "LSTOP-<n>"`）。
- **`submit_order(**kwargs)`**：判定是否条件单——`kwargs.get("trigger_price") is not None` 或 `kwargs.get("trail_offset") is not None` 或 `str(kwargs.get("order_type","")).lower() in {"stop","stopmarket","stop_limit","stoplimit","stoptrail","stoptraillimit"}`。
  - 是条件单：构造 `LocalStopOrder`（含底层下单需要的 side/quantity/price/order_type/tag/position_effect/reduce_only 等），`register`，返回本地 id（**不再走 submitter、不报错**）。
  - 否则：照走 `self._submitter.submit_order(**kwargs)`（现状）。
- **`cancel_order(order_id)`**：`if self._stop_book.cancel(order_id): return`（本地撤）；否则 `self._gw.cancel_order(order_id)`（现状）。
- **`cancel_all_orders(symbol=None)`**：先撤本地簿匹配项，再走现有柜台撤单。
- **`get_open_orders(symbol=None)`**：现有柜台 open_orders + `self._stop_book.open_orders(symbol)`（本地挂单也可见，对齐回测「止损单在 open_orders 里」）。
- **新增 `check_stop_triggers(symbol, last, high=None, low=None)`**：`triggered = self._stop_book.check(symbol, last, high, low)`；对每个 triggered：换算底层单类型（stop→Market、stoplimit→Limit）后 `self._submitter.submit_order(...)`（`trigger_price=None`、`trail_offset=None`、`order_type` 换算、其余参数透传）。底层单成交经柜台事件 → `on_order`/`on_trade` 正常回调。submitter 抛错（如未就绪/被拒）由其内部 `notify_strategy_error` → `on_error` 上报，止损单已消费（移除）。

### 2.3 `strategy_events.py` 钩子（broker_live 盯价）
- `on_bar_event(strategy, bar, ctx)`：在 `strategy._check_order_events()` 之后、`call_user_callback(strategy, "on_bar", ...)` 之前，加：
  ```python
  check = getattr(getattr(strategy, "execution", None), "check_stop_triggers", None)
  if callable(check):
      check(bar.symbol, bar.close, high=bar.high, low=bar.low)
  ```
- `on_tick_event`：同处加 `check(tick.symbol, tick.price)`（high/low 缺省 None → 用 last）。
- 回测 `SimExecution` 无 `check_stop_triggers` → `getattr` 得 None → 跳过。**回测零影响**。
- 触发在用户 `on_bar` 之前完成（与回测 `cross_stop_order` 先于 `on_bar` 一致）。

## 3. 数据流

- 挂止损：`submit_order(order_type="StopMarket", side="Sell", quantity=100, trigger_price=9.5)` →（broker_live）`BrokerExecution.submit_order` → 注册 `LocalStopOrder(local_id="LSTOP-1")` → 返回 `"LSTOP-1"`。挂单期 `get_open_orders()` 含它。
- 盯价触发：每 bar/tick →（Rust 引擎驱动）`strategy_events.on_bar_event` → `execution.check_stop_triggers("600000.SH", close, high, low)` → 命中 `low <= 9.5` → 提交底层 `Market Sell 100` 到柜台 → 柜台成交推送 → `on_trade`（P2 已统一形状）。
- 撤单：`cancel_order("LSTOP-1")` → 本地簿删除。

## 4. 错误处理

- 触发后底层提交失败（柜台错误/未就绪）：submitter 现有 `notify_strategy_error` → `on_error`；止损单已消费。不重试（YAGNI；可后续加 requeue）。
- `check_stop_triggers` 对无该 symbol 挂单/空簿：no-op。
- 追踪首次参考价缺失：用首个观察价初始化。
- broker_live 下 `order_submitter` 的 trigger_price/trail 报错**保留为底层兜底**（BrokerExecution 在其上拦截，submitter 永不收到 trigger_price；直接调 submitter 传 trigger_price 仍报错=编程保护）。故其 rejection 测试不改。

## 5. 测试策略（TDD）

- `LocalStopBook` 单测：register/cancel/open_orders；触发方向（buy `>=`、sell `<=`）；bar high/low vs tick last；StopTrail 追踪参考价更新 + 触发；多挂单/多 symbol 隔离。
- `BrokerExecution` 单测：`submit_order` 条件单→注册返回本地 id（不调 submitter）、普通单→走 submitter；`cancel_order` 本地优先、否则柜台；`get_open_orders` 含本地挂单；`check_stop_triggers` 命中→用 fake submitter spy 断言提交了正确底层单（Market/Limit、trigger_price=None、side/qty 对）。
- `strategy_events` 钩子：broker_live（fake execution 带 check_stop_triggers）on_bar/on_tick 调用之；SimExecution（无该方法）不调用（回测零影响）——用 spy/计数。
- 端到端：broker_live 装配下提交 stop、喂价触发、断言底层单提交（以 fake gateway/mock）。
- 回测零回归：`test_stop_orders.py` 仍绿（Rust 原生止损不变）；full `tests/` 全绿。
- ruff `E,F,I,D`；Rust 零改动断言（`git diff -- '*.rs' *.pyi` 空）。

## 6. 影响面

- 新增 `python/akquant/gateway/local_stop_book.py`（+ 测试）。
- 改 `python/akquant/gateway/broker_execution.py`（stop book + 拦截 + cancel/open_orders + check_stop_triggers）。
- 改 `python/akquant/strategy_events.py`（on_bar/on_tick 钩子）。
- 文档：`docs/zh/advanced/qmf_broker_gateway.md` 加「实盘条件/止损单（本地）」小节。
- `order_submitter.py` 不改（保留兜底报错）。

## 7. 不做（YAGNI / 后续）

- 不加 `on_stop_order` 回调（回测无对应；保持两模式一致）。
- 不做本地 id → 柜台 id 的事件重映射（触发后底层单用柜台 id；记文档）。
- 触发提交失败不 requeue。
- 不动 get_open_orders 的跨模式形状统一（P2 未统一该形状；本地挂单以 LocalStopOrder 形态列出，带 id/symbol/side/quantity/order_type/trigger_price/status）。
- OCO/Bracket 的本地化不在此（回测已有 OCO/Bracket；broker_live 本地化后续）。

## 8. 待确认

1. 覆盖 StopMarket/StopLimit/StopTrail/StopTrailLimit 四类——默认已选（对齐回测）。
2. 触发在用户 on_bar **之前**——默认（对齐回测 cross_stop_order 顺序）。
3. 不加 on_stop_order 回调——默认（一致性）。
4. 本地 id 触发后转柜台 id、不重映射——默认（YAGNI，记文档）。
