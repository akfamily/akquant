# Broker 连接/就绪生命周期设计（实盘体验 ①）

日期：2026-07-03
状态：待评审
动机：让策略「方便做实盘」——去掉 `hasattr(ctx, "submit_order")` 土办法，给出明确的「broker 已就绪、可下单」信号，并修复 LiveRunner 不调 `connect()` 导致的登录缺口。

## 1. 目标

1. **修复 connect 缺口**：LiveRunner 目前只调 `trader_gateway.start()`、**从不调 `connect()`**。CTP 的 `start()` 内部驱动完整鉴权（SPI），故 CTP 能用；但 QMF 把登录放在 `connect()`、WS 放在 `start()`——真实 LiveRunner broker_live 下 QMF 会用空 token 起 WS、**永不登录**，`place_order` 因缺 bearer 失败。示例 40/41 仅因手动调 `connect()` 才能跑。
2. **就绪信号**：`ctx.broker_ready`（bool）+ 可选 `on_broker_connected(ctx)` 回调，在交易网关确认可交易后触发一次。
3. **`submit_order` 恒在**：函数式 ctx 在首个 bar 前即绑定 `submit_order`；未就绪时调用给**清晰错误**（非 `AttributeError`）。策略从此写 `if ctx.broker_ready:` 而非 `hasattr`。

均为向后兼容的纯增益；`paper`/回测行为不变。

## 2. 已确认决策

- **connect 修复 = A：LiveRunner 契约化 `connect()→start()`**。协议契约：`connect()` 建立会话（QMF：HTTP 登录；幂等、失败即抛，fail-fast），`start()` 启动流式（QMF：WS；CTP：异步 SPI 登录）。
- **就绪判定用 `heartbeat()` 轮询**（协议已有 `heartbeat()`）：`connect()→start()` 后，LiveRunner 轮询 `heartbeat()` 直到 True 或超时，再置 `broker_ready` 并触发 `on_broker_connected`。理由：CTP 登录是异步 SPI，`connect()` 返回≠可交易；`heartbeat()` 是两者通用的「真正可交易」探针（QMF：`auth_status`；CTP：`ready_to_trade`）。

## 3. 现状（已核实）

- `live.py` run() 序列：起行情/交易网关线程（仅 `start()`）→ `time.sleep(2.0)`（魔法等待）→ 建策略 → 绑回调 → 装 `submit_order` → 跑。**无 `connect()`、无真正就绪确认。**
- `TraderGateway` 协议已含 `connect()`、`start()`、`heartbeat()`。CTP native 有 `ready_to_trade`（`heartbeat()` 应据此）；QMF `heartbeat()=auth_status(keepalive)`（登录后为 True）。
- `FunctionalStrategy(Strategy)`：`submit_order` 是方法，但函数式 ctx 的注入时机导致示例需 `hasattr` 守卫。
- LiveRunner 已有 `on_order/on_trade/on_reject/on_error` 回调参数与分发机制。

## 4. 组件设计

### 4.1 `live.py`（核心）——连接契约 + 就绪门
- run() 中，交易网关启动改为：**先同步 `trader_gateway.connect()`（fail-fast，登录失败立即抛并中止）**，再 `_start_gateway_thread(trader_gateway.start, ...)`。
- 用**就绪门**替换裸 `time.sleep(2.0)`：`start()` 后轮询 `trader_gateway.heartbeat()`，间隔 ~0.2s，上限 `broker_ready_timeout`（新 gateway_option / 参数，默认 10s）。
  - 就绪（heartbeat True）→ `self._broker_ready = True`；给每个策略实例置 `broker_ready = True`；分发 `on_broker_connected(ctx)`（若提供）。
  - 超时未就绪 → 记 WARNING 并置 `broker_ready = False`；`on_broker_connected` 不触发（策略据 `broker_ready` 自行跳过下单）。**不**强制抛（避免瞬时抖动即崩），可由 `broker_ready_required=True`（默认 False）选择改为抛。
- `paper`/回测（无 `broker_live`）：`broker_ready` 默认 **True**（模拟撮合始终可下单），不涉及网关就绪。

### 4.2 `strategy` / ctx —— `submit_order` 恒在 + `broker_ready`
- 策略实例/ctx 在 setup（run 循环前）即具备 `submit_order` 与 `broker_ready`（默认 False，`broker_live` 未就绪时；`paper`/回测为 True）。函数式与类式一致。
- `broker_live` 下 `broker_ready=False` 时调用 `submit_order` → 抛清晰 `RuntimeError("broker 尚未就绪，请在 broker_ready=True 后下单")`（非 `AttributeError`），也可被 `on_error` 捕获。

### 4.3 `live.py` 新增回调参数
- `on_broker_connected: Optional[Callable[[Any], None]] = None`（函数式）/ 类式策略可实现 `on_broker_connected(self)` 方法。与现有 `on_order` 等一致地存储与分发。

### 4.4 各 adapter 契约核对
- **QMF**：`connect()` 已做登录、`start()` 用 `self._client.token`——契约 A 下 connect 先行，token 就绪，WS 正常。`heartbeat()` 已可用。**无需改 QMF 代码**（除非验证发现问题）。
- **CTP**：确认 `connect()` **幂等/安全**（LiveRunner 现在会先调它再 `start()`）；`heartbeat()`/`ready_to_trade` 反映真正可交易。若 CTP `connect()` 与 `start()` 有重复动作，需保证幂等（本阶段核对，必要时最小修正）。
- miniqmt/ptrade stub：`connect()`/`heartbeat()` 为简单实现，契约下正常。

## 5. 数据流（broker_live 启动）

`LiveRunner.run()` → `trader_gateway.connect()`（登录，fail-fast）→ 线程 `trader_gateway.start()`（WS/SPI）→ 轮询 `heartbeat()` ≤ `broker_ready_timeout` → True → `broker_ready=True` + `on_broker_connected(ctx)` → 首个 bar：`if ctx.broker_ready: ctx.submit_order(...)`。

## 6. 错误处理

- `connect()` 抛（登录失败）→ run() 中止并上抛（fail-fast，实盘不带病启动）。
- 就绪超时 → WARNING + `broker_ready=False`；`broker_ready_required=True` 时改为抛。
- 未就绪下单 → 清晰 `RuntimeError`（可被 `on_error` 捕获）。

## 7. 测试策略（TDD）

- LiveRunner 就绪门（用 fake trader gateway：connect 记录调用、heartbeat 先 False 后 True）：`connect()` 在 `start()` 前调用；heartbeat 转 True 后 `broker_ready` 置真且 `on_broker_connected` 触发一次；heartbeat 恒 False → 超时后 `broker_ready=False`、未触发回调。
- `submit_order` 恒在：`broker_live` 未就绪调用抛清晰 `RuntimeError`（非 AttributeError）；就绪后正常路由。
- `paper` 模式：`broker_ready` 默认 True，行为不变。
- QMF：`connect()` 先行使 `start()` 的 WS token 非空（用 fake client 验证调用顺序）。
- 现有 gateway/live_runner 全套无回归；ruff `E,F,I,D`；示例更新后实跑。

## 8. 不做（YAGNI / 冻结）

- 不重构 `submit_order` 的 monkey-patch 注入机制本身（只保证「恒在 + 就绪清晰报错」）。
- 不做②可重试重发、④行情装配、⑤统一账户读——各自独立后续。
- **不做 ③「submit_order 回测/实盘签名一致」——紧接本阶段做为独立一块。** 已核实的差异（供 ③ 用）：
  回测 `Strategy.submit_order` 独有、broker_live `BrokerOrderSubmitter.submit_order` 缺失的参数
  = `broker_options / trail_offset / trail_reference_price / fill_policy / slippage / commission`
  （回测策略传这些 → 实盘 `TypeError`）；且 `trigger_price`、`tag` 在 broker 版被 `_ = ...` **静默丢弃**
  （回测的止损单实盘不触发且不报错——③ 须改为明确报错或真做条件单）。本阶段 ① 不动这些。
- 不改 `paper`/回测的既有语义。

## 9. 影响面（预计改动文件）

- `python/akquant/live.py`（connect 契约、就绪门、`on_broker_connected`、`broker_ready`、超时参数）
- `python/akquant/gateway/order_submitter.py` 或 strategy 侧（`submit_order` 恒在 + 未就绪清晰报错、`broker_ready` 属性）
- `python/akquant/gateway/protocols.py`（文档化 connect/start/heartbeat 契约注释）
- CTP adapter（核对 `connect()` 幂等；必要时最小修正）
- 示例 `examples/39/40/41`、`docs/zh/advanced/qmf_broker_gateway.md`、live 文档（用 `broker_ready` 替代 `hasattr`、演示 `on_reject`）

## 10. 待确认

1. connect 修复 = A（已选）；就绪判定用 `heartbeat()` 轮询（默认已采，因 CTP 异步登录需要）。
2. `broker_ready_timeout` 默认值（暂定 10s）与 `broker_ready_required` 默认 False（超时不崩，仅告警）。
3. 未就绪下单是「抛清晰错误」还是「静默丢弃 + on_error」——暂定**抛清晰 `RuntimeError`**（也走 on_error）。
