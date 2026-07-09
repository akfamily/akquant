# QMF（期魔方）Broker 网关

QMF broker 通过 HTTP/WS 对接部署在服务器侧的 chibi_quant 前置机网关（恒生 T2 柜台），
使 akquant 策略能对期魔方证券账户实盘下单/撤单/查询并接收委托与成交回报。

- 仅实现 `TraderGateway`（交易）；**行情继续走 akquant 现有 feed**，本 broker 不提供行情。
- 依赖为可选组：`pip install 'akquant[qmf]'`（httpx / websocket-client / cryptography）。
- 能力矩阵：Phase 1 证券（`position_effect=False`，无开平/融券概念）。

## 快速运行

前置：chibi_quant 网关已启动（联调可用 `--mock`），客户端与网关约定同一 `CHIBI_PASSWORD_KEY`。

```python
from akquant import DataFeed
from akquant.gateway import create_gateway_bundle
from akquant.gateway.broker_models import UnifiedOrderRequest

bundle = create_gateway_bundle(
    broker="qmf",
    feed=DataFeed(),
    symbols=["600000.SH"],
    base_url="http://127.0.0.1:18080",
    ws_url="ws://127.0.0.1:18080/api/v1/stream",
    qmf_user_id="u",
    account_content="8888000001",
    password="明文交易密码",   # 客户端内部以 AES-256-GCM 加密后再上送
    input_content="1",
    content_type="1",
    password_key="<base64(32B) 共享密钥>",
)
trader = bundle.trader_gateway
trader.connect()               # 登录，获取 gateway token
trader.place_order(
    UnifiedOrderRequest(
        client_order_id="demo-1", symbol="600000.SH", side="Buy",
        quantity=100, price=10.5, order_type="Limit",
    )
)
print(trader.query_account())
print(trader.query_positions())
trader.disconnect()
```

完整示例：`examples/40_qmf_broker_live_demo.py`。

## 说明

- 下单返回柜台 `entrust_no` 作为 `broker_order_id`；适配器内部维护
  `entrust_no -> client_order_id` 反查表，用于把推送/查询结果映射回策略订单。
- 委托与成交回报经 `WebSocket /api/v1/stream` 推送；网关不缓存推送历史，断线重连后由
  `sync_open_orders` / `sync_today_trades`（HTTP 查询）补齐。
- 注册方式与其它内置 broker 一致（`create_gateway_bundle(broker="qmf", ...)`），
  另见 [自定义 Broker 注册](custom_broker_registry.md) 与
  [Broker 能力矩阵](broker_capability_matrix.md)。

## 期权（Phase 2）

同一个 `broker="qmf"` 通过**双会话**支持期权：证券会话（`asset_prop="0"`）与期权会话
（`asset_prop="B"`）。装配时传 `enable_options=True` 即额外登录期权会话并声明期权能力
（`features` 含 `"options"`、`broker_extra_fields` 含 `entrust_oc`/`covered_flag`/`entrust_prop`）。
证券路径不受影响（默认 `enable_options=False`）。

期权下单用 `asset_type="option"`，期权专属语义经 `extra` 传入：

```python
bundle = create_gateway_bundle(
    broker="qmf", feed=DataFeed(), symbols=["10003456.SH"],
    base_url="http://127.0.0.1:18080", ws_url="ws://127.0.0.1:18080/api/v1/stream",
    qmf_user_id="u", account_content="8888000001", password="明文交易密码",
    input_content="1", content_type="1", password_key="<base64(32B)>",
    enable_options=True,
)
trader = bundle.trader_gateway
trader.connect()
trader.place_order(
    UnifiedOrderRequest(
        client_order_id="opt-1", symbol="10003456.SH", side="Buy",
        quantity=1, price=0.05, order_type="Limit", asset_type="option",
        extra={"entrust_oc": "O", "covered_flag": "0", "entrust_prop": "F0"},
    )
)
```

- `entrust_oc`：`O`=开仓 / `C`=平仓 / `X`=行权（必填）；`covered_flag`：`1`=备兑 / `0`=非（默认 `0`）。
- 期权路由到 `/api/v1/option/*`；`query_positions`/`sync_*` 合并证券与期权。
- 启用期权后 `query_account` 合并证券与期权资产（`/option/assets`），详见下节。
- 完整示例：`examples/41_qmf_option_live_demo.py`。

## 只读查询（Phase 3a）

- 启用期权后（`enable_options=True`）`query_account()` 返回**合并**账户
  （证券资金 + 期权资产汇总为 `equity`/`cash`/`available_cash`）；未启用期权时仍只返回证券资金。
- `trader.query_settlements(start_date, end_date, stock_type=None)` /
  `trader.query_fund_flow(start_date=None, end_date=None)` 查询证券交割单 / 资金流水。
- `trader.query_option_history_orders/trades/settlements(start_date, end_date)`
  查询期权历史委托/成交/交割单，需先 `enable_options=True` 建立期权会话，
  否则抛出 `RuntimeError`。
- 以上方法均非 `TraderGateway` 协议方法，返回柜台**原始行** `list[dict]`（不做 Unified 建模）。

## 期权实时回报（Phase 3b）

启用期权会话（`enable_options=True`）后，`start()` 会在证券推送 WS 之外，额外建立一路
绑定**期权 token** 的推送连接。服务端按 `(account_type, fund_account)` 路由，期权回报
（issue_type 33011 成交 / 33012 委托）经该连接推送，分发到策略的 `on_trade` / `on_order`，
与证券回报路径一致。去重（按 `trade_id`）与冷启动 seed（`query_trades()` 已合并期权成交）
均自动继承，无需额外配置；未启用期权时不建立该连接。

## 期权行权 + 备兑（Phase 3c）

启用期权后，以下**便捷方法**（非 `TraderGateway` 协议，返回柜台原始行）可用；未启用期权时抛 `RuntimeError`：

- 行权查询（只读）：`query_option_exercise_assignments()`（行权指派）、
  `query_option_exercise_settlements()`（行权交割）、`query_option_exercise_debts()`（行权负债）、
  `query_option_history_exercise_assignments(start_date, end_date)` /
  `query_option_history_exercise_settlements(start_date, end_date)`（历史）。
- 备兑：`query_option_covered_shortages()`（备兑不足，读）、
  `query_option_covered_transferable(exchange_type, lock_direction, stock_code=None)`（可划转，读）、
  `covered_transfer(exchange_type, stock_code, entrust_amount, lock_direction)`（**备兑证券划转，写**；
  `lock_direction` 为 `"1"`=锁定 / `"2"`=解锁，返回柜台原始 `dict`）。

期权账户字段由服务端按会话注入，调用方无需传递。

## 期权合约 / 额度只读（Phase A）

启用期权后，以下**便捷方法**（非协议，返回柜台原始行）提供合约元数据与下单前额度：

- 合约元数据（列表）：`query_option_contracts(stock_code=None, option_code=None)`、
  `query_option_underlyings(stock_code=None)`、`query_option_strategies(optcomb_code=None)`、
  `query_option_position_limits(stock_code=None)`、`query_option_contract_tips(money_type="0")`。
- 下单前额度/提示（`dict`）：
  `query_option_enable_amount(exchange_type, option_code, opt_entrust_price, entrust_prop, entrust_bs, entrust_oc, covered_flag=None)`（可委托数量）、
  `query_option_underlying_amount_tip(exchange_type, option_code, entrust_amount, entrust_bs, entrust_oc)`（标的持仓提示）。

均需 `enable_options`，否则 `RuntimeError`；`entrust_bs` `1`买/`2`卖，`entrust_oc` `O`开/`C`平/`X`行权。

## 可转债交易（Phase F）

走**证券会话**（始终可用，无需 `enable_options`）的**便捷方法**（非协议，原始行透传）：

- `place_convertible_bond_order(stock_code, exchange_type, entrust_prop, entrust_amount, stock_account=None, stb_stock_property=None)`（下单，写，返回 `dict`）
- `cancel_convertible_bond_order(entrust_no)`（撤单，写，返回 `dict`）
- `query_convertible_bond_orders(stock_code=None, entrust_no=None, query_flag=None, en_entrust_prop=None)`（委托查询，`list`）
- `query_bond_putback_info(stock_code=None)`（回售信息查询，`list`）

client 自动注入 `fund_account`；`entrust_prop`（转股/回售等）按柜台语义由调用方传原始值。

## 实盘就绪（broker_ready）

`trading_mode="broker_live"` 下 `LiveRunner` 会先 `connect()`（登录）再 `start()`
（起 WebSocket 推送），随后轮询 `trader_gateway.heartbeat()` 直至就绪或超时
（`broker_ready_timeout`，默认 10s）；就绪状态写回策略上下文的 `broker_ready` 属性。

- 策略应以 `if ctx.broker_ready:` 门首单，而不是 `hasattr(ctx, "submit_order")`——
  `submit_order` 在 `broker_live` 模式下经 `strategy.execution`（`BrokerExecution`）路由、
  始终可调用，但 broker 未就绪前调用它会直接抛出清晰的 `RuntimeError`
  （`broker 尚未就绪，请在 broker_ready=True (on_broker_connected 之后)再下单`）。
- 就绪达成时，`LiveRunner` 会对策略与各 slot 触发 `on_broker_connected(ctx)`
  （策略方法与 `LiveRunner(on_broker_connected=...)` 函数式回调均支持）。
- `paper`/其它非 `broker_live` 模式下 `broker_ready` 默认即为 `True`，不受该守卫影响。
- 拒单与错误分别通过 `on_reject(ctx, order)` / `on_error(ctx, error)` 回调上报，
  不要依赖 `on_order` 里再判断状态字符串。
- 就绪判定基于登录（`heartbeat`）。QMF 登录完成即可下单/查询（HTTP），但推送 WS 可能
  略晚建立；就绪到 WS 建立之间的委托/成交回报由断线补齐（`sync_open_orders`/
  `sync_today_trades` 的 HTTP 补齐）兜底，不会丢。

完整示例：`examples/39_live_broker_submit_order_demo.py`。

## 执行接口 ExecutionBackend（回测/实盘同一套）

策略的状态读（`get_position`/`get_account`/…）、下单（`submit_order`）、撤单
（`cancel_order`/`cancel_all_orders`）与组合目标类下单（`order_target*`）
统一经 `strategy.execution` 这一个 `ExecutionBackend` 接口调用，不再对策略对象
做 `setattr` 猴补：

- 回测/`paper` 下 `strategy.execution` 是 `SimExecution`，转发 Rust 引擎 `ctx`，
  零回归。
- `broker_live` 下 `strategy.execution` 是 `BrokerExecution`，柜台为唯一真相。

策略代码全程只认 `self.execution`（或经既有的 `get_position()`/`submit_order()`
等便捷方法，二者最终都落到 `strategy.execution`），两种模式下写法一致，切换
`trading_mode` 无需改策略。

## 实盘状态与撤单（③b/③c）

`broker_live` 下，策略的**状态读**与**撤单**方法经 `BrokerExecution` 直接读/写
真实柜台（回测/`paper` 下走 `SimExecution`/Rust 引擎 `ctx`，零回归）。柜台是唯一真相。

- **状态读走柜台**：`get_position(symbol=None)` / `get_available_position(symbol=None)` /
  `get_account()` / `equity` / `get_open_orders(symbol=None)` 转发
  `trader_gateway.query_*`。`symbol` 省略时回退到当前 bar/tick 的标的（与回测一致）。
  应答进一个短生命缓存；柜台推送 **成交/委托**（`on_trade`/`on_order`）到达时缓存**失效**，
  下次读重新查——兼顾正确与调用量。柜台查询异常时记日志并返回上次缓存，不中断策略。
- **`get_account()` 键对齐回测**：返回与回测 `get_account()` 同形状的 dict（`cash`/
  `available_cash`/`equity`/`market_value`/…共 15 键）；柜台无法提供的键给 `0.0`/合理默认，
  以避免按回测写法的策略在实盘 `KeyError`。
- **撤单走柜台**：`cancel_order(order_id)` 直接调 `trader_gateway.cancel_order(order_id)`
  （`broker_live` 下 `submit_order` 返回的即 broker_order_id，故策略持有的 `order_id`
  本就是柜台单号）；`cancel_all_orders(symbol=None)` 遍历 `sync_open_orders()` 逐个撤，
  可按 `symbol` 过滤。
- **组合目标类下单已支持**：`order_target` / `order_target_value` /
  `order_target_percent` / `rebalance_weights` 在 `broker_live` 下现按柜台真实
  持仓/资金 sizing 下单——经统一执行接口 `ExecutionBackend`，与回测下走
  `SimExecution` 是同一套调用方式，不再报错。

## 实盘持仓同步（成交即更新）

`broker_live` 下成交事件由 `wrap_state_invalidation` 经 `BrokerStateCache.apply_fill`
**同步**叠总持仓 delta，故 `get_position`/`positions`/`self.position` 成交后**立即准**
（不再失效→异步重查而滞后）；**可用持仓**仍走柜台查询（成交后失效重查，T+1/T+0
归柜台，可能滞后一个查询）；**按 `trade_id` 去重**——恢复循环每周期
`sync_today_trades()` 会重放当日成交，`apply_fill` 是加性的，去重保证每笔只叠一次、
不因重放漂移（无 `trade_id` 的成交退回幂等 `invalidate` 重查）；账户级（多 slot
缓存都叠同一账户 delta）；**单一来源**（不新增 `self.pos`）。

此外，`BrokerEventBridge.queue_event` 在事件入队处对成交事件按 `trade_id`
**会话级去重**——恢复循环每周期重放当日成交时不重复触发
`on_trade`/`_process_order_groups`（补漏保留：断线期间漏推的新成交仍派发一次）；
无 `trade_id` 的成交无法去重、照常派发。

冷启动/盘中重启就绪激活（消双计）：`broker_ready` 就绪后（策略回调触发前）跑一次
激活——第一步先丢弃队列中**待派发**的成交事件（`discard_pending_trades()`）并把
其 `trade_id` 灌入去重基线：dispatch/recovery 线程在 ready 前已启动、回调也已
绑定，激活前可能已有实盘推送的成交排在队列里；若这些事件在 eager-seed 快照**之后**
才被 dispatch 线程 drain，会对已经 seed 过的缓存再 `apply_fill` 一次 delta，
而该成交其实已经烘进了快照——造成过计。故这些"待派发的历史成交"直接丢弃，
不叠加 `apply_fill`、也不重放给 `on_trade`（order/account 事件不受影响，照常保留）。
第二步再急切 seed 各 slot 持仓（整柜台快照，含当日已成交），最后把当时
`sync_today_trades()` 返回的 `trade_id` 灌入去重基线；此后恢复循环重放的
"基线前成交"会被去重丢弃（不叠加 `apply_fill`、也不重放给 `on_trade`），只有
激活后产生的新成交（新 `trade_id`）才会叠加 delta 并派发；恢复的成交重放本身
门控于该激活标志，激活前不跑。残留竞态窗口进一步收窄为仅"丢弃待派发成交"之后、
"seed 持仓"与"灌基线"两次查询期间**新到达**的推送（比丢弃前更小），原则上仍可能
对那一笔成交或高计或低计，直到下次重启/`invalidate()` 自愈——v1 薄款下接受的
残留限制。

对账（v1 薄款，明确边界）：冷启动整柜台 seed 为权威基线，`invalidate()` 全量重 seed
可用于对账；**会话中不自动对账**——事件流完整时事件溯源即准，但**漏接一笔成交事件**
会使总持仓漂移，直到冷启动/显式 `invalidate()` 重 seed 自愈（会话内周期性重查或
断线重连触发对账留作后续）。

## 事件模型统一（P2）

`on_order`/`on_trade` 在回测与 `broker_live` 两种模式下收到**同一属性形状**的事件对象,
策略代码不必按模式分支读字段:

- **回测**：`on_order`/`on_trade` 收原生 Rust `Order`/`Trade` 对象,未改动,零回归。
- **`broker_live`**：`Unified*`（`UnifiedOrderSnapshot`/`UnifiedTrade`）经
  `python/akquant/gateway/broker_event_adapter.py` 的 `map_order_snapshot`/`map_trade`
  适配为 `StrategyOrder`/`StrategyTrade`（dataclass）, 与回测 `Order`/`Trade` 同属性名、
  同枚举类型：
  - `status` 为 `OrderStatus` 枚举（非字符串）,`side` 为 `OrderSide` 枚举。
  - 字段名对齐：`avg_fill_price` → `average_filled_price`;`broker_order_id` → `id`;
    `trade_id` → `id`（Trade）;`broker_order_id` → `order_id`（Trade）;
    `timestamp_ns` → `timestamp`（Trade）。
  - `side`/`quantity`/`price`/`order_type`/`time_in_force` 等下单请求携带的字段,
    由**提交请求缓存**（下单时记录的 `UnifiedOrderRequest`）回填;取不到时给合理默认
    （如 `commission=0.0`、`side=None`）,不会 `AttributeError`。
  - `owner_strategy_id` 与回测 `Order`/`Trade` 一样对外暴露（`broker_live` 下由
    `LiveRunner` 按 client_order_id 解析后回填,取不到为 `None`）。
- **`on_execution_report`** 为 `broker_live` 专属回调（柜台原始回报,回测无对应事件）；
  策略基类已提供 no-op 默认实现,未定义该方法的策略在 `broker_live` 下不会崩。

同一读字段逻辑（如 `order.status is OrderStatus.Filled` 后处理）可以直接复用在
回测与实盘策略回调里,参见 `tests/test_event_model_parity.py`。

## 实盘条件/止损单（本地）

柜台不支持原生条件单，`broker_live` 下 `submit_order(order_type="StopMarket"/"StopLimit"/
"StopTrail"/"StopTrailLimit", trigger_price=.../trail_offset=...)` 现由**客户端本地止损簿**
（`python/akquant/gateway/local_stop_book.py`）盯价触发，不下发柜台：

- **触发语义与回测 Rust 原生止损一致**：买方向 `>=`、卖方向 `<=`；行情来源上，
  bar 用 `high`/`low`（`on_bar` 每根收盘后驱动一次），tick 用最新价（`on_tick`
  每笔驱动一次）；`StopTrail`/`StopTrailLimit` 追踪止损按同 bar 内先棘轮更新
  `high`/`low` 再判触发的语义（同 bar 内允许触发）。
- **触发后动作**：命中即向柜台提交底层单——限价类（`StopLimit`/`StopTrailLimit`）
  提交 `Limit`（带 `price`），其余提交 `Market`（`trigger_price=None`，已转为
  普通单，不再是条件单）。
- **可见与可撤**：挂单期间经 `get_open_orders()` 可见（与柜台真实挂单合并返回）；
  `cancel_order(本地 id)` 可撤（本地 id 形如 `LSTOP-N`，先查本地簿再回退柜台撤单）。
- **无 `on_stop_order` 回调**：与回测行为一致，本地簿本身不触发策略回调；触发后
  提交的底层单走正常下单/成交回报路径，成交经 `on_trade` 正常回调。
- **id 连续性**：本地止损触发后向柜台提交底层单，后续 `on_order`/`on_trade` 仍用
  策略持有的本地 id（`LSTOP-n`）——`BrokerExecution` 记 `broker_order_id → 本地 id`，
  `LiveRunner` 适配事件时覆盖 `id`/`order_id`（`broker_order_id`/`client_order_id`
  字段仍保留柜台真实值），策略代码不必感知底层单号的切换。
- **触发提交失败重试**：触发后底层单提交若失败（柜台未就绪/被拒），该止损单
  重新入簿、下一 tick/bar 重试（上限 3 次），每次经 `on_error(exc, "stop_trigger",
  order)` 上报；超限放弃。单次提交失败不会中断整个实盘 run。

参见 `tests/test_local_stop_scenario.py`（端到端：提交 stop → 经
`strategy_events._drive_local_stops` 喂价 → 触发提交底层单）与
`tests/test_stop_id_remap_scenario.py`（端到端：触发止损 → 记 remap →
底层单成交推送经 `map_trade(local_id=...)` 还原为本地 `LSTOP-n`）。

## 实盘 OCO/Bracket（本地协调）

`place_oco`/`place_bracket` 在 `broker_live` 与回测**共用同一套 Python 协调逻辑**
（`Strategy._process_order_groups` → `_process_pending_bracket` + `_process_oco_trade`）；
差别只在**由谁驱动**：回测由 Rust 引擎在成交时驱动，`broker_live` 由真实柜台成交经
`BrokerEventBridge._dispatch_strategy_event` 的 `trade` 分支在 `on_trade` 之后驱动
（经 `_safe_strategy_callback` 异常隔离；无组时 no-op）。

- **OCO 一腿成交撤对手**：组内任一订单成交，协调器撤销同组其余订单
  （`cancel_order` 经 `BrokerExecution`：本地 id `LSTOP-n` 走本地簿，柜台 id 走柜台）。
- **Bracket 入场成交激活**：进场单成交后自动挂出止损腿（`trigger_price=` → 本地止损簿
  `LSTOP-n`）与止盈腿（`price=` → 柜台限价单），并把两腿绑定为 OCO——止损触发成交或
  止盈成交都会撤掉对手腿。
- **止损触发也能撤止盈**：止损腿触发提交底层单后，其成交经 P4a 事件重映射仍以 `LSTOP-n`
  上报（见上节「id 连续性」），故 `_process_oco_trade` 能据此撤掉止盈对手腿。
- **并发安全**：协调器在 broker 事件派发线程运行，与引擎线程可能并发访问同一批
  OCO/bracket 状态，故 `Strategy` 的 OCO/bracket 字典由 `RLock`（`_order_group_lock`，
  不入 pickle、反序列化重建）、本地止损簿由内部 `Lock` 各自保护。
- **异常不中断 run**：协调器抛错经 `on_error(exc, "_process_order_groups", trade)` 上报，
  单笔成交的协调失败不会中断整个实盘 run。

参见 `tests/test_oco_bracket_broker_live_scenario.py`（端到端：OCO 一腿成交撤对手；
Bracket 入场成交激活止损=本地簿 + 止盈=柜台并自动 OCO）与
`tests/test_broker_bridge_drives_order_groups.py`（桥在 `trade` 后驱动协调器且异常隔离）。

## 范围与后续

组合策略（338013/14）、行权指派/交割管理、备兑划转、可交易数量(338010)、
组合/行权/交割相关历史查询、历史查询分页透传、期权独立实时 WS 订阅、Market 委托属性、
完整柜台状态集与密钥下发方案属于后续 / 待确认项。
