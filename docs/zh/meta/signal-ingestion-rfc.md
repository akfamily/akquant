# RFC:外部信号平台接入

> **状态**:草案(Draft) · **日期**:2026-08-07 · **范围**:新增 `akquant.signal` 子包 + Rust 指令入口 + `RealtimeExecutionClient` 补实;**允许破坏性变更**(broker_live 下单时序与 `OrderReceipt` 语义)
>
> 对标实现:**Freqtrade**(producer/consumer WebSocket + `POST /forceenter`)、**vn.py**(WebTrader:独立进程 FastAPI + RpcClient ↔ 交易进程 RpcServer)、**nautilus_trader**(`MessageBusConfig.external_streams` + Redis backing,回调 `on_data`)、**QuantConnect LEAN**(自定义数据源拉信号)。
>
> 与 [hooks-rfc.md](hooks-rfc.md) / [timer-api-rfc.md](timer-api-rfc.md) 的关系:**严格遵守二者已确立的不变量**——不新增 `on_signal` 钩子(hooks-rfc 设计原则第 1 条字面举 `on_signal` 为反例),不新增引擎事件刻度。本 RFC 的信号**不经策略回调**,直接成为订单,故两条不变量都不受触碰。
>
> 与 [issue-329-fix-plan.md](issue-329-fix-plan.md) 的关系:信号单 `created_at` 落在任意墙钟时刻,与 bar 时间戳天然不对齐,正是 #329 第 3 环暴露的那个面。该修复已落地,本 RFC **只补一条针对信号单的回归测试**锚死它,不重开引擎缺陷。

---

## 0. 背景与动机

需求来自对接**外部量化信号平台**:信号在平台侧生成,AKQuant 只负责接收指令、风控、执行、审计。

现状是「底层能力齐备、上层入口缺失」:

- 行情侧已有跨线程通道(`DataFeed.create_live()` 的 channel),CTP 行情网关就跑在独立 daemon 线程上往里推数据;
- `broker='replay'` 证明了「自定义网关在 daemon 线程推事件」这一形态可行;
- `register_broker` + entry-point 组 `akquant.brokers` 已是成熟的插件注册机制;
- `strategy_loader` 甚至已支持动态加载策略代码。

但**指令侧没有任何外部入口**:订单只能由策略在回调内通过 `self.buy()` / `self.sell()` 产生。想接信号平台,用户只能自己起线程写队列,再在 `on_bar` 里排空——延迟取决于行情到达,标的不活跃或盘中间隙时指令会压到下一根 bar。

盘点过程中另外查出一个**既存缺口**(见 3.3):broker_live 下引擎风控不在下单路径上。它与本需求是同一处架构的两面,必须一并处理——否则外部信号将不经风控直连柜台。

## 1. 目标与非目标

**目标**

- 外部指令与策略回调下单**走完全同一条路**(同一风控、同一审计、同一持仓账本)。
- 指令注入到落地的延迟**不依赖行情到达**。
- 传输层可插拔(进程内队列 / HTTP / Redis / MQ),语义层与传输层解耦。
- 幂等:同一 `signal_id` 重复投递不重复下单。
- 危险能力**默认关闭**,网络端点默认只绑本机且强制鉴权。
- 补实 broker_live 的风控路径,使 `strategy_max_*` 在实盘首次真实生效。

**非目标**

- **不做信号生成**:因子计算、择时、组合优化都在平台侧,AKQuant 不介入。
- **不新增策略钩子**:见头部与 8.1。
- **不做信号回测**:历史信号回放用既有 `broker='replay'` 覆盖,不在本 RFC。
- **不内建平台协议适配**:只定契约与参考实现,具体平台的字段映射由用户或插件仓完成。

## 2. 设计原则

1. **单一下单路径**:任何订单来源(策略回调 / 外部信号 / 本地止损)都汇进 `Event::OrderRequest`,风控是这条路上不可绕过的一环。新增来源不得新开旁路。
2. **信号不是钩子**:外部指令直接成为订单,不为它开策略回调。用户需要干预时,干预点是**信号源**(过滤、改写、拒绝)而非策略回调。
3. **传输与语义解耦**:`Signal` 契约与 `SignalSource` 协议稳定,HTTP / Redis / MQ 都只是它的实现。
4. **危险默认关闭**:能触发真实下单的网络端点,默认 `enabled=False`、默认绑 `127.0.0.1`、缺鉴权则启动失败(不降级为警告)。照抄 Freqtrade `force_entry_enable` 的取舍。
5. **幂等优先于顺序**:信号平台重连/重推是常态。按 `signal_id` 去重,不假设投递顺序。

## 3. 现状审计

### 3.1 三条看似自然的注入路径,均不可行

| 路径 | 结论 | 硬理由 |
|---|---|---|
| 推 `Event::Timer` 进 **feed** channel | ✗ 静默丢弃 | `src/pipeline/stages/data.rs` 的 `FeedAction::Event` 分支只给 `Bar`/`Tick` 取时间戳,其余落 `_ => 0`;随后 `timestamp <= engine.snapshot_time` 必然成立 → `ProcessorResult::Loop`。`RealtimeDataClient::peek_timestamp`(`src/data/client.rs`)同样把 Timer 打成 0 |
| 新增 `on_signal` 钩子 | ✗ 撞已落地原则 | [hooks-rfc.md](hooks-rfc.md) 设计原则第 1 条:「When-not-what……`on_bar` 而非 `on_signal`」——字面把 `on_signal` 举为反例。Nautilus 的同类能力也用 `on_data` + 类型分流,不开语义钩子 |
| 外部线程直接调 `strategy.buy()` / `ctx.schedule()` | ✗ 不安全 | 前者撞 `python/akquant/strategy_trading_api.py` 的 `if strategy.ctx is None: raise RuntimeError("Context not ready")`;后者 ctx 是 pyclass,`run()` 期间 `py_ctx.borrow(py)` 有 `PyBorrowMutError` 风险 |

### 3.2 正解:`EventManager` 的 crossbeam sender

`src/event_manager.rs` 用 `crossbeam_channel::unbounded()`,`sender()` 返回的 `Sender<Event>` 是 `Send + Sync + Clone`——**天然跨线程**。策略下单本就走这条路:`src/engine/core.rs` 把它作为 `event_tx` 注入 `StrategyContext`,`src/context.rs` 里 `tx.send(Event::OrderRequest(order))`。

**注入后无需等待行情**,因为 pipeline 的编排(`src/engine/core.rs` 的 `add_processor` 序列)把 `ChannelProcessor` 排在**第一位**,`DataProcessor` 在其后:

```text
ChannelProcessor  ← 排空 event_manager
DataProcessor     ← 无行情时 FeedAction::Wait → ProcessorResult::Loop → continue 'main_loop
ExecutionProcessor / ChannelProcessor / StrategyProcessor / ...
```

`Loop` 重启主循环后**先跑 ChannelProcessor**,注入的 `OrderRequest` 即被消费 → `risk_manager.check_and_adjust` → `Event::OrderValidated` → `add_active_order`。无需伪造 tick,无需 timer 轮询。延迟上界是 `DataProcessor` 里 `wait_peek` 的 1 秒 timeout(4.1 将其降为微秒级)。

### 3.3 既存缺口:broker_live 下引擎风控不在下单路径上 ⚠️

这是盘点中查出的、**与本需求正交但必须一并修**的缺陷。

broker_live 的订单不走引擎:`_runner.py` 用 `_install_broker_order_submitter` 覆盖策略的 `submit_order`,`python/akquant/gateway/order_submitter.py` 里直接 `self._trader_gateway.place_order(request)` 送柜台。该文件**全文没有 `risk_manager` 调用**,也不经过 `ChannelProcessor`。

而 `strategy_max_order_value` / `strategy_max_order_size` / `strategy_max_position_size` / `strategy_max_daily_loss` / `strategy_max_drawdown` / `strategy_risk_budget` 这一组参数,是经 `engine.set_strategy_max_*_limits()` 下发到 **Rust `risk_manager`** 的。

**结论:这些风控参数在 `trading_mode='broker_live'` 下实际不生效。** 用户按文档配了限额,实盘却不拦单。

根因在执行器是空壳(`src/execution/realtime.rs`):

```rust
impl ExecutionClient for RealtimeExecutionClient {
    fn on_order(&mut self, _order: Order) {
        // In real impl, send to broker API      ← 函数体为空
    }
}
```

引擎在 broker_live 下 `use_realtime_execution()` 装的就是它,于是 `OrderValidated` 落到 `add_active_order` 后无人送柜台;Python 侧只能绕开引擎自己直连,风控随之被绕开。

测试没兜住:`tests/test_live_runner_broker_bridge.py` 用 `_DummyEngine` 只断言「配置被下发到 engine」,未断言**风控真的拦住了订单**。

## 4. 变更提案

### 4.1 P0:统一指令入口(**已落地**)

| 改动 | 位置 | 说明 |
|---|---|---|
| std mpsc → crossbeam | `src/data/client.rs` 的 `RealtimeDataClient`、`src/data/feed.rs` 的 `live_sender` | 换完**删掉了** `Arc<Mutex<Sender>>` 包装——它存在只因 `std::sync::mpsc::Sender` 非 `Sync`,crossbeam 的 `Sender` 本身 `Sync`。少一层锁 |
| `select!` 零延迟唤醒 | `DataClient::wait_peek_with_wakeup` + `WaitOutcome`;`DataProcessor` 传入 `EventManager::receiver()` | 用 `Select::ready_timeout` 同时探测行情通道与引擎内部事件通道。**只探测就绪、不取走**唤醒通道的事件——消费仍归 `ChannelProcessor`,否则事件会被吞。就绪即返回 `FeedAction::Wait` → 主循环 `Loop` → 回到首位的 `ChannelProcessor` |
| `SignalPort` pyclass | 新增 `src/signal_port.rs`;`Engine::signal_port(strategy_id)` 取用 | 持 `Sender<Event>` 克隆 + `owner_strategy_id`,`submit(...)` 构造 `Order` 并发 `Event::OrderRequest`。入参校验(空 symbol / 非正数量 / 未知枚举)在端口内完成 |
| `signal_port_ready` 回调 | `run_live` / `LiveRunner` | 会话启动前把端口交给调用方。P2 的 `SignalSource` 将建于其上 |

`crossbeam-channel` 0.5 已在 `Cargo.toml`,无新增依赖。

`SignalPort` 只负责构造 `Order` 并投递,**不做风控**——风控由 `ChannelProcessor` 统一执行,这是原则 1 的直接体现。已实测:注入单同样被 `strategy_max_order_value` 拦下并触发 `on_reject`。

#### 4.1.1 使用约束(实测得出,已写入 `run_live` 文档)

**① 回调内启动的线程必须确认其已就绪再返回。** `signal_port_ready` 由 runner 同步调用,一返回主线程即进入 Rust 主循环并长期持有 GIL;若新线程此刻尚未被调度,它可能整场会话都拿不到执行机会:

```python
def bind(port):
    running = threading.Event()
    def worker():
        running.set()
        ...  # port.submit(...)
    threading.Thread(target=worker, daemon=True).start()
    running.wait(timeout=5.0)   # 关键
```

**② 注入单只在"后续"市场事件到来时成交。** 落在最后一根 bar 之后的注入会停在 `New` 状态——这是撮合语义的必然(无价格无法成交),非缺陷。

**③ 不存在 GIL 饥饿。** 曾怀疑 live 会话会饿死外部线程,已实测否决:探针线程以 0.05s 周期运行 2 秒,最大间隔 0.063s。

**验收**:`tests/test_signal_port_injection.py` 5 条(外部线程注入被接受并成交、注入单仍过风控被拦、submit 即时返回、入参校验、市价单注入),连跑三轮稳定通过。

### 4.2 P1:broker_live 报单前置风控(**已落地**)

修 3.3 的缺口:让 `strategy_max_*` 在实盘真实拦单。

#### 4.2.1 被实测否决的原始设计

原方案是「让 `RealtimeExecutionClient` 持有 Python 下单桥,订单经 `Event::OrderRequest` → `ChannelProcessor` 风控 → `OrderValidated` → `on_order` 回调下单」。

**实测不可行**。报单发生在策略回调内,而策略回调是 `Engine::run(&mut self)` 从 Rust 内部回调进 Python 的——那一刻引擎对象正被独占可变借用,任何经 Python 侧触达 `Engine` 的调用都会:

```text
RuntimeError: Already borrowed
```

这条约束对 `&self` 方法同样成立(pyo3 的 `&mut` 借用独占),因此**"在策略回调内调用引擎任何方法"整体不可行**,不只是风控。这也顺带解释了为什么 broker_live 当初会绕开引擎自建下单路径。

#### 4.2.2 实际落地:无状态限额校验

把限额判定抽成**不依赖 `&Engine`** 的自由函数,两条路径共用:

```text
src/risk/strategy_limits.rs
├── exceeds_order_value / exceeds_order_size / exceeds_position_size   ← 判定逻辑单一来源
├── check_all(...)                                                     ← 串联顺序与引擎一致
└── check_strategy_limits(...)  #[pyfunction]                          ← 暴露给 Python

Engine::check_strategy_*_limit ──转发──┐
                                       ├──> 同一批自由函数
BrokerOrderSubmitter._check_risk ──────┘   (拒单文案逐字一致, 无第二套规则)
```

`BrokerOrderSubmitter` 收**限额快照**(`{"max_order_value": {"_default": 500.0}, ...}`)而非 engine 引用,由 `LiveRunner` 在装配 submitter 时传入。报单前调 `check_strategy_limits`,拒单则构造 `Rejected` 的 `StrategyOrder` 依次派发 `on_order` / `on_reject`(与回测同口径:回测的风控拒单也经 `ExecutionReport(Rejected)` 落到 `on_reject`)。

#### 4.2.3 实际影响

**零破坏性**——这是与原设计最大的差别:

- `OrderReceipt.primary` 语义**不变**(仍是柜台 id),`broker_execution.py` 的本地止损路径无需重构;
- 下单仍是**同步**的,策略代码无需改动;
- `__engine_rule_version__` **无需递增**(未改引擎行为,仅新增一个不参与回测路径的 pyfunction)。

唯一可观测变更:此前配了 `strategy_max_*` 却从未被拦的实盘策略,现在会真的被拦——这是修复而非回归。

**覆盖范围与残留缺口**:本次覆盖 `max_order_value` / `max_order_size` / `max_position_size` 三项(判定只需限额 + 委托 + 持仓快照)。`max_daily_loss` / `max_drawdown` / `strategy_risk_budget` / `reduce_only_after_risk` / `risk_cooldown_bars` 依赖引擎累计盈亏与预算用量,受同一条借用约束限制,**在 broker_live 下仍未生效**。要补齐需把这些累计量也做成可跨边界读取的快照,属独立议题。

**验收**:`tests/test_broker_live_risk_enforcement.py` 三条端到端断言(拦超限名义 / 放行限额内 / 拦超限数量),`src/risk/strategy_limits.rs` 8 条单元测试(含边界值、多空对称、文案一致性、串联顺序)。

### 4.3 P2:`akquant.signal` 模块(**已落地**,传输层实现留 P3)

```text
python/akquant/signal/
├── __init__.py        # 公开 run_live(signal_source=...) 所需符号
├── models.py          # Signal(pydantic):signal_id/symbol/action/quantity/price/
│                      #   order_type/strategy_id/timestamp/tag
├── protocols.py       # SignalSource 协议:start()/stop()/on_signal(cb),形状对齐 MarketGateway
├── dedup.py           # 按 signal_id 幂等去重
├── dispatcher.py      # 信号 → SignalPort.submit(),含审计与拒单回执
├── registry.py        # entry-point 组 akquant.signal_sources
└── sources/
    ├── queue.py       # QueueSignalSource:进程内队列(参考实现 + 测试基座)
    ├── http.py        # HttpSignalSource:FastAPI webhook(可选 extra)
    └── redis.py       # RedisSignalSource:Redis Stream 消费(可选 extra)
```

- **幂等**照抄 `python/akquant/gateway/broker_strategy_api.py` 里 `applied_fill_ids` 的思路(会话级 set + 无 id 时退回保守路径)。恢复循环重放成交已证明过这套模式的必要性。
- **审计**接 `python/akquant/gateway/order_audit.py`,记录「signal_id → client_order_id → broker_order_id」全链,可回溯。
- **拒单回执**:风控拒或柜台拒时回调信号源,让平台侧知道指令未生效。缺这一环平台会以为下成功了。
- **registry** 复用 `python/akquant/gateway/brokers/plugins.py` 的失败隔离范式(单插件失败不拖垮 import)。
- **strategy_id 路由**:`Signal.strategy_id` 接进已有的 `strategies_by_slot` / 按 strategy_id 的 `strategy_max_*` 字典,使不同信号源走各自的风控预算。

#### 4.3.1 实际落地形态

实现比原计划多一层 **`OrderSink`** 抽象,这是必需的:两种模式的下单出口完全不同(paper 走引擎注入、broker_live 走柜台通道),但幂等/审计/回执逻辑必须只有一份。

```text
python/akquant/signal/
├── models.py        Signal / SignalResult / SignalAction / SignalStatus (pydantic)
├── protocols.py     SignalSource / OrderSink 协议 + SignalSourceBase 便利基类
├── dedup.py         SignalDedup:有界 LRU, 达容量时告警(静默淘汰=静默重复下单)
├── sinks.py         PaperOrderSink(SignalPort) / BrokerOrderSink(BrokerOrderSubmitter)
├── dispatcher.py    SignalDispatcher:幂等 → 下单 → 审计 → 回执, 唯一决策点
└── sources/queue.py QueueSignalSource:参考实现 + 测试基座
```

`run_live(signal_source=...)` 托管其生命周期:`bind` → `start`(引擎循环前)→ `stop`(收尾)。

**两处与原计划的偏离**:

1. **`SignalPort` 只用于 paper**。`BrokerOrderSink` 直接走 `BrokerOrderSubmitter` —— 因为引擎的 `RealtimeExecutionClient` 不向柜台报单(见 4.2.1 的调研),经引擎注入的订单在 broker_live 下永不成交。两种模式的**风控覆盖面因此不同**,这是引擎架构事实而非配置项,已在模块 docstring 与 `run_live` 文档中写明。
2. **信号源启动失败必须中止会话**(不同于 `signal_port_ready` 交付失败只记日志)。信号源是该会话的**唯一订单来源**,静默继续会跑出一个永不下单的空会话。

**幂等的两条边界**(实现中确定,均有测试锚定):

- 投递**抛异常**时放开去重标记 —— 那一笔并未真正下单,平台重推应被受理;
- 出口**同步返回未下单**(broker 侧被前置风控拦下)时**不**放开 —— 指令已判定不可执行,重推只会再被拒一次。

**验收**:`tests/test_signal_module.py` 7 条 —— 信号到引擎并成交、同 id 投三次只下一单、拒单回执给来源、去重在 16 线程并发下原子、达容量淘汰并计数、投递失败后可重推、契约校验(含 `action` 字符串归一)。连跑两轮稳定。

调用形态与 `run_backtest` / `run_live` 保持对称,新增一个参数:

```python
from akquant import run_live
from akquant.signal import HttpSignalSource

run_live(
    instruments=[...],
    broker="qmf",
    trading_mode="broker_live",
    signal_source=HttpSignalSource(
        host="127.0.0.1", port=8765, token=os.environ["AKQUANT_SIGNAL_TOKEN"]
    ),
    strategy_cls=None,        # 纯信号驱动:不需要策略
)
```

`strategy_cls=None` 且 `signal_source` 存在时为**纯信号驱动**会话:引擎照常跑行情、持仓、结算、风控,只是订单来源全部是外部指令。这也是 `strategy_cls` 首次允许为 None——目前 `run_live` 会在无策略时无事可做。

信号源与策略**可以共存**:策略产自有信号,平台推补充指令,两者经同一风控汇入同一账本。

### 4.3.2 P3 实际落地(传输层与安全)

```text
python/akquant/signal/
├── security.py             TokenAuth / sign / AuthError:鉴权与防重放
└── sources/
    ├── http.py             HttpSignalSource —— 标准库 http.server
    └── redis_stream.py     RedisSignalSource —— Redis Stream 消费组
```

**与原计划的偏离:HTTP 源不用 FastAPI,改标准库 `http.server`。**

理由是可验证性:这个端点能触发真实下单,安全逻辑最需要在 CI 里被真的测到——起服务、发请求、断言拒绝。而 FastAPI/uvicorn 是可选依赖,本机与 CI 环境未必安装(**实测本机三者全缺**),届时只能靠 mock 覆盖,等于把最关键的一环留在测试之外。标准库对"接一个 webhook"这个单一职责完全够用。高吞吐/复杂路由场景走 4.5 的部署建议(独立 Web 进程 + Redis)。

**Redis 源用 Stream 而非 List**:`XREADGROUP` 提供消费组与显式 ack,崩溃重启后未 ack 的消息仍在 pending 中可被重投;`BLPOP` 取走即丢,崩在处理中途就丢单。坏消息也 ack——`dispatch` 内部已按 `signal_id` 幂等,不 ack 会让它永远堵在 pending 里反复重投。

**验证边界(必须如实说明)**:

- HTTP 源:**真实端到端**——起真实服务、用 httpx 发真实请求,覆盖正常受理/坏 token/缺签名/坏负载/错路径/并发重复。
- Redis 源:注入 fake client 覆盖消费循环(XREADGROUP → dispatch → XACK)、坏消息 ack、BUSYGROUP 容错。**真实 Redis 连接路径未验证**(环境未装 `redis` 包)。
- `redis` 经 `signal-redis` extra 提供,惰性导入,未安装时只在实际使用时报错,不影响 `import akquant`。

### 4.4 安全基线(硬约束,非建议)

`HttpSignalSource` 是能触发真实下单的网络入口。以下为实现层面的硬约束:

| 约束 | 落地行为 | 状态 |
|---|---|---|
| 默认关闭 | 不传 `signal_source` 即零监听。**未做** `enabled=False` 开关——它与"强制鉴权"重复:没有 token 就构造不出实例,再加一个布尔开关只是多一处可被误配的地方 | ✅(以更强形式) |
| 默认本机 | `host` 默认 `127.0.0.1`;绑非本机**直接 `ValueError`**,须显式 `allow_remote=True` 才允许,且打警告。比原设计的"仅警告"更严 | ✅(更严) |
| 强制鉴权 | `token` 非空字符串,否则构造即 `ValueError`。**须显式挡 `None`**:走 `str(None)` 会得到字面量 `"None"`,那就成了看似有鉴权的端点——从缺失的环境变量读 token 正是这个场景(测试期间实际踩到并修) | ✅ |
| 防重放 | HMAC-SHA256(`secret`,`{ts}.{body}`)+ 时间戳窗口(默认 ±30s)+ `signal_id` 幂等,三重。`compare_digest` 常量时间比较,避免按字节泄漏 | ✅ |
| 不做区分预言机 | 鉴权失败统一回 `401 {"error":"unauthorized"}`,不区分"token 错"/"签名错"/"超窗",具体原因只进日志 | ✅ |
| 幂等重复回 200 | 重复投递回 `200 + duplicate` 而非错误码——回错误码会让平台一直重试 | ✅ |
| 传输层不承诺 | HTTPS 与公网暴露须在反向代理层解决,文档与构造警告中均写明 | ✅ |

对标依据:Freqtrade 的 `force_entry_enable` 默认关闭、JWT Bearer 鉴权、默认只听 localhost,且官方文档明确警告不要暴露到公网、不原生支持 HTTPS。本 RFC 采同一姿态,并在文档中同样明确:**HTTPS / 公网暴露由用户在反向代理层解决,AKQuant 不承诺传输层安全**。

### 4.5 部署形态建议

vn.py 的 WebTrader 把 Web 服务放在**独立进程**(FastAPI + RpcClient ↔ 交易进程 RpcServer),交易主循环与 HTTP 服务之间有进程边界。这个取舍值得沿用。

因此文档应推荐:

- **生产**:外部进程收 HTTP/WS,只把规范化后的 `Signal` 投进 Redis Stream,交易进程用 `RedisSignalSource` 消费。uvicorn 不进交易进程,故障与负载都被进程边界隔开。
- **开发/单机**:`HttpSignalSource` 同进程直收,省一个组件。
- **测试**:`QueueSignalSource`,无网络依赖。

### 4.6 顺带查出的既存缺陷:`duration` 在无行情时不生效(**已修**)

`LiveRunner._apply_time_limit` 是通过 patch 策略的 `on_bar` / `on_tick` 实现的——墙钟检查只在**有行情事件时**才执行:

```python
def wrapped_on_bar(bar):
    if time.time() - start_time > duration_sec:
        raise KeyboardInterrupt(...)
    original_on_bar(bar)
```

于是行情一停(网关线程退出、或盘后无推送),引擎永远阻塞在 `wait_peek` 上,`on_bar` 不再被调用,`duration` 便永不触发,**会话挂死**。

这与 `_runner.py` 里"`duration` 仍作安全网:……墙钟兜底避免挂死"的注释所承诺的行为不符——它不是墙钟兜底。

实测触发路径:自定义行情网关推完最后一根 bar 即退出、且未声明 `bounded_event_total` → 会话永久挂起。`replay` 之所以不挂,是因为它声明了 `bounded_event_total`,由 `_apply_bounded_event_limit` 在最后一根 bar 处终止(同样挂在 `on_bar` 上,只是恰好能触发)。

**修法(已落地)**:把截止时刻下沉到等待循环本身,不再依赖事件到达。

- Rust:`Engine.session_deadline_ns` 字段 + `set_session_deadline_ns(Option<i64>)` pymethod(非正值视为不限);`DataProcessor::process` 在 **feed 等待之前**比对墙钟,到点即 `finalize_current_timestamp` 后 `ProcessorResult::Break`。放在等待之前是必要的——等待本身可长达 1 秒,而行情停摆时 `Wait` 会无限循环。
- Python:`_apply_time_limit` 改为设置引擎时限;**保留**原有的回调 patch 作为互补(有行情时能更早在事件边界处停下),故无行为回退。引擎不支持该方法时打警告降级。

**验证**:`tests/test_live_duration_wallclock.py` —— 自定义网关推一根 bar 即退出、刻意不声明 `bounded_event_total`(否则会掩盖缺陷),断言会话在时限附近结束。测试自带 `join(timeout)`,缺陷复现时表现为断言失败而非套件卡死。

**已确认该测试能抓到缺陷**:临时屏蔽 `set_session_deadline_ns` 后,会话 12 秒内不结束(挂死);启用后 3 秒时限正常退出。

`bounded_event_total` 仍挂在 `on_bar` 上,**未改**——它的语义就是"数行情事件",在行情事件上计数是正确的;`duration` 现在是它的真正兜底。

## 5. 破坏性变更清单(供 CHANGELOG)

**P1(已落地)**:

1. **行为变更** `strategy_max_order_value` / `max_order_size` / `max_position_size` 在 `trading_mode='broker_live'` 下开始真实拦单(修复 3.3)。此前这些限额在实盘静默失效,依赖旧行为(即"配了也不拦")的部署会看到订单被拒。
2. **新增** Rust pyfunction `check_strategy_limits`(模块级,非 Engine 方法)。
3. 无 API 破坏:`OrderReceipt` 语义、下单同步性、`__engine_rule_version__` 均不变(理由见 4.2.3)。

**P0 / P2 及以后(待实施)**:

4. **新增** `run_live(signal_source=...)`;`strategy_cls` 允许为 `None`(纯信号驱动)。
5. 内部:`RealtimeDataClient` / `DataFeed.live_sender` 由 std mpsc 改为 crossbeam,`Arc<Mutex<Sender>>` 包装移除。
6. `SignalPort` 走 `Event::OrderRequest` 注入时,其订单**经** `ChannelProcessor` 的完整风控(与策略回调下单不同——后者受 4.2.1 的借用约束,只能走前置快照校验)。两条路径的风控覆盖面因此不同,需在文档中写明。

## 6. 分期实施

| 阶段 | 内容 | 状态 |
|---|---|---|
| **P1** | broker_live 报单前置风控 + 拦单回归测试(修 3.3) | ✅ **已落地** |
| **P0** | crossbeam 统一 + `select!` 唤醒 + `SignalPort` + `signal_port_ready` | ✅ **已落地** |
| **P2** | `Signal` / `SignalSource` / `OrderSink` / `dedup` / `dispatcher` / `QueueSignalSource` + `run_live` 接线 | ✅ **已落地** |
| **P3** | `HttpSignalSource` + `RedisSignalSource` + 安全基线 + 可选 extra + 示例 | ✅ **已落地** |
| **P4** | 中英文档章节 + 教材映射表 + entry-point 插件化 | 待实施 |

P1 先行的判断已被验证正确:3.3 是正在影响实盘用户的风控失效,紧急度高于信号接入本身;且它暴露的借用约束(4.2.1)直接改写了 P0 的设计前提——`SignalPort` 必须在**引擎线程之外**注入 `Event::OrderRequest`(crossbeam sender 可跨线程,不碰 pyclass 借用),不能设计成"策略回调内调用"的形态。

P0 与 P2 的依赖关系不变(P2 需要 `SignalPort`)。

## 7. 验收 / 测试

**P0**

- Rust 单测:外部线程持 `Sender<Event>` 注入 `OrderRequest`,断言 `ChannelProcessor` 在**无任何行情事件**的情况下消费到它并落进 `active_orders`。
- 延迟测试:注入到落地的墙钟耗时,断言 `< 50ms`(`select!` 生效的证据;未生效时会是 ~1s)。

**P1(已完成,实跑结果)**

- `tests/test_broker_live_risk_enforcement.py` —— 端到端 `broker_live` 会话(`market_broker="replay"` + 桩 trader gateway),三条断言:超限名义不到柜台且触发 `on_reject`、限额内正常报单(防误杀)、超限数量同样拦下。**修复前第一条为 RED**(`strict=True` 的 xfail),修复后转 PASS。
- `src/risk/strategy_limits.rs` —— 8 条单元测试:边界值(恰好等于上限不拒)、无参考价时跳过名义校验、多空绝对值对称、减仓放行、拒单文案带符号、`check_all` 串联顺序与引擎一致、无限额时放行、`Sell` 映射为负 delta。
- 全量实跑:**1512 条 Python 测试通过**(含 golden 基线未变)、**141 条 Rust 测试通过**、`ruff check` + `ruff format --check` 通过、`clippy` 对新代码零警告。
- 未做:`tests/test_live_runner_broker_bridge.py` 那组 `_DummyEngine`「只验证配置下发」的断言仍保留(它们本身没错,只是不足);新增文件已覆盖真实拦截,故不改动既有测试。

**P2 / P3**

- 幂等:同一 `signal_id` 投递 3 次,断言只报单 1 次。
- **信号单终态化**(#329 回归锚):对停牌/无行情标的注入信号,断言订单在其可撮合切片过后进入终态(`Rejected`),不会逐日累积挤占保证金。参见 [issue-329-fix-plan.md](issue-329-fix-plan.md)。
- 鉴权:无 token 启动断言 `ValueError`;错误签名断言 401 且不产生订单;超窗时间戳断言被拒。
- 拒单回执:风控拒单后断言信号源收到回执。
- HTTP/Redis 源用 mock 传输测试(不依赖真实服务),与两个 plugin 仓的 `httpx MockTransport` 范式一致。

## 8. 明确不做的方向(记录否决理由)

### 8.1 新增 `on_signal` 策略钩子

最直觉的设计,但撞 [hooks-rfc.md](hooks-rfc.md) 设计原则第 1 条,该原则**字面**把 `on_signal` 举为反例(「`on_bar` 而非 `on_signal`」)。外部佐证:nautilus_trader 的同类能力(外部流 / 自定义数据)落地回调是 `on_data`,靠 `isinstance` 类型分流,同样不为语义开钩子。

更根本的是:信号已经是**指令**而不是**事件**,它不需要策略再做一次决策。让它经过策略回调是多余的一跳,还会把「外部指令」与「自有信号」的责任边界搅混。**否决。**

### 8.2 把信号包装成 Timer 或伪 Tick 推进 feed channel

Timer 会被静默丢弃(3.1)。伪 Tick 能唤醒引擎,但会污染 `last_prices`、指标计算与 bar 聚合——用假行情驱动真实下单,是不可接受的语义污染。**否决。**

### 8.3 Python 侧队列 + timer 自持轮询

零 Rust 改动即可实现:信号源写 `queue.Queue`,策略在 `on_timer` 里排空并自排下一枚 timer(`ctx.schedule` 写 `timers_arc`,回调返回后引擎收走)。延迟 ≤~1s。

在**禁止改 Rust**的前提下这是最优解,但本 RFC 允许改 Rust,则它的三个缺点无法接受:① 延迟比 4.1 高三个数量级;② 每秒一枚 timer 会持续触发 `finalize_current_timestamp`,与 #329 家族的时序修复面持续摩擦;③ 信号必须经策略回调,违反 8.1 的结论。**降级为「不改 Rust 时的临时方案」,不作为目标形态。**

### 8.4 在引擎外自建风控层

即「外部信号自己做一遍限额检查再直连柜台」。这会产生两套风控实现、两份持仓视图,且把 3.3 的缺口固化为设计。**否决——风控必须是下单路径上不可绕过的一环(原则 1)。**
