# P4b：OCO / Bracket 的 broker_live 本地化设计

日期：2026-07-04
状态：待评审
动机：回测里 OCO(`place_oco`)与 Bracket(`place_bracket`)是 Rust 引擎原生(peer-cancel-on-fill、entry-fill 激活止损/止盈并自动 OCO)。broker_live 下 `_engine` 未设、走 Python fallback dict(`_oco_groups`/`_oco_order_to_group`/`_pending_brackets`),但其消费者 `_process_oco_trade`/`_process_pending_bracket` 只由 `check_order_events`(读 Rust `ctx.recent_trades`)驱动;broker_live 真实成交走 `BrokerEventBridge`→直接 `on_trade`、从不经该路径 → **broker_live OCO/Bracket 实际什么都不做**(挂了组不撤、入场成交不激活)。P4b 让它工作:把既有协调器接到真实成交路径,并闭合由此暴露的并发缺口。依赖 P4a(remap 使触发止损成交的 `order_id`=`LSTOP-n`,与 OCO 组键对齐)。

## 0. 范围与边界

- **纯 Python**;Rust 不改,无 `.rs`/`.pyi` 编辑。
- **回测零回归**:协调器逻辑(`_process_oco_trade`/`_process_pending_bracket`/`place_oco`/`place_bracket`)行为不变;新增的锁在回测也只是无争用开销;协调器仅在 broker_live 由真实成交额外驱动(回测仍由 `check_order_events` 驱动,broker_live 的 `ctx.recent_trades` 对柜台单为空→不双触)。
- 复用既有 fallback 协调逻辑(不重写 OCO/bracket 语义);复用 P4a remap 与 P3 LocalStopBook/BrokerExecution。

## 1. 组件设计

### 1.1 驱动协调器(核心接线)
`BrokerEventBridge._dispatch_strategy_event`(`gateway/broker_event_bridge.py`)的 `trade` 分支,在 `on_trade` 派发后加一行:
```python
        elif event_name == "trade":
            self._safe_strategy_callback(strategy, "on_trade", payload)
            self._safe_strategy_callback(strategy, "_process_order_groups", payload)
```
- `payload` 是已适配的 `StrategyTrade`(`order_id` 经 P4a = LSTOP-n / 柜台 id;含 `symbol`/`quantity`,协调器只读这三个字段)。
- 复用 `_safe_strategy_callback`(`live.py`,`getattr(strategy, name)(payload)` + try/except→`on_error`)→ **异常隔离**:协调器内 `self.sell`/`self.cancel_order` 抛错不会打断 drain 线程。
- 仅 broker_live:bridge 只在 broker_live 存在;`_process_order_groups` 在 `_use_engine_*` 为 True 时早返回(broker_live 为 False)、无组时 dict 查空→no-op。故对每笔成交调用安全。

### 1.2 `_process_order_groups` 线程安全(OCO/bracket dict)
broker drain 线程调 `_process_order_groups`,而引擎线程上用户 `on_bar` 可调 `place_bracket`/`place_oco` → 同一批 dict 跨线程 mutation。加锁:
- `Strategy` 新增 `self._order_group_lock: threading.RLock`(`__new__`/reset 初始化;RLock 因 `_process_pending_bracket`→`place_oco` 重入)。
- **pickle 处理(重要)**:`Strategy` 会被 pickle(checkpoint/warm_start),而 `RLock` 不可 pickle。故 `__getstate__` 须剔除 `_order_group_lock`(或以 `state.pop`),`__setstate__`/`__new__` 恢复时重建;`_process_order_groups`/`place_*` 取锁前用 `getattr(self, "_order_group_lock", None)` 容错(缺失则临时建/或用 nullcontext),防旧 snapshot 恢复期竞态。
- `place_oco`、`place_bracket`、`_process_order_groups` 方法体用 `with self._order_group_lock:` 包裹对 `_oco_groups`/`_oco_order_to_group`/`_pending_brackets` 的读写。引擎-native 分支(`_use_engine_*`)不受影响。

### 1.3 `LocalStopBook` 线程安全
bracket 的止损腿由 broker drain 线程经 `_process_pending_bracket`→`self.sell(trigger_price=)`→`BrokerExecution._register_local_stop`→`LocalStopBook.register` 注册;`check_stop_triggers` 由引擎线程读/pop → 跨线程访问同一 `_orders` dict。加锁:
- `LocalStopBook` 新增内部 `threading.Lock`,`register`/`cancel`/`open_orders`/`check` 全程持锁(`check` 的迭代+pop 复合操作尤需)。纯内部、对调用方透明。

## 2. 数据流(broker_live bracket 全程)

- `place_bracket(sym, qty, entry_price, stop_trigger_price, take_profit_price)`:入场 `self.buy(price=entry_price)`→柜台限价单(柜台 id=E)`_pending_brackets[E]={...}`(持 `_order_group_lock`)。
- 入场成交:柜台推 fill(broker_order_id=E)→ drain 适配 `StrategyTrade(order_id=E)`→`on_trade`→`_process_order_groups`→`_process_pending_bracket`:pop `_pending_brackets[E]` → `self.sell(trigger_price=stop)`(→LocalStopBook,`LSTOP-1`)+`self.sell(price=take)`(→柜台限价,id=T)→`place_oco("LSTOP-1", T)`(`_oco_groups[g]={LSTOP-1,T}`)。
- 出场(任一):
  - 止盈成交:柜台推 fill(order_id=T)→`_process_oco_trade`:撤对手 `LSTOP-1`→`BrokerExecution.cancel_order`→本地簿撤。
  - 止损触发:引擎线程 `check_stop_triggers` 命中→提交底层市价单(柜台 id=S)+`record_stop_remap(LSTOP-1, S)`;柜台推 fill(order_id=S)→ 适配经 remap→`StrategyTrade(order_id="LSTOP-1")`→`_process_oco_trade`:撤对手 `T`→柜台撤。
- 纯 OCO 同理:`place_oco(a,b)`,任一成交撤另一。

## 3. 错误处理

- 协调器调用经 `_safe_strategy_callback` → 异常→`on_error`,不打断 drain。
- 撤对手单失败(柜台/本地):在协调器内 `self.cancel_order` 抛→被 `_safe_strategy_callback` 捕获→`on_error`;组已从 dict 移除(避免重复处理)。
- 锁:RLock 允许 `place_bracket`/`_process_order_groups`→`place_oco` 重入;LocalStopBook Lock 无重入需求。

## 4. 测试策略(TDD)

- **锁**:`_process_order_groups`/`place_oco`/`place_bracket` 在持锁下操作(可用可重入性 + 简单并发 smoke:两线程分别 place/process 不崩、状态一致);`LocalStopBook` 并发 register/check 不崩(GIL 下用计数/顺序断言)。
- **接线**:`BrokerEventBridge` trade 派发后调 `_process_order_groups`(fake strategy 记录调用 + payload=StrategyTrade);order/execution_report/account 分支不调。
- **OCO broker_live**:place_oco 两单,喂一单成交(StrategyTrade order_id=其一)→ 协调器撤对手(经 fake execution.cancel_order spy);验证本地止损对手(LSTOP)与柜台对手两种撤法。
- **Bracket broker_live**:place_bracket → 入场成交事件→ 提止损(LocalStopBook)+止盈(submitter)+ place_oco 组建;再喂止盈成交→撤止损(本地簿)。
- **端到端**:结合 P4a remap——止损触发→底层单成交(remap 后 order_id=LSTOP)→撤止盈对手。
- 回测零回归:既有 OCO/bracket 测试(`test_strategy_extras.py`/`test_engine.py`)全绿;full `tests/` 全绿。
- ruff `E,F,I,D`;Rust 零改动断言。

## 5. 影响面

- `python/akquant/strategy.py`:`_order_group_lock`(RLock)初始化 + `place_oco`/`place_bracket`/`_process_order_groups` 加锁。
- `python/akquant/gateway/broker_event_bridge.py`:`_dispatch_strategy_event` trade 分支加协调器调用。
- `python/akquant/gateway/local_stop_book.py`:内部 `Lock` 守 register/cancel/open_orders/check。
- 文档:`docs/zh/advanced/qmf_broker_gateway.md` 加「实盘 OCO/Bracket(本地协调)」小节。

## 6. 不做(YAGNI / 后续)

- 不重写 OCO/bracket 语义(复用既有 fallback)。
- 不做引擎-native 与 broker 协调器的并存(`_engine` 在 broker_live 恒未设,`_use_engine_*` 恒 False)。
- 不做部分成交下 bracket 的分批激活(按整笔 entry fill 激活,与回测 fallback 一致)。
- 不做协调延迟到引擎线程的方案(用锁+异常隔离在 drain 线程直接协调即可)。

## 7. 待确认

1. 协调器在 broker drain 线程直接跑(锁+异常隔离),不延迟到引擎线程——默认。
2. 复用既有 `_process_order_groups` fallback 逻辑,仅接线+加锁——默认。
3. LocalStopBook 与 OCO/bracket dict 各自加锁(而非全局单锁)——默认(各守自身状态、边界清晰)。
