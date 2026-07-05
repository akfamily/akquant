# P4a：broker_live 本地止损单 id 连续性(remap) + 触发提交失败 requeue 设计

日期：2026-07-04
状态：待评审
动机：P3 的 broker_live 本地止损单有两个缺口:(1)止损触发后 `BrokerExecution.check_stop_triggers` 向柜台提交底层单、拿到新的 `broker_order_id`,但策略持有的是本地 id `LSTOP-n`——后续 `on_order`/`on_trade` 用柜台 id,策略无法把回报关联回它下的止损单(与回测「止损单触发后仍同 id」不一致);(2)触发提交若抛错(柜台未就绪/被拒)**当前无捕获,一路上抛到 `engine.run` → 整个实盘 run 终止**(严重),且触发单已从簿中移除→永久丢失。P4a 修这两点。是 P4b(OCO/Bracket broker_live 本地化)的前置(OCO 按 order_id 匹配 peer,本地止损单需 remap 后才能匹配)。

## 0. 范围与边界

- **纯 Python**;Rust 引擎不改,无 `.rs`/`.pyi` 编辑。
- **回测零回归**:remap/requeue 只在 broker_live 的 `check_stop_triggers`/事件适配路径;回测 `SimExecution` 无 `check_stop_triggers`,adapter 的 `local_id` 缺省 None→行为不变。
- 复用 P2 的适配注入点(`_adapt_strategy_payload` + `map_order_snapshot`/`map_trade`)与 P3 的 `LocalStopBook`/`BrokerExecution`。

## 1. C — 触发提交失败:不崩 + requeue

`BrokerExecution.check_stop_triggers` 循环内对 `self._submitter.submit_order(**kwargs)` 加 try/except:
- **成功**:拿 `broker_order_id` 返回值,做 remap(见 §2)。
- **失败**:`order.submit_attempts += 1`;若 `< MAX_STOP_SUBMIT_ATTEMPTS`(常量=3)→ `self._stop_book.register(order)` **重新入簿**(下 tick 重试);否则**放弃**(不再入簿)。两种情况都经 `self._s.on_error(exc, "stop_trigger", order)`(guarded,on_error 缺失/自身抛错不传播)上报。
- **绝不让异常传播出 `check_stop_triggers`**(避免终止整个 run)。
- `LocalStopOrder` 加字段 `submit_attempts: int = 0`。
- 重入簿的止损单下 tick 会被 `LocalStopBook.check` 重新判触发(trigger_price 已在触发时确定/追踪已棘轮),价格仍越过则再触发重试;柜台恢复即成功。

## 2. B — 本地 id → 柜台 id 连续性(remap)

目标:策略持 `LSTOP-n`,触发后该单的 `on_order`/`on_trade` 仍报 `id`/`order_id`=`LSTOP-n`。

- **捕获 + 记录**:`check_stop_triggers` 成功提交后 `bid = self._submitter.submit_order(**kwargs)`;若 `self._record_stop_remap` 可用,调 `self._record_stop_remap(order.local_id, bid)`。
- **注入**:`BrokerExecution.__init__` 加参 `record_stop_remap: Callable[[str, str], None] | None = None`;`BrokerRuntime.__init__` 加参 `record_stop_remap` 并在 `install_submitter` 构造 `BrokerExecution(...)` 时透传;`LiveRunner` 把 `self._record_stop_remap` 注入 `BrokerRuntime`。
- **LiveRunner 状态**:`self._broker_to_local_stop_id: dict[str, str] = {}`(broker_order_id → LSTOP-n),在 broker-bridge 状态初始化处(同 `_order_requests`/`_client_to_broker_order_ids`)建立。
  - `_record_stop_remap(local_id, broker_order_id)`:`self._broker_to_local_stop_id[str(broker_order_id)] = str(local_id)`。
  - `_lookup_stop_local_id(payload) -> str | None`:用 `self._payload_field(payload, "broker_order_id")` 解出 bid → 返回 `self._broker_to_local_stop_id.get(str(bid))`。
  - 终态清理:现有 `_close_order_mapping`(终态 pop `_order_requests`/id 映射处)一并 `self._broker_to_local_stop_id.pop(str(broker_order_id), None)`。
- **适配**:`_adapt_strategy_payload` 对 order/trade 先 `local_id = self._lookup_stop_local_id(payload)`,把 `local_id=local_id` 传给 `map_order_snapshot`/`map_trade`。
- **adapter**:`map_order_snapshot(snapshot, request=None, owner_strategy_id=None, local_id=None)`:`id = local_id or broker_order_id`;`map_trade(..., local_id=None)`:`order_id = local_id or broker_order_id`。`broker_order_id`/`client_order_id` 字段保留原值(供参考)。`local_id=None`(普通单/回测无此)→ 行为不变。

顺序保证:P2 的修复已让适配发生在 `drain_events` 中、`_update_broker_state`(清理)之前——故终态事件的适配仍能查到 remap(与 request 回填同)。

## 3. 数据流

- 触发提交:`check_stop_triggers` → `submitter.submit_order` 返回 `broker_order_id`=B9;`record_stop_remap("LSTOP-1", "B9")` → LiveRunner `_broker_to_local_stop_id["B9"]="LSTOP-1"`。
- 成交回报:柜台推 `UnifiedTrade(broker_order_id="B9", ...)` → `drain_events` → `_adapt_strategy_payload("trade", payload)`:`local_id=_lookup_stop_local_id(payload)="LSTOP-1"` → `map_trade(payload, local_id="LSTOP-1")` → `StrategyTrade(order_id="LSTOP-1", ...)`。策略 on_trade 看到自己下的止损单 id。
- 失败重试:`submitter.submit_order` 抛(未就绪)→ `attempts=1<3` → 重入簿 + `on_error`;下 tick 再触发重试;3 次仍失败→放弃 + `on_error`。

## 4. 错误处理

- `check_stop_triggers` 捕获所有 submit 异常,不传播;经 `on_error` 上报;超限放弃。
- `on_error` 调用 guarded(缺失/抛错不影响其它止损单处理)。
- remap 未命中(普通单、非触发单)→ `local_id=None` → id 用 broker_order_id(原行为)。
- remap 清理:仅在底层单终态清理时 pop,避免泄漏;未终态前保留(供多次 on_order 部分成交都报 local id)。

## 5. 测试策略(TDD)

- requeue:fake submitter 抛错 → `check_stop_triggers` 不抛、止损单仍在簿(attempts=1)、on_error 被调;连续 3 次后止损单不在簿(放弃)、on_error 调 3 次。成功路径:submitter 返回 bid → record_stop_remap 被调 `(local_id, bid)`。
- adapter:`map_order_snapshot(..., local_id="LSTOP-1").id == "LSTOP-1"`;`map_trade(..., local_id="LSTOP-1").order_id == "LSTOP-1"`;`local_id=None` → id=broker_order_id(不变)。
- LiveRunner:`_record_stop_remap` 存;`_lookup_stop_local_id` 用 broker_order_id 解;`_adapt_strategy_payload` 命中→id 为 local;终态清理 pop 不泄漏。
- 端到端:broker_live 触发一个止损 → 底层单成交推送 → 策略 on_trade 收到的 order_id 是 LSTOP-n。
- 回测零回归:full `tests/` 全绿;`map_*` 的 local_id 缺省 None 使既有 on_order/on_trade 适配不变。
- ruff `E,F,I,D`;Rust 零改动断言。

## 6. 影响面

- `python/akquant/gateway/local_stop_book.py`:`LocalStopOrder` 加 `submit_attempts`。
- `python/akquant/gateway/broker_execution.py`:`check_stop_triggers` try/except+requeue+remap 记录;`__init__` 加 `record_stop_remap` 参数;`MAX_STOP_SUBMIT_ATTEMPTS`。
- `python/akquant/gateway/broker_runtime.py`:透传 `record_stop_remap`。
- `python/akquant/live.py`:`_broker_to_local_stop_id`、`_record_stop_remap`、`_lookup_stop_local_id`、`_adapt_strategy_payload` 传 local_id、终态清理、注入。
- `python/akquant/gateway/broker_event_adapter.py`:`map_order_snapshot`/`map_trade` 加 `local_id` 参数。
- 文档:`docs/zh/advanced/qmf_broker_gateway.md`「本地止损单」小节补 id 连续性 + 失败重试说明。

## 7. 不做(YAGNI / 后续)

- P4b:OCO/Bracket 的 broker_live 本地化(独立后续,依赖本 remap)。
- requeue 的退避/间隔策略(本次固定每 tick 重试、上限 3 次)。
- 部分成交时 local id 的更细状态机(现按 broker_order_id 一律映射 local id 即可)。

## 8. 待确认

1. requeue 上限 3 次后放弃(默认)。
2. id 连续性 = 覆盖 `id`/`order_id` 为 LSTOP-n,保留 broker/client id 字段(默认)。
3. remap 存 LiveRunner、经回调注入 BrokerExecution(默认,同 record_order_request 模式)。
