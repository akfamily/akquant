# broker_live 成交重放去重（on_trade 恢复重放）设计

日期：2026-07-06
状态：设计待评审
关联：[[live-broker-readiness]] 「实盘成交同步更新持仓缓存」终审暴露的 pre-existing 遗留 #1——`on_trade`（及 `_process_order_groups`）在恢复重放时重复触发。

## 目标（一句话）

broker_live 下同一笔成交（同 `trade_id`）**每会话只派发一次**，杜绝恢复循环每周期 `sync_today_trades()` 重放导致 `on_trade`/`_process_order_groups` 反复触发；同时保留恢复"补漏"能力（断线期间漏推的成交仍会被派发一次）。

## 背景（现状核实）

- 恢复循环 `BrokerRecovery.run_cycle` 每周期（默认 1.0s）**无条件**调 `sync_today_trades()`，对每笔当日成交 `_queue_broker_event("trade", trade)`（`broker_recovery.py:79-83`）。
- 队列去重 `BrokerEventBridge.queue_event` 用 `_event_keys`（trade 键 = `trade:{trade_id}`），但 `drain_events` **每次 drain 清空 `_event_keys`**（`broker_event_bridge.py:51`）→ 仅**批内**去重；跨周期同一 `trade_id` 重新入队 → 再次派发 → `on_trade` + `_process_order_groups` 重复触发。
- 本应作会话级去重的 `LiveRunner._broker_trade_keys`（`live.py:762` 建、`:1020` populate）**从不被读**——死代码。
- 前序「持仓同步」已给 `apply_fill` 加了 `trade_id` 去重（`wrap_state_invalidation`），但那只护住**持仓 delta**；`on_trade`/`_process_order_groups` 仍重复触发。

**危害**：`on_trade` 重复回调（用户逻辑重复执行）；`_process_order_groups` 重复跑 → OCO 对手腿重复撤单尝试、Bracket 重复激活尝试。account 重放幂等（无害）；order 重放另有状态语义（不在本次范围）。

## 决策（用户已定）

- **仅 trade 去重**（成交不可变 → 每会话派发一次语义明确正确）。order 有状态（提交/部分/成交/撤）不可盲目去重，单独留后续；account 无害。
- 无 `trade_id` 的成交**无法去重** → 照常派发（罕见；文档写明，与 `apply_fill` 的无 id 退回一致）。

## 架构

**纯 Python，Rust 零改动。** 在**单一入口** `BrokerEventBridge.queue_event`（实盘推送与恢复重放都经此）加**会话级 `trade_id` 去重**：某 `trade_id` 已入过队则丢弃（不进 `_event_store` → 下游 `_update_broker_state`/`on_trade`/`_process_order_groups`/`apply_fill` 都不再见到该重复）。

### 实现（`gateway/broker_event_bridge.py`）

- `__init__` 加 `self._seen_trade_ids: set[str] = set()`（会话级，**不随 drain 清空**）。
- `queue_event` 改：
  ```python
  def queue_event(self, event_name: str, payload: Any) -> None:
      """Add a broker event to the dispatch queue with semantic deduplication."""
      event_key = self._make_event_key(event_name, payload)
      trade_id = ""
      if event_name == "trade":
          raw = self._payload_to_dict(payload).get("trade_id")
          trade_id = str(raw) if raw else ""
      with self._event_lock:
          if trade_id:
              if trade_id in self._seen_trade_ids:
                  return  # 会话级: 该成交已入队(实盘推送/恢复重放), 丢弃防重复派发
              self._seen_trade_ids.add(trade_id)
          if event_key in self._event_keys:
              return
          self._event_keys.add(event_key)
          self._event_store.append((event_name, payload))
  ```
- `drain_events` 不变（仍只清 `_event_keys`；`_seen_trade_ids` 保持会话级）。

### 为何保留"补漏"

恢复重放的意义是补断线期间漏推的成交。去重按 `trade_id`：**没见过**的 `trade_id`（漏推）→ 不在 `_seen_trade_ids` → 正常入队派发一次（补上）；**见过**的（已实盘推送）→ 丢弃（不重复）。故去重**不损**补漏，只除重复。

### 与既有 apply_fill 去重的关系

`wrap_state_invalidation` 的 `applied_fill_ids`（持仓 delta 去重）在本改动后成为**冗余的第二道防线**（重复 trade 已在 queue 处丢弃、不再到达 `_update_broker_state`）——保留，防御任何绕过 queue 的路径，且处理无 `trade_id` 情形。两处去重独立、语义一致。

## 测试策略

- **跨 drain 去重**：`queue_event("trade", {trade_id:T,...})` → drain（派发 on_trade 一次）→ 再 `queue_event` 同 T → drain → **on_trade 总共只调一次**、`_event_store` 第二次为空。
- **补漏**：未见过的新 `trade_id` 经 queue → 正常派发（模拟漏推补上）。
- **无 trade_id 不去重**：两笔无 `trade_id` 的成交（不同 payload）→ 都派发（记录该限制）。
- **_process_order_groups 不重复**：重复 trade 不再触发第二次 `_process_order_groups`（可用 fake 策略计数）。
- **不同 trade_id 不误杀**：T1、T2 各派发一次。
- **既有 bridge/contract 测试**：`tests/test_live_runner_broker_bridge.py`、`tests/gateway_contract/test_event_bridge_contract.py`、`tests/test_broker_bridge_*`——凡不涉重复 `trade_id` 的，行为不变应仍绿；若某测试恰好复用同一 `trade_id` 跨 drain 期望多次派发，按新语义更新（记录）。
- 全量 `tests/` 全绿；ruff；**Rust 零改动断言** 空。

## 非目标（YAGNI）

- **order 重放去重**（有状态，需状态键区分新旧；单独后续）。
- **account 重放**（幂等无害，不处理）。
- **`_seen_trade_ids` 有界化/按日重置**（日成交量级内无忧；多日长驻进程再说）。
- 移除死代码 `_broker_trade_keys`（本次不动，避免扩面；可后续清理）。
- 断线重连触发整柜台对账（前序已列后续）。

## 交付物

- `gateway/broker_event_bridge.py`（`queue_event` + `__init__` 的 `_seen_trade_ids`）。
- `tests/test_broker_trade_redelivery_dedup.py`（上述）+ 按需更新受影响既有 bridge 测试。
- docs：`docs/zh/advanced/qmf_broker_gateway.md`「实盘持仓同步」小节旁补一句——成交按 `trade_id` 会话级去重，恢复重放不重复触发 `on_trade`/`_process_order_groups`，补漏保留。

## 收尾

- 分支 `feat/broker-trade-dedup`（基于 dev）；中文 Conventional Commits + `--no-verify`；未获明确要求不 push。
- 最终 opus 全分支评审后合并 dev。
