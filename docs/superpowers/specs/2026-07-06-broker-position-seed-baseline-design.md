# broker_live 冷启动"seed vs 重放"双计修复(持仓基线激活)设计

日期：2026-07-06
状态：设计待评审
关联：[[live-broker-readiness]] 「实盘成交同步更新持仓缓存」+「成交重放去重」暴露的确认 bug——盘中重启双计。

## 目标（一句话）

broker_live 冷启动/盘中重启时，消除"持仓快照(已含今日成交) + 恢复重放同批成交叠 delta"导致的**持仓静默高计**：在一个**激活点**原子地 seed 持仓 + 把当时的 `trade_id` 灌入 dedup 基线，且激活前不 `apply_fill`、恢复不重放成交。

## 背景（确认的 bug + 关键时序）

- **已复现**(组件级)：cache 从 `query_positions` seed 得 100(快照**已含**今日成交 T)→ 恢复 `sync_today_trades` 重放 T(trade_id 全新，`_seen_trade_ids` 启动为空)→ `apply_fill(+100)` → **200**。盘中重启且今日已有成交时触发。
- 根因：**快照(含至 seed 时的成交) 与 event-sourcing(叠成交) 混用**，任何"既在快照里、又被 apply_fill 应用"的成交都会双计。唯一根治：**seed 与 dedup 基线在同一激活点原子确立，且激活前禁止 apply_fill / 禁止恢复重放成交**。
- **关键时序**(`live.py:run`)：`_bind_broker_callbacks`(:361)已启动 drain dispatcher **与恢复线程**(`_start_broker_recovery`)，随后才 `_install_broker_order_submitter`(:363)、`_await_broker_ready`(:364)。恢复线程默认 1s 间隔；就绪(登录)前其查询失败被跳过，登录一完成恢复的下一周期就会成功重放成交——**与"就绪后灌基线"存在跨线程竞态**。故基线激活必须**门控恢复的成交重放**直到激活完成。
- `_seen_trade_ids` 在 `BrokerEventBridge`(bridge 由 `LiveRunner._broker_event_bridge` 持有)；各 slot `BrokerStateCache` 在 `BrokerRuntime._broker_state_caches`。`apply_fill` 已"仅 `_total_loaded` 时叠"(未 seed no-op)。

## 决策（用户已定）

- 在**就绪激活点**做"急切 seed 持仓 + 灌 trade 基线"；基线**丢弃基线前成交**(不 apply、不重放 on_trade——重启按快照重建，不重跑当日逻辑)；残留"两查询间隙"竞态文档化(取 seed 先→残留极小**低计**，比高计对风险安全)。

## 架构

**纯 Python，Rust 零改动。** 引入一次性**激活**：seed + 基线 + 置激活标志；恢复的成交重放门控在激活标志后。

### 1. `BrokerEventBridge.mark_trades_seen(trade_ids)`（新）

```python
def mark_trades_seen(self, trade_ids: Iterable[str]) -> None:
    """把 trade_id 灌入会话级 dedup 基线(已烘进持仓快照, 后续重放将被 queue_event 丢弃)."""
    with self._event_lock:
        for tid in trade_ids:
            if tid:
                self._seen_trade_ids.add(str(tid))
```

### 2. `LiveRunner._baseline_broker_state(trader_gateway)`（新，激活一次）

```python
def _baseline_broker_state(self, trader_gateway):
    # 顺序: 先 seed 持仓(快照), 后灌 trade 基线 → 两查询间隙的成交残留为"低计"(较安全)。
    for cache in self._broker_runtime._broker_state_caches:
        try:
            cache.positions()          # 急切 seed 总持仓(_total_loaded=True)
        except Exception: log...       # 保守: 失败不炸, 交给懒 seed
    sync = getattr(trader_gateway, "sync_today_trades", None)
    if callable(sync):
        try:
            tids = [getattr(t, "trade_id", None) for t in sync()]
            self._broker_event_bridge.mark_trades_seen(tids)
        except Exception: log...
    self._broker_baseline_done = True
```
- `self._broker_baseline_done`：`__init__`/`_init_broker_bridge_state` 初始化为 `False`。
- 调用点：`_await_broker_ready` 内 `ready=True` 之后、`_dispatch_broker_connected` 之前（此时仍在 `run()` 建立阶段，策略回调尚未开始 → 激活先于任何策略读持仓）。未就绪(超时)则不激活(不进 broker_live 交易)。

### 3. 恢复门控成交重放（`broker_recovery.py`）

`run_cycle` 的 `sync_today_trades` 重放段(:79-83)加门控：**激活未完成前不重放成交**(避免抢在基线前 apply)。经新回调 `should_replay_trades()`（→ `lambda: self._broker_baseline_done`）：
```python
if callable(sync_today_trades) and self._should_replay_trades():
    for trade in sync_today_trades():
        self._queue_broker_event("trade", trade)
```
（`sync_open_orders`/`query_account` 不门控——幂等、不喂 apply_fill。`BrokerRuntime` 构造 recovery 时注入 `should_replay_trades` 回调；默认无回调时视为 True，保后向兼容。）

### 4. 为何这样根治

- 激活点：seed 快照(权威基线) + 把快照已含的 `trade_id` 全标 seen。此后 `_total` = 快照 + 仅激活后新成交的 delta。
- 激活前：`apply_fill` no-op(cache 未 seed)；恢复不重放成交(门控)；激活先于策略读持仓 → 无过早懒 seed。
- 激活后：恢复重放"基线前成交"→ 命中 `_seen` → `queue_event` 丢弃(不 apply、不 on_trade)；新成交(新 trade_id)→ 正常 apply。
- 顺带修：重启不再把当日历史成交重放给 `on_trade`(策略不重跑一天逻辑)。
- 残留：seed 与基线两次查询之间到来的成交(极小窗口)——取 seed 先 → 该成交在快照后、基线前 → 被标 seen 不 apply、且不在快照 → 轻微低计，文档化。

## 测试策略

- **双计已修(核心, 复现原 bug)**：cache seed 含 T(100) → `mark_trades_seen(["T"])` → 经 wrap/queue 重放 T → 丢弃 → 持仓仍 100(改前为 200)。
- **激活后新成交仍 apply**：基线后来的新 trade_id 经 apply_fill 叠 delta。
- **`mark_trades_seen`**：灌入的 id 后续 `queue_event("trade", {trade_id})` 被丢(不入 store)；空/None 跳过。
- **恢复门控**：`_broker_baseline_done=False` 时 recovery `run_cycle` 不重放成交(`_queue_broker_event` 未被以 trade 调用);`True` 后重放。用 fake gateway + 计数验证。
- **激活时序**：`_await_broker_ready` 就绪→调 `_baseline_broker_state`→置 `_broker_baseline_done`;未就绪不置。
- **不重放 on_trade 历史**：激活+门控下，重启不对基线前成交触发 on_trade(经既有 bridge 测试风格断言)。
- **既有 recovery 测试**：`test_live_runner_broker_recovery.py`/`test_live_runner_broker_bridge.py` 的 recovers_from_sync——现需先激活(置 `_broker_baseline_done=True`)才重放成交；按新门控更新(记录)。
- 全量 `tests/` 全绿；ruff；**Rust 零改动断言** 空。

## 非目标（YAGNI）

- 消除激活两查询间隙的残留竞态(需柜台原子"持仓+成交"快照或序列号，柜台不提供)——文档化即可。
- mid-session 自动对账/断线重连重激活(生产不 mid-session `invalidate()`；断线重连再激活留后续)。
- order/account 重放门控(幂等，无需)。

## 交付物

- `gateway/broker_event_bridge.py`（`mark_trades_seen`）。
- `gateway/broker_recovery.py`（`should_replay_trades` 门控）+ `gateway/broker_runtime.py`（注入回调）。
- `live.py`（`_baseline_broker_state` + `_broker_baseline_done` + `_await_broker_ready` 接线 + recovery 回调 `lambda: self._broker_baseline_done`）。
- `tests/test_broker_position_seed_baseline.py`（上述）+ 更新受影响 recovery 测试。
- docs：`docs/zh/advanced/qmf_broker_gateway.md`「实盘持仓同步」小节补——冷启动激活基线(seed+标 seen)消双计、重启不重放历史 on_trade、残留低计窗口。

## 收尾

- 分支 `feat/broker-seed-baseline`（基于 dev）；中文 Conventional Commits + `--no-verify`；未获明确要求不 push。
- 最终 opus 全分支评审后合并 dev。
