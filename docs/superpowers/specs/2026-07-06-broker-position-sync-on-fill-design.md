# 实盘成交同步更新持仓缓存（单一来源）设计

日期：2026-07-06
状态：设计待评审
关联：[[live-broker-readiness]] 「事件溯源 `self.pos` 台账」的**重定向**——不新增 `self.pos` 入口（与 `get_position` 冗余），改为在源头让实盘持仓缓存成交即更新，使现有 `get_position`/`positions`/`self.position` 实盘也同步准。

## 目标（一句话）

broker_live 下，成交事件同步把持仓 delta 应用到 `BrokerStateCache` 的总持仓，让 `get_position`/`positions`/`self.position` 成交后**立即准**（不再依赖"失效→异步重查柜台"而滞后），保持**单一真相来源**、**零新增 API 面**。

## 背景（现状 + 决策依据）

- 现有持仓入口（`get_position` 243 次调用 / `positions` 85 / `get_available_position` 42 / `self.position` 对象 24）**都读同一个源**（回测 ctx / 实盘 `BrokerStateCache`）。评估结论（见对话）：这些是同源的多种人体工学视图，各有定位（标量/批量/可用/富对象），**不是 footgun**，**不精简**；真正的最佳实践是**单一来源**，靠本设计达成。故**不加 `self.pos`**（会与 `get_position` 冗余）。
- 当前 `BrokerStateCache`：`invalidate()` 置所有 `*_loaded=False`；`positions()`/`available_positions()` 惰性 `_load_positions()`（查柜台）。`wrap_state_invalidation`（`broker_strategy_api.py:50`）在 `order`/`trade` 事件对**所有** slot 缓存 `invalidate()`。
- **滞后源**：成交 → invalidate → 下次 `get_position` 重查柜台，而柜台快照可能尚未反映刚成交（传播延迟），或重查加网络时延。
- **持仓是账户级**：每 slot 一个 `BrokerStateCache` 但都绑同一 `trader_gateway`，`query_positions()` 无 slot 过滤（查整账户）。故一笔成交的 delta 应用到**所有**缓存是对的（各缓存镜像同一账户），无按 slot 归属难题。
- `UnifiedTrade`：`symbol`/`side`(str "Buy"/"Sell")/`quantity`(float)/`position_effect`；`UnifiedPosition`：`quantity`/`available_quantity`。

## 决策（用户已定）

- **事件溯源总持仓**（成交同步 delta），**可用持仓仍走柜台查询**（T+1/T+0 按资产类型，柜台权威；事件溯源可用需复刻规则、易错）。
- **薄款、不自动对账**：启动/恢复整柜台重查 seed（权威覆盖）；会话中只叠 delta、不自动重查；漏接一笔成交会漂移——留后续。
- **持仓入口不动**（`position(symbol)` 可选项留后续，本次不做）。

## 架构

**纯 Python，Rust 零改动，仅改 broker_live 状态缓存与事件接线；回测（ctx）不受影响。**

### 1. `BrokerStateCache` 改造（`gateway/broker_state_cache.py`）

- **拆 loaded 标志**：`_positions_loaded` → `_total_loaded` + `_available_loaded`（总持仓与可用解耦，避免可用重查覆盖事件溯源的总持仓）。
- `positions()`：`if not self._total_loaded: self._load_positions()`（整查：填 `_positions` + `_available`，两标志置 True——即 seed/reconcile）；返回 `dict(self._positions)`。
- `available_positions()`：`if not self._available_loaded: self._load_available()`（**仅**填 `_available` + `_available_loaded`，**不动** `_positions`）；返回 `dict(self._available)`。
- **`apply_fill(symbol, signed_qty)`**（新）：
  ```python
  def apply_fill(self, symbol: str, signed_qty: float) -> None:
      # 仅在总持仓已 seed 时叠 delta; 未 seed 则不动
      # (下次 positions() 从柜台整快照 seed, 已含该笔, 避免双计)。
      if self._total_loaded:
          self._positions[symbol] = self._positions.get(symbol, 0.0) + signed_qty
  ```
  不触碰 `_available`（可用交给重查）。
- **粒度失效**（新，供接线区分 order/trade）：`invalidate_available()`（`_available_loaded=False`）、`invalidate_account()`、`invalidate_open_orders()`。保留 `invalidate()`（全部，供 recovery/启动重 seed）。
- `_load_available()`（新）：查柜台，**仅**置 `_available` + `_available_loaded`（不动 `_positions`）。`_load_positions()` 仍置两者（整 seed）。

### 2. 事件接线（`gateway/broker_strategy_api.py` `wrap_state_invalidation`）

```python
def _wrapped(event_name, payload):
    update_broker_state(event_name, payload)
    caches = get_caches() or ()
    if event_name == "trade":
        symbol = _trade_field(payload, "symbol")
        signed = _signed_fill_qty(payload)  # Buy:+qty, Sell:-qty
        for cache in caches:
            if cache is not None:
                if symbol:
                    cache.apply_fill(symbol, signed)  # 总持仓同步 delta
                cache.invalidate_available()           # 可用重查(T+1)
                cache.invalidate_account()             # 现金变
                cache.invalidate_open_orders()         # 成交可能关单
    elif event_name == "order":
        for cache in caches:
            if cache is not None:
                cache.invalidate_open_orders()         # 挂撤单变
                cache.invalidate_account()
                # 不动持仓/可用: 挂撤单不改持仓
```
辅助：`_trade_field(payload, name)`（getattr 优先、dict.get 兜底）；`_signed_fill_qty(payload)`（取 side/quantity；`str(side).split(".")[-1].lower()=="buy"` → `+qty` 否则 `-qty`）。

### 3. 对账（recovery）

启动/恢复走既有 `invalidate()`（全部）→ 下次 `positions()` 从柜台整 seed（权威覆盖累积 delta），即重定期对账点。会话中不自动重 seed 总持仓（否则触发滞后重查、且可能双计）。

### 4. 符号约定

`apply_fill` 按 side 带符号（Buy `+`、Sell `-`），与"净持仓"解读一致——A股长仓精确；期货空头沿用既有缓存 `query_positions` 的 quantity 约定（现状即以原始 quantity 存，非本次引入）。文档写明。

## 测试策略

- **成交同步准**：mock gw 首查返回底仓，`positions()` seed；`apply_fill("X", +100)` 后 `positions()["X"]` 立即 +100，**未再查柜台**（mock 计数不增）。
- **防双计**：未 seed 时 `apply_fill` no-op；随后 `positions()` seed（mock 返回已含该笔）→ 值正确、无叠加。
- **可用重查不覆盖总持仓**：seed 总+可用 → `apply_fill` → `invalidate_available()` → `available_positions()` 重查（mock 变）→ `_positions`（总）仍含 delta 不被覆盖。
- **order 事件不动持仓**：经 `wrap_state_invalidation` 发 order → `positions()` 不变、不触发重查。
- **trade 事件接线**：发 trade → 所有缓存 `apply_fill` 被调、总持仓不失效、可用/account/open_orders 失效。
- **多缓存账户级**：两缓存，一笔 trade → 两者总持仓都 +delta。
- **side 兼容**：Buy `+`、Sell `-`；str 与枚举兜底。
- **既有 broker_state / live_runner 测试**：凡断言"成交后总持仓重查/invalidate 全部"的，改为断言新行为（成交叠 delta、总持仓不失效）——不是弱化，是行为更新（记录于 plan）。
- 全量 `tests/` 全绿；ruff；**Rust 零改动断言** 空。

## 非目标（YAGNI）

- **不加 `self.pos`**（与 `get_position` 冗余）。
- **不精简**持仓入口（`get_position`/`positions`/`get_available_position`/`self.position` 保持）。
- **可用持仓不事件溯源**（走柜台查询，T+1/T+0 归柜台）。
- **不自动对账**（仅启动/恢复 seed；漏单漂移留后续）。
- **`position(symbol)`** 富对象任意标的——可选、留后续。
- 回测/ctx 不改。

## 交付物

- `gateway/broker_state_cache.py`（`apply_fill` + 拆 loaded + 粒度失效 + `_load_available`）。
- `gateway/broker_strategy_api.py`（`wrap_state_invalidation` 区分 order/trade + `_signed_fill_qty`/`_trade_field`）。
- `tests/test_broker_position_sync.py`（上述）+ 更新受影响既有测试。
- docs：`docs/zh/advanced/qmf_broker_gateway.md` 加「实盘持仓同步（成交即更新）」小节：成交同步 delta、可用仍重查、启动/恢复对账、单一来源（get_position 等实盘也准）。

## 收尾

- 分支 `feat/broker-position-sync`（基于 dev）；中文 Conventional Commits + `--no-verify`；未获明确要求不 push。
- 最终 opus 全分支评审后合并 dev。
