# broker_live 下单签名一致（③a）设计

日期：2026-07-03
状态：待评审
动机：便捷下单族（`buy/sell/order_target/short/cover/...`）在 `broker_live` 会 `TypeError`，因为它们把回测专有参数转发给 broker 版 `submit_order`，而后者不接受。目标：让这一族在实盘可用，并对每个参数定明确语义。

## 1. 背景审计（实盘 API 发散全貌，供 ③a/b/c/d 参考）

`broker_live` 下 `BrokerOrderSubmitter.install()` **只覆盖** `submit_order`/`can_submit_client_order`/`get_execution_capabilities`；其余策略方法仍读写 `strategy.ctx`（sim 引擎），而**broker 成交从不回灌引擎**（`BrokerEventBridge` 仅触发回调）。由此三类发散：

- **A 崩（TypeError）**：`buy/sell/short/cover`（`strategy_trading_api.py:229/289/1732/1799`）无条件把 `fill_policy/slippage/commission/trail_offset/trail_reference_price` 转发给 broker `submit_order`（14 参数、无 `**kwargs`、不含这些）→ `TypeError`。继承者：`order_target/order_target_value/order_target_percent/order_target_weights/order_target_positions/close_position/buy_all/stop_buy/stop_sell`。**← 本阶段 ③a 修这类。**
- **B 静默走 sim（③b/③c）**：`get_position/get_available_position/get_account/get_portfolio_value/get_open_orders` 读 `ctx`（实盘为 sim/空，非柜台）；`cancel_order/cancel_all_orders` → `ctx.cancel_order`（sim，非柜台）；`buy_all/order_target_*` 用 `ctx.cash`/持仓算量。
- **C 静默丢语义（③d）**：`stop_buy/stop_sell`、任何 `trigger_price` → broker `submit_order` `_ = trigger_price` 丢弃。

本阶段仅做 **③a**；B/C 记录在案、后续 ③b（撤单路由）/③c（实盘状态读）/③d（条件单）分别处理。

## 2. 目标（③a）

`broker_live` 的 `BrokerOrderSubmitter.submit_order` 接受与回测 `submit_order` **一致的完整签名**，并逐参数定明确语义：便捷下单族不再 `TypeError`；回测专有的模拟旋钮被安全忽略；实盘不支持的条件/追踪语义**明确报错**（不再静默）。

## 3. 已确认决策

- 让 broker 版 `submit_order` 增补缺失参数：`fill_policy / slippage / commission / trail_offset / trail_reference_price / broker_options`（与回测 `submit_order` 对齐）。
- 逐参数语义：
  - **模拟旋钮（实盘由真实市场/柜台决定）→ 收下忽略 + 一次性告警**：`fill_policy`、`slippage`、`commission`、`broker_options`（`broker_options` 后续可映射到 `extra`，本阶段忽略告警）。
  - **不支持的条件/追踪 → 明确报错(不静默)**：`trigger_price is not None` → `RuntimeError`（broker_live 不支持条件/止损触发单）；`trail_offset is not None` 或 `order_type ∈ {StopTrail, StopTrailLimit}` → `RuntimeError`（不支持追踪止损）。
  - `tag` 继续收下（元数据，无害）。
  - **关键**：便捷族传这些参数时默认值为 `None`——`None` 一律**放行/忽略**，只有**显式非 None（真用了）**才忽略告警或报错。因此 `self.buy(symbol, qty)`（全 None）在实盘正常下单。

## 4. 组件设计（单点改动）

### 4.1 `python/akquant/gateway/order_submitter.py` —— `BrokerOrderSubmitter.submit_order`
- 签名增补 6 个参数（默认 `None`），置于现有参数之后，与回测 `submit_order` 命名一致。
- 方法体在现有 `broker_ready` 守卫之后、下单逻辑之前插入「参数语义闸」：
  - 模拟旋钮非 None → `logger.warning`（一次性，注明"broker_live 忽略回测模拟参数 X"）后忽略。
  - `trigger_price is not None` → `raise RuntimeError("broker_live 暂不支持条件/止损触发单(trigger_price)")`。
  - `trail_offset is not None` 或 `order_type` 属追踪族 → `raise RuntimeError("broker_live 暂不支持追踪止损单")`。
  - 其余（含 `tag`）不影响既有 `UnifiedOrderRequest` 构造。
- **不改** `buy/sell/...`（它们已转发；本阶段让 broker `submit_order` 接住即可）。
- **不改**回测 `strategy_trading_api.submit_order`（已接受这些参数）。

## 5. 数据流（实盘 `self.buy(...)`）

`self.buy(symbol, qty)` → `strategy_trading_api.buy` → `strategy.submit_order`(=`BrokerOrderSubmitter.submit_order`)，此时 `fill_policy/slippage/commission/trail_*=None` → 语义闸全放行 → 正常构造 `UnifiedOrderRequest` → `place_order`。若 `self.buy(symbol, qty, trigger_price=X)` → 语义闸抛清晰 `RuntimeError`。

## 6. 错误处理

- 显式条件/追踪 → 清晰 `RuntimeError`（也可被 `on_error` 捕获）。
- 模拟旋钮 → 告警但不阻断。
- 其余沿用现有（`broker_ready` 守卫、能力校验、`extra` 校验）。

## 7. 测试策略（TDD）

- broker_live `submit_order` 接受完整签名不 `TypeError`：直接构造 `BrokerOrderSubmitter`，以「便捷族会传的全套 kwargs（模拟旋钮=None）」调用 → 到达 `place_order`。
- **验收（用户要求）**：模拟经 `buy()` 路径——把一个 `broker_ready=True` 的策略桩的 `submit_order` 设为 `BrokerOrderSubmitter.submit_order`，调 `strategy_trading_api.buy(strategy, symbol, qty)` → 不再 `TypeError`、`place_order` 被调用。
- 模拟旋钮非 None（`fill_policy`/`slippage`/`commission`）→ 不崩、有告警、正常下单。
- `trigger_price` 非 None → 清晰 `RuntimeError`；`trail_offset` 非 None / 追踪 order_type → 清晰 `RuntimeError`。
- 现有 gateway/live_runner 全套无回归；ruff `E,F,I,D`。

## 8. 不做（YAGNI / 冻结）

- ③b 撤单路由、③c 实盘状态读（get_position/account 反映柜台）、③d 条件单真做——各自独立后续。
- 不用 `capability.features` 门控条件/追踪支持（本阶段一律清晰报错；未来可用 features 放开支持该能力的 broker）。
- 不把 `broker_options` 真映射到 `extra`（本阶段忽略告警）。
- 不改回测语义、不改便捷族实现。

## 9. 影响面

- `python/akquant/gateway/order_submitter.py`（`BrokerOrderSubmitter.submit_order` 签名 + 语义闸）
- Tests：`tests/test_gateway_broker_submitter_signature.py`
- （可选）`docs/zh/advanced/qmf_broker_gateway.md` 或 live 文档补一句"实盘忽略回测模拟参数、条件/追踪单明确报错"。

## 10. 待确认

1. 条件/追踪单：**清晰报错**（默认已选）vs 现在就用 `capability.features` 门控（支持的 broker 放行）。默认前者，后者留作未来。
2. `broker_options`：本阶段**忽略告警**（默认）vs 立刻映射到 `extra`。默认忽略。
3. 模拟旋钮告警级别：`warning` 一次性（默认）vs `debug`。默认 warning（实盘更需可见）。
