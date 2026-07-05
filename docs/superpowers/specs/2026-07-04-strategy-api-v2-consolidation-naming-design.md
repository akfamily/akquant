# Strategy API v2：下单族收敛 + 命名一致性(硬改)设计

日期：2026-07-04
状态：待评审
动机：对比 vnpy/backtrader/backtesting.py/zipline 评估后，akquant 策略编写接口**功能已是超集**，但存在(1)下单族重复与两处隐藏行为分歧、(2)冗余薄封装、(3)命名不一致(getter 方法 vs 属性混用、复合单动词不统一、多标的目标单命名不显性)。本次借破坏性窗口收敛为一致的 v2 接口。用户已明确授权：**全硬改、不留 deprecation 别名**（副本环境）。

## 0. 范围边界（关键）

- 本次只改**策略作者可见接口**：`Strategy` 类的公共方法/属性 + `strategy_trading_api.py` 的自由函数。
- **不改** `ExecutionBackend` 协议（`SimExecution`/`BrokerExecution` 的 `get_positions()`/`get_cash()`/`get_account()`/… 是内部管线，保持原名）。例如新的 `Strategy.positions` 属性内部仍调用 `strategy.execution.get_positions()`。迁移时必须区分 `strategy.get_positions()`（改）与 `execution.get_positions()`（不改）。
- 不改回测引擎、gateway、feed。

### Rust / Python 边界（本次 100% Python，Rust 零改动）
已核实 `./src/*.rs` 与 `python/akquant/akquant.pyi`：
- **Rust→Python 按名回调**仅：`_on_bar_event`/`_on_tick_event`/`_on_timer_event`/`_flush_pending_order_events`（`src/engine/core.rs`）、`_on_start_internal`/`on_start`（`src/engine/python.rs`）——**均不在改名清单**，回调不动。
- Rust 里的 `get_cash`/`get_positions` 是 `StrategyContext` 的 `#[getter]`（`src/context.rs:675/685`，Python 侧即 `ctx.cash`/`ctx.positions`）与 `Portfolio` 的 `#[pymethods]`（`src/portfolio.rs:144/151`）——属**引擎/ctx 层**，v2 不碰。
- `Strategy` 是纯 Python 类，**不在 `.pyi`**（`.pyi` 由 `src/bin/stub_gen.rs` 生成）。不改 Rust ⇒ 无需 `cargo build`/`maturin`、无需改 `.pyi`。
- talib Rust 后端本次不涉及。

**迁移硬约束**：改名只动 `Strategy` 类方法/属性 + 指定 `strategy_trading_api` 自由函数；**绝不改**：`ctx.cash`/`ctx.positions`/`ctx.get_position()`、`portfolio.get_cash()`/`portfolio.get_positions()`、`execution.get_positions()` 等（ExecutionBackend 协议）、Rust `#[getter]`/`#[pymethods]` 定义、`_on_*_event` 派发名。这条与下面「区分 receiver」的完备性守卫合并检查。

### 两层澄清
用户可见接口 = `Strategy` 类的公共方法/属性；`strategy_trading_api.py` 自由函数是**内部 impl**。
  - 状态读 getter→property：只删 `Strategy` 的公共方法 + 加属性；属性委托到现有自由函数 impl（impl 保留其名，不需改）。
  - 被改名的**下单函数**（`rebalance_positions`/`rebalance_weights`/`place_bracket`/`place_oco`）：`Strategy` 方法与对应自由函数**两层一并改名**（自由函数名即调用目标）。
  - `ExecutionBackend` 协议方法名一律不改。

## 1. 下单族功能收敛

### 1.1 共享核心
新增 `_target_to_orders(strategy, symbol, target_qty, price, **kwargs) -> Optional[str]`，替换 `_order_target_core`：
1. `symbol = resolve_symbol(strategy, symbol)`；`current = float(strategy.execution.get_position(symbol))`；`delta = target_qty - current`。
2. **统一按 `lot_size` 向下取整 delta**（沿用现 `order_target_value` 行 1059-1070 的取整逻辑：`delta>0 → (delta//lot)*lot`；`delta<0 → -((|delta|//lot)*lot)`）。
3. **不做** `cancel_all_orders`。
4. `delta>0 → buy`；`<0 → sell`；`==0 → None`。

### 1.2 各目标类函数改为走核心
- `order_target(symbol, target, price)` → `_target_to_orders(...)`。**行为变化：现在会按 lot 取整**（之前不取整）。
- `order_target_value(symbol, target_value, price)` → 保留 value→qty（现有定价解析），**去掉隐藏的 `cancel_all_orders(symbol)` 与内联取整**，改调 `_target_to_orders(symbol, target_qty, price)`。**行为变化：不再自动撤单**。
- `order_target_percent` → 仍 `order_target_value`（继承取整+不撤单）。
- `rebalance_weights`（原 `order_target_weights`）→ 逐标的 `order_target_value`（继承）。
- `rebalance_positions`（原 `order_target_positions`）→ 逐 leg 调 `_target_to_orders`（**现在会取整**；保留其现有 broker_live 做空能力校验/plan 构建逻辑不变）。
- `close_position(symbol)` → **全平当前持仓**：经共享核心但 **`round_to_lot=False`**（`_target_to_orders(strategy, symbol, 0, round_to_lot=False)`），卖出精确持仓含 A 股零股，真正 flatten。（`_target_to_orders` 加 `round_to_lot: bool = True` 参数；`order_target`/value/percent 用默认取整——不能买 137 股；只有 close_position 全平不取整。修正原「闭仓恒整手」假设——零股持仓由分红/拆股产生，A 股允许卖零股。）

### 1.3 硬删薄封装（无别名）
- `stop_buy` / `stop_sell`（`Strategy` 方法 + 自由函数两层）→ 删除；迁移到 `submit_order(side, trigger_price=...)` 或 `buy/sell(..., trigger_price=...)`。
- `buy_all`（`Strategy` 方法 + 自由函数）→ 删除；迁移到 `order_target_percent(1.0)`。
- **保留** `short`/`cover`（期货开平语义）、`place_trailing_stop`/`place_trailing_stop_limit`（StopTrail，非纯冗余）。

## 2. 命名一致性(v2，全硬改无别名)

### 2.1 状态读：无参廉价=属性 / 带参=方法
| 现在（删） | v2（新） | 备注 |
|---|---|---|
| `get_cash()` 方法 | **`cash`** 只读属性 | 内部 `return self.execution.get_cash()` |
| `get_portfolio_value()` 方法 | **`equity`** 只读属性（已存在，设为权威） | 删 `Strategy.get_portfolio_value()`；`equity` 当前调 `self.get_portfolio_value()`（strategy.py:2226），改为直接调 `_get_portfolio_value_impl(self)`（自由函数 impl 保留） |
| `get_positions()` 方法 | **`positions`** 只读属性 | 内部 `self.execution.get_positions()` |
| `hold_bar(symbol)` | **`get_holding_bars(symbol)`** | 带参→方法；语义计数正名 |
| `get_position(symbol)` / `get_available_position(symbol)` | 不变 | 带参=方法 |
| `get_account()` | 不变 = **权威全量账户源** | — |
| `get_open_orders`/`get_order`/`get_trades` | 不变 | — |

新增只读属性 `cash`（无碰撞，已核实 `Strategy` 无同名属性）。`equity` 属性已存在，保留为权威。

### 2.2 下单命名：保留惯例动词 + 统一复合单前缀 + 多标的显性
| 现在（删/改） | v2 | 备注 |
|---|---|---|
| `submit_order` / `buy` / `sell` / `short` / `cover` | 不变 | 四大框架通用惯例，改反而差 |
| `order_target` / `order_target_value` / `order_target_percent` | 不变 | 单标的目标（backtrader 标准） |
| `order_target_positions` | **`rebalance_positions`** | 多标的绝对数量调仓 |
| `order_target_weights` | **`rebalance_weights`** | 多标的目标权重调仓 |
| `place_bracket_order` | **`place_bracket`** | 去冗余 `_order` |
| `create_oco_order_group` | **`place_oco`** | 动词统一 `place_`，去 `_group` |
| `place_trailing_stop` / `place_trailing_stop_limit` | 不变 | 已是 `place_` |
| `cancel_order` / `cancel_all_orders` | 不变 | — |

### 2.3 指标
- 删除 `register_indicator` 别名；统一 `register_precomputed_indicator`。
- 两套指标接口(OO vs talib)的边界澄清与流式指标库补齐 → **不在本次**（路线图）。

## 3. 迁移(全硬改)

- **删除旧名**，不留别名、不发 DeprecationWarning。
- 迁移**所有 in-repo 调用**：`python/`（含内部实现）、`examples/`、`tests/`、`docs/`。规模(去 `.pyc`)：`get_positions`~136、`get_portfolio_value`~70、`order_target_positions`~66、`buy_all`~59、`order_target_weights`~58、`get_cash`~47、`hold_bar`~33、`place_bracket_order`~17、`create_oco_order_group`~12、`stop_buy`/`stop_sell`~10、`register_indicator`~10。
- **必须区分** `strategy.get_positions()`（改为 `strategy.positions`）与 `execution.get_positions()`（不改）——逐处判定 receiver。
- **golden 重生成**：`order_target`/`rebalance_positions`（新增取整）与 `order_target_value`/`percent`/`rebalance_weights`（去撤单）会改变受影响回测输出；跑受影响用例、重生成 golden、spec/PR 标注变更集。

## 4. 错误处理

- 硬删/硬改后：外部策略调用旧名 → `AttributeError`（运行时，非 import 期）。这是已授权的破坏性代价。
- 属性化的读（`cash`/`equity`/`positions`）在 `ctx`/execution 未就绪时的回退语义，与其对应 execution 后端一致（backtest ctx None → 0.0/空；不新增守卫）。
- `_target_to_orders` 无守卫（沿用 `_order_target_core` 语义）；`rebalance_positions`/`rebalance_weights`/`buy_all`(删) 的 `_require_execution_ready` 守卫在各自入口保留（`rebalance_*` 保留）。

## 5. 测试策略(TDD)

- **核心**：`_target_to_orders` 的 delta/取整/不撤单单测（fake execution + submit spy）。
- **各 target 函数**：order_target 现在取整、order_target_value 不再撤单、close_position delegate、rebalance_positions/weights 取整——回测 + broker_live 对照。
- **命名**：`cash`/`equity`/`positions` 属性返回值 = 对应 execution 读；`get_holding_bars` = 旧 `hold_bar` 值；旧名不再存在（`assert not hasattr(Strategy, "get_cash")` 等负向断言，防回潮）。
- **完备性守卫**：每个改名项做一次全库 grep（`python/ examples/ tests/ docs/`）确认旧名 0 残留（`.pyc` 除外），防漏改导致运行时 AttributeError。
- **回归**：full `tests/` 全绿（受影响 golden 重生后）；ruff `E,F,I,D`。

## 6. 影响面

- `python/akquant/strategy_trading_api.py`：核心收敛、函数改名、删薄封装。
- `python/akquant/strategy.py`：`Strategy` 方法改名/删除、新增 `cash` 属性、`get_holding_bars`、delegate。
- `python/akquant/indicator*` / `strategy.py`：去 `register_indicator` 别名。
- `examples/`、`tests/`、`docs/`：全量迁移到 v2 名。
- `tests/golden/`：受影响用例重生成。

## 7. 不做（YAGNI / 路线图）

- getter/property 更大范围统一（如 `get_position(symbol)` 保持方法——正确，不动）。
- 两套指标接口边界澄清 / 内置流式指标库补齐（backtrader 50 个的对标）。
- 多周期 Bar 合成器（vnpy `BarGenerator`）。
- 成本/手数配置从 `Strategy` 裸属性迁入配置对象（架构级）。
- P2 统一事件模型、P3 本地条件单（既有路线图）。

## 8. 待确认

1. 全硬改无别名（已定）。
2. `rebalance_positions/weights` 改名（已定）。
3. 取整统一开启、撤单统一关闭（已定）。
4. 硬删 `stop_buy/stop_sell/buy_all`（已定）。
5. 风险提示：~500 处硬改、无别名 → 漏改点会在**运行时**而非 import 期暴露；缓解=每项改名的全库 grep 完备性守卫 + full suite。
