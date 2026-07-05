# 成本/手数配置收敛（cost/lot config consolidation）设计

日期：2026-07-05
状态：设计待评审
关联：[[live-broker-readiness]] Strategy API v2 roadmap 遗留项之一（"cost/lot config off Strategy bare attrs"）。

## 目标（一句话）

把 Strategy 上 6 个可写的成本/手数**裸属性**统一收敛为**单一真源、引擎注入、只读、写入即报错并指向显式配置**的属性，堵住"影子配置"陷阱与 `commission_rate`(标量)/`commission_policy`(dict) 冗余。

## 背景与问题（现状核实）

**真源**：回测成本核算 100% 在 Rust 引擎，来自 `run_backtest(commission_policy=/commission_rate=/min_commission=/stamp_tax_rate=/transfer_fee_rate=/lot_size=, ...)` 顶层参数（也可经 `BacktestConfig`/`InstrumentConfig`/broker_profile）。

**裸属性现状**：`Strategy.__new__`（`strategy.py:470-477`）设 6 个默认：
`commission_rate=0.0`、`commission_policy={"type":"percent","value":0.0}`、`min_commission=0.0`、`stamp_tax_rate=0.0`、`transfer_fee_rate=0.0`、`lot_size=1`。引擎在 init 时（`engine.py:2767-2787`）把 config 值**拷贝**到这些属性上。

**三个问题：**
1. **影子配置陷阱**：用户在策略 `__init__` 写 `self.commission_rate=0.001` 以为在设成本；但
   - 费率属性被引擎在 init 后**无条件覆盖**（`engine.py:2767-2779` 无 `None` 守卫），用户写被静默清掉；
   - 且真实核算读 Rust-from-config，根本不看该属性 → 用户的写**要么被覆盖、要么只微调买量估算**，两种都是静默错误。
2. **冗余**：`commission_rate`(旧标量) 与 `commission_policy`(dict) 并存两种说法。
3. **几乎纯影子**：唯一功能读取者是 `calculate_max_buy_qty`（`strategy_trading_api.py:926-941`），且只读 `commission_rate`/`min_commission`/`transfer_fee_rate` 三个；**`commission_policy` 与 `stamp_tax_rate` 在 Python 侧零读取**（引擎写进来后无人消费）。

**lot_size 的差异**：`lot_size` 注入有 `None` 守卫（`engine.py:2786`），且 docstring 明确 "A股回测请务必设置为 100" —— 是**文档承诺的便捷输入**，不是纯陷阱。但顶层 `run_backtest(lot_size=100)` 同样便捷，故本次**统一**处理（用户裸写 → 报错指向 `run_backtest(lot_size=)`），消除"两处能设、一处被覆盖一处不被"的不一致。

## 决策（用户已定）

- **统一**：费率 5 项 + `lot_size` 全部同一机制处理，不再拆分对待。
- **写入即报错**（raise-on-write，非 warn-then-ignore）：把静默陷阱变成响亮、可教学的失败。个人/早期项目，干净硬改优于弃用周期。

## 方案架构

**纯 Python，Rust 零改动**（config→Rust 路径不变；只改 Strategy 面与引擎注入方式）。

### 1. 单一内部真源 `self._cost_config`

`Strategy.__new__` 用一个内部 dict 承载全部成本/手数配置（替代 6 个裸属性赋值）：

```python
instance._cost_config = {
    "commission_policy": {"type": "percent", "value": 0.0},  # 规范源
    "min_commission": 0.0,
    "stamp_tax_rate": 0.0,
    "transfer_fee_rate": 0.0,
    "lot_size": 1,  # int 或 Dict[str, int]
}
```

`commission_rate`(标量) **不再独立存储**，改为从 `commission_policy` 派生的只读视图（`type=="percent"` 时取 `value`，否则 0.0）——消除冗余。

### 2. 六个只读 `@property` + 报错 setter

在 `Strategy` 上定义（与 v2 的 `cash`/`equity`/`positions` 属性同一 idiom）：

- getter：读 `self._cost_config`（`commission_rate` 从 policy 派生）。
- setter：`raise AttributeError(<指向 run_backtest/BacktestConfig 的引导信息>)`。

引导信息示例（费率）：
> `commission_policy 是回测配置项，请用 run_backtest(commission_policy={"type":"percent","value":0.0003}) 或 BacktestConfig 设置，不要写 self.commission_policy。`

引导信息示例（lot_size）：
> `lot_size 是回测配置项，请用 run_backtest(lot_size=100)（A股）或 InstrumentConfig(lot_size=) 按标的设置，不要写 self.lot_size。`

### 3. 引擎注入改私有入口

`engine.py:2767-2787` 的 `setattr(current_strategy, "commission_rate", ...)` 直写会被报错 setter 挡住，故改为调用新私有方法：

```python
strategy._inject_cost_config(
    commission_policy=commission_policy,
    min_commission=min_commission,
    stamp_tax_rate=stamp_tax_rate,
    transfer_fee_rate=transfer_fee_rate,
    lot_size=lot_size,  # None 时不覆盖既有(保留默认/上一次)
)
```

`_inject_cost_config` 更新 `self._cost_config`（`lot_size=None` 时保留原值，维持现有 `None` 守卫语义）。这是**框架内部**唯一写路径，绕过 public setter。

### 4. 消费者不变

`calculate_max_buy_qty` 继续读 `strategy.commission_rate`/`min_commission`/`transfer_fee_rate`（现在走 property getter）——数值与行为**不变**。

### 5. pickle

`_cost_config` 是纯数据（float + dict），**照常入 pickle**（不同于 P4b 的锁）。`__setstate__` 兜底：若旧 state 无 `_cost_config`，用默认重建（同 `__new__`）。

## 破坏性与迁移

**破坏面**：任何 `self.commission_rate=`/`self.commission_policy=`/`self.min_commission=`/`self.stamp_tax_rate=`/`self.transfer_fee_rate=`/`self.lot_size=` 写入 → 现抛 `AttributeError`。

**迁移**：改为 `run_backtest(...)` 顶层参数或 `BacktestConfig`/`InstrumentConfig`。已知 ~9 处写点（examples/tests/docs）：
- `examples/textbook/ch09_funds.py`
- `tests/test_close_position_delegate.py`、`test_composed_ctx_not_ready.py`、`test_execution_composed_parity.py`、`test_strategy_sizing_broker_live_guard.py`、`test_target_orders_core.py`
- 其中直接用 `Strategy.__new__(Strategy)` 后设裸属性的测试（含我方 P4b 系列不设这些，但成本相关测试会）→ 迁移为设 `_cost_config` 或经正规 config。
- docs 中的 plan 文件不需运行，按需更新示例文字。

**教材同步**（CLAUDE.md 约定）：若改到教材配套示例，需更新 `docs/zh/textbook/index.md` 的示例映射表与对应章「本章实践入口/快速运行」。

## 测试策略

- **property 只读**：读 6 项返回注入值；写任一 → `AttributeError`，信息含 `run_backtest`/config 指引。
- **commission_rate 派生**：设 `commission_policy={"percent",0.0003}` 经注入 → `commission_rate==0.0003`；非 percent → 0.0。
- **引擎注入生效**：跑一个带 `commission_policy`/`lot_size` 的 `run_backtest`，断言策略读到的值 == 传入值；`lot_size=None` 时保留默认。
- **calculate_max_buy_qty 数值不变**：同输入下与改造前结果一致（golden 或等式断言）。
- **pickle 往返**：`_cost_config` 入 snapshot、恢复一致；warm_start/checkpoint 测试绿。
- **全量回归**：`tests/` 全绿（现 1043 passed）；成本相关 golden 不漂移（漂移则 `git checkout -- tests/golden/current/`）。
- **Rust 零改动断言**：`git diff --stat <merge-base dev HEAD>..HEAD -- '*.rs' 'python/akquant/akquant.pyi'` 空。

## 非目标（YAGNI）

- 不动下单动词面（子项 A 评估为已达最佳实践，不做）。
- 不改 Rust 成本模型、不改 `run_backtest` 顶层参数签名（仅改其向 Strategy 的注入方式）。
- 不引入新的 `CostConfig` dataclass 公有类型（内部 dict 足够；如未来需要再抽）。
- 不处理 broker_live 的实盘费率（实盘成本由柜台结算，非本框架职责）。

## 收尾

- ruff `check`+`format --check` 全绿；`scripts/check_docs_links.py` passed（若改 docs）。
- 分支 `feat/cost-lot-config-consolidation`（基于 dev）；中文 Conventional Commits + `--no-verify`；未获明确要求不 push。
- 最终 opus 全分支评审后合并 dev。
