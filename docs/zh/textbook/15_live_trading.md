# 第 15 章：实盘交易系统与运维

> ⏱️ 预计阅读 ~30 分钟 ｜ 🎯 难度 ★★★★☆（核心）

量化投资的终极目标是实盘获利。从回测到实盘，不仅是代码环境的切换，更是对**系统稳定性**、**执行效率**和**风险控制**的全面考验。本章将介绍 `AKQuant` 的实盘架构，并深入探讨订单管理系统 (OMS)、风控系统 (RMS) 以及高可用部署方案。

## 学习目标

- 理解回测与实盘在数据、订单、风控和运维上的关键差异。
- 掌握网关、OMS、RMS、热启动与动态策略加载的基本作用。
- 建立准实盘与实盘迁移中的最小工程化意识。

## 前置知识

- 已掌握回测引擎、策略开发与风险管理基础。
- 了解部署、日志与监控的基本概念即可。

## 本章实践入口

- 主示例：[examples/textbook/ch15_live_trading.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch15_live_trading.py)
- 进阶示例：[examples/textbook/ch15_strategy_loader.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch15_strategy_loader.py)
- 离线实盘示例（`broker="replay"`，无需柜台）：[examples/38_live_functional_strategy_demo.py](https://github.com/akfamily/akquant/blob/main/examples/38_live_functional_strategy_demo.py)
- 行情/交易源分开指定示例（无需柜台）：[examples/39_live_mixed_broker_demo.py](https://github.com/akfamily/akquant/blob/main/examples/39_live_mixed_broker_demo.py)
- 日志/审计示例（自包含，无需网关）：[examples/66_logging_audit_demo.py](https://github.com/akfamily/akquant/blob/main/examples/66_logging_audit_demo.py)
- 外部信号接入示例（HTTP webhook，无需柜台）：[examples/61_signal_platform_webhook.py](https://github.com/akfamily/akquant/blob/main/examples/61_signal_platform_webhook.py)
- 函数式孪生示例：[examples/textbook/ch15_live_trading_functional.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch15_live_trading_functional.py)（主示例孪生）、[examples/textbook/ch15_strategy_loader_functional.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch15_strategy_loader_functional.py)（进阶示例孪生）
- 对应指南：[实盘函数式指南](../advanced/live_functional_quickstart.md)、[外部信号平台接入](../advanced/signal_ingestion.md)

## 快速运行与验收

```bash
python examples/textbook/ch15_live_trading.py
python examples/textbook/ch15_live_trading_functional.py
python examples/textbook/ch15_strategy_loader.py
python examples/textbook/ch15_strategy_loader_functional.py
python examples/38_live_functional_strategy_demo.py
python examples/39_live_mixed_broker_demo.py
python examples/66_logging_audit_demo.py
python examples/61_signal_platform_webhook.py
```

验收要点：

1. 示例可启动并完成最小实盘流程演示。
2. 日志中可观察到订单状态、网关事件或风控检查信息。
3. 调整风控参数后，策略行为变化符合预期。
4. `66_logging_audit_demo.py` 能看到：敏感字段被脱敏、订单审计单独落 JSON 文件、`language="zh"` 只改控制台审计行而文件恒英文。
5. `38_live_functional_strategy_demo.py` 与 `39_live_mixed_broker_demo.py` **无需任何柜台或可选依赖即可实跑**：前者用 `broker="replay"` 验证函数式回调链路，后者演示行情源与交易源分开指定。两者都应打印出 4 根 bar 后自行结束（不必等到 `duration` 超时）。

## 15.1 实盘架构与接口

### 15.1.1 回测与实盘的差异

| 维度 | 回测 (Backtest) | 实盘 (Live Trading) |
| :--- | :--- | :--- |
| **时间流** | 历史数据重放 (Replay) | 实时数据流 (Stream) |
| **成交机制** | 假设成交 (Perfect Fill) | 真实撮合 (Partial/Reject) |
| **延迟** | 零延迟 (Zero Latency) | 网络延迟 + 内部处理延迟 |
| **状态管理** | 内存状态 (Transient) | 持久化状态 (Persistent) |
| **缺省卖出量** | `sell()` 不传量时平总持仓 | 平**可用**持仓（T+1 当日买入部分不可卖） |
| **标的静态属性** | `InstrumentConfig` 注入，字段齐全 | 仅 `Instrument` 可回读字段，期权/结算类字段为 `None` |

前两行是数据与撮合层面的固有差异，后两行是接口层面的差异：同一段策略代码在两侧都能跑，但缺省下单量与可读的标的字段不完全一致，写策略时不要假设"回测读得到的字段实盘也读得到"。

### 15.1.2 交易接口 (Gateway)

`AKQuant` 通过适配器模式支持多种柜台接口，以便用同一套上层逻辑对接不同市场。其中 CTP (China Trading Platform) 是期货市场的标准接口，支持行情与交易链路；MiniQMT 是面向本地 A 股交易生态的适配接口入口，但当前仓库内置实现更偏向占位骨架与联调层，不应直接理解为已完成生产级实盘适配；PTrade 则是可接入券商量化终端的适配接口入口，其当前仓库内置实现同样更偏向占位骨架与联调层，不应直接理解为已完成生产级实盘适配。在实盘模式下，`DataFeed` 会切换为实时行情源，交易执行则由对应 broker gateway 负责。

需要特别区分两层含义：

1. `AKQuant` 提供的是统一交易框架与 adapter 接口；
2. 某个 broker 是否真正支持 A 股集合竞价专用委托、新股/新债打新、券商特定价格类型与业务代码，取决于该 broker adapter 是否已经补齐对应柜台语义。

因此，当前内置 `MiniQMT/PTrade` 不应被视为“开箱即用支持 A 股集合竞价与打新”的官方承诺；这类场景通常仍需通过自定义 broker 或增强现有 adapter 来落地。

CTP 交易链路支持 `execution_semantics_mode`：

*   `strict`（默认，推荐生产）：终态仅由柜台订单回报确认。
*   `compatible`：兼容旧行为，允许部分场景在本地提前推进终态。

当内置网关不满足需求时，可以通过注册机制扩展自定义 broker，且注册 broker 会被工厂优先解析，再回退到内置 `ctp/miniqmt/ptrade`。

```python
from akquant import DataFeed
from akquant.gateway import create_gateway_bundle, register_broker

register_broker("demo", demo_builder)
bundle = create_gateway_bundle(
    broker="demo",
    feed=DataFeed(),
    symbols=["000001.SZ"],
)
```

一个 broker 并不必须同时提供行情与交易两条通道：`GatewayBundle` 的
`market_gateway` 与 `trader_gateway` 是两个独立可选字段。内置的 `replay` 就只有
行情（无法下单），而某些券商/柜台插件只有交易通道（收不到行情，`on_bar` /
`on_tick` 不触发、`self.current_tick` 恒为 `None`）。

这两类「单边 broker」可以组合使用：同时指定 `market_broker`（行情源）与
`trader_broker`（交易源），各供一侧。

```python
run_live(
    strategy_cls=MyStrategy,
    instruments=instruments,
    market_broker="replay",                # 行情源
    trader_broker="my_trade_only_broker",  # 交易源
    trading_mode="paper",
    gateway_options={"bars": bars},
)
```

两侧必须**同时写明**，只给一侧会报错——若让 `broker` 兼任缺失的那侧，它就一词双义
了（读 `broker='qmf', market_broker='replay'` 得先知道「qmf 只有交易通道」才能推出
`broker` 在此指交易源）。`broker` 只用于「单个 broker 供两侧」的场景，此时语义不变。

这个组合的价值在于：既能用确定性回放数据驱动策略，又能把订单真正发往柜台仿真
环境做联调。

建议结合以下文档落地：

*   [自定义 Broker 注册](../advanced/custom_broker_registry.md)
*   [自定义 Broker 生产接入清单](../advanced/custom_broker_production_checklist.md)
*   [行情源与交易源分开指定](../reference/api.md#mixed-market-trader-broker)

### 15.1.3 离线验证实盘通路：内置 `replay` 行情源

学到这里会遇到一个现实困难：**没有柜台账号，实盘代码就没法跑**。而实盘链路恰恰是
最需要提前验证的部分——策略能否收到 bar/tick、多品种是否都到齐、`current_tick`
是否正确、订单回调是否触发。

`AKQuant` 内置了 `broker="replay"`：把一段确定性的 `Bar` / `Tick` 序列推入实盘
数据通路（`DataFeed` → 引擎 → `on_bar` / `on_tick`），无需柜台、也无需
`openctp-ctp` 等可选依赖即可实跑。

```python
from akquant import AssetType, Instrument, run_live
from akquant.akquant import Bar

run_live(
    strategy_cls=MyStrategy,
    instruments=[Instrument(symbol="DEMO_A", asset_type=AssetType.Stock, ...)],
    broker="replay",
    trading_mode="paper",
    gateway_options={"bars": bars, "freq": "1min"},   # list[Bar] / list[Tick] / DataFrame
    duration="60s",                   # 安全网
)
```

事件按时间戳升序推送，多品种全局交错；数据放完后会话**自行结束**（`replay` 通过
`metadata` 声明事件总数，引擎据此终止，无需等 `duration` 超时）。

上面顺带传了 `"freq": "1min"`：行情网关可以声明数据周期，框架把它注入策略的只读属性
`self.freq`，策略因此不必把周期写死在代码里或从外部参数重复传一遍。取值用**回测侧口径**
（`"1min"` / `"1d"`），与 `run_backtest(freq=)` 一致——klinedata 这类网关会把自己的表示法
（`period="M1"`）转换后再注入，所以同一套策略代码在回测与实盘读到的是同一个值。不声明时
`self.freq` 为 `None`（CTP 等只有逐笔源的网关、trader-only broker 都是这种情形），框架**不做
推断**，请在策略里显式处理 `None`。

要认清它的边界，别把它当"实盘彩排"的全部：

1. **只有行情，不模拟成交**（`trader_gateway=None`），因此不能用于
   `trading_mode="broker_live"`；撮合由 paper 模式的模拟执行后端负责。
2. **不覆盖 timer 语义**。回放数据带历史时间戳，而实盘引擎按墙钟判定 timer 到期，
   两条时间线错位——`on_timer` / `schedule_daily` 在回放会话中的行为不作保证，
   要验证定时任务请用回测。
3. **非正时间戳会被引擎静默丢弃**，导致声明的事件总数永远达不到、会话挂死。常见
   诱因是日期列存在无法解析的值（产出 `NaT` 进而变成非正时间戳）。构建期已有校验，
   但数据源不完全受控时，仍建议显式传 `duration` 兜底。

配套示例 `examples/38_live_functional_strategy_demo.py` 用它跑通了完整的函数式实盘
回调链路；更多细节见[实盘函数式指南](../advanced/live_functional_quickstart.md)的
「离线验证」一节。

### 15.1.4 回测 → 实盘最小切换清单

从回测走向实盘，最稳妥的路径不是"一步到位接柜台"，而是按下面这份清单逐层确认。它也帮你把"哪些是 AKQuant 已就绪的能力、哪些仍需自己补齐"分清楚。

1. **先跑 paper（模拟盘）**：同一套 `Strategy` 代码先以 `paper` 模式运行，确认信号、下单与日志链路无误，再切 `broker_live`。切勿跳过这一步直接实盘。
2. **查询执行能力再下单**：实盘前用 `self.get_execution_capabilities()` 读取 `account_mode`、`supports_short_sell`、`position_effect` 等字段，据此决定是否启用做空、`close_today` 等语义，避免回测能跑、实盘被拒。
3. **数据源切换**：把历史重放的 `DataFeed` 换成实时行情源，由对应 broker gateway 驱动。若策略逻辑依赖数据周期，用只读属性 `self.freq` 读取（回测由 `run_backtest(freq=)` 注入，实盘由行情网关声明，两侧口径统一），不要把周期写死或另传一份参数；网关未声明时它是 `None`，需显式处理。
4. **网关选型要清醒**：内置 `MiniQMT/PTrade` 当前更偏占位骨架与联调层，**不应视为已完成生产级 A 股适配**；集合竞价专用委托、打新等场景通常需自定义 broker 或增强 adapter（见 15.1.2 与《自定义 Broker 注册/生产接入清单》）。
5. **成交语义从严**：CTP 链路使用 `execution_semantics_mode="strict"`（默认、推荐生产）——撤单/拒单/成交等**终态一律以柜台 `OnRtnOrder` 回报为准**，不要凭本地请求成功就推进状态（详见 15.2.2）。
6. **风控前置必须开**：实盘务必显式配置 RMS 前置风控（单笔最大委托量、资金使用率、日内撤单次数、策略级止损），它是防"乌龙指"的最后一道防线（见 15.3）。
7. **状态可恢复**：用 `save_checkpoint` 定期落盘、`run_from_checkpoint` 重启后续跑，保证宕机后"断点续传"（见 15.5.4、15.6.2）。
8. **可观测性到位**：启动前用 `akquant.configure_logging(LogConfig(profile="live", file_json=True, ...))` 打开结构化日志，并接入监控告警（见 15.5.2、15.8）。

一句话原则：**先 paper 后实盘、先查能力后下单、终态以柜台回报为准、风控与可观测性先于收益。**

## 15.2 订单管理系统 (Order Management System, OMS)

OMS 是实盘交易的核心，负责维护订单的全生命周期状态。

### 15.2.1 订单状态机

实盘中的订单状态远比回测复杂，常见状态包括：

1.  **New**：策略已创建订单。
2.  **Submitted**：订单已提交到交易通道。
3.  **Accepted**：柜台/交易所确认接收。
4.  **PartiallyFilled**：部分成交。
5.  **Filled**：全部成交。
6.  **Cancelled**：已撤单。
7.  **Rejected**：废单（如资金不足、不在交易时间、风控拒绝）。

### 15.2.2 状态同步 (Synchronization)

策略持仓 (`Strategy Position`) 与柜台持仓 (`Broker Position`) 可能因网络丢包或人工干预而不一致。

*   **定时同步**：每隔 N 秒查询柜台持仓，强制覆盖本地状态。
*   **事件驱动**：通过 `on_order`、`on_trade`（以及可选 `on_broker_event`）实时更新状态并做审计落盘。

在 CTP 严格模式下，建议遵循以下判定：

1.  发送撤单请求成功 ≠ `Cancelled`，必须等待 `OnRtnOrder(Cancelled)`。
2.  收到报单错误 ≠ `Rejected`，应以最终 `OnRtnOrder` 状态为准。
3.  `Filled` 以订单回报终态确认，成交回报用于补充成交明细与审计。

## 15.3 风险管理系统 (Risk Management System, RMS)

在实盘中，**风控前置 (Pre-trade Risk Check)** 是防止“乌龙指”的最后一道防线。

### 15.3.1 核心风控规则

1.  **单笔最大委托量 (Max Order Size)**：防止代码错误导致的天量下单。
2.  **资金使用率限制 (Margin Usage Limit)**：防止满仓操作，预留安全垫。
3.  **日内撤单次数限制**：交易所对频繁撤单有惩罚措施（如上期所 500 次）。
4.  **策略级止损**：当策略当日亏损超过 N% 时，强制平仓并停止运行。

## 15.4 算法交易 (Algorithmic Execution)

对于大资金，直接下单会产生巨大的**冲击成本 (Market Impact)**。算法交易旨在拆解大单，降低成本。

### 15.4.1 TWAP (Time Weighted Average Price)

时间加权平均价格算法的思路是将大单均匀拆分到一段时间内执行。在逻辑上，它每隔 $t$ 秒下单 $q$ 手；正因为拆单节奏固定，它更适用于流动性均匀的市场。

### 15.4.2 VWAP (Volume Weighted Average Price)

成交量加权平均价格算法则根据历史成交量分布来调整节奏，在流动性好的时候多下单、流动性差的时候少下单，其目标是让成交均价尽量接近市场 VWAP。

## 15.5 实盘部署与运维

### 15.5.1 部署环境

在部署环境上，云服务器 (ECS) 推荐使用靠近交易所机房的节点（如上海、深圳）以降低延迟；而 Docker 容器化则用来确保实盘环境与测试环境完全一致，从而避免 "It works on my machine" 这类环境差异问题。

### 15.5.2 监控与报警

监控与报警通常由三层手段共同支撑。心跳监测 (Heartbeat) 负责确保程序存活；日志 (Logging) 负责详细记录每一笔 Tick、Signal 和 Order；消息推送则集成钉钉/飞书/邮件机器人，实时推送成交和异常信息。

推荐在实盘或准实盘启动前显式打开日志，而不是依赖默认输出：

```python
import akquant

akquant.configure_logging(
    akquant.LogConfig(
        profile="live",
        level="INFO",
        console=True,
        filename="logs/live.log",
        file_level="DEBUG",
        file_json=True,
        file_max_bytes=50_000_000,
        file_backup_count=5,
    )
)
```

这样做有几个直接收益：

*   `on_order` / `on_trade` / `on_reject` 中的策略日志会自动携带 `order_id`、`client_order_id`、`strategy_id`、`symbol` 等结构化字段。
*   网关与执行链路中的 warning 也会进入同一套日志管线；例如拒单、未知撤单、收盘过期、严格语义下终态尚未确认等问题，都更容易统一排查。
*   如果打开 `file_json=True`，后续接入日志平台、告警系统或审计落盘会更顺手。

**订单审计（可脱机复盘的凭证）。** 在 `broker_live` 下，每一笔订单的提交 / 回报 / 成交 / 撤单 / 拒单都会经 `akquant.audit.order` 命名空间产出结构化 INFO 审计。设置 `order_audit_file` 后，这些审计还会以 JSON line 单独落到一份纯审计文件——**进程停止后仍可仅凭它重建订单全生命周期，用于对账与事故复盘**：

```python
akquant.configure_logging(
    akquant.LogConfig(
        profile="live",
        console=True,
        console_level="WARNING",          # 控制台只留需人关注的拒单/断连
        filename="logs/live.log",
        file_level="INFO",
        order_audit_file="logs/orders_audit.log",  # 逐笔订单审计 JSON 流
        order_audit_level="INFO",
    )
)
```

高频策略实盘时强烈建议这样分流：控制台 `WARNING` 保持清爽，逐笔 INFO 审计单独落盘，避免刷屏又不丢凭证。

**实盘柜台拒单不抛异常**

柜台明确回绝一笔报单时（如「证券可用数量不足」「委托价不符合最小变动单位」），
`buy()` / `sell()` / `order_target_*` **不会抛异常**，而是返回空回执并触发
`on_reject`，与回测口径一致。请用 `on_reject` 感知拒单，不要依赖 `try/except`。

若因超时或连接断开导致**订单状态不可知**（报文可能已到柜台），框架不会谎报拒单：
此时触发的是 `on_error(exc, "order_submit", request)`，并在审计流水里记一条
`order_submit_unknown`。这笔单的真实状态由下一轮 `sync_open_orders` 对账浮出。

**例外：本地止损单的重试口径按失败原因区分，而非"一律交给对账"。** 止损/跟踪止损单
不下发柜台，由框架本地盯价触发后再调用 `submit_order` 报出（见
`python/akquant/gateway/broker_execution.py` 的 `check_stop_triggers` /
`_handle_stop_submit_failure`），触发提交失败时按 `OrderReceipt.failure` 分流：

- **柜台明确拒单**（`failure == "rejected"`，订单确定不存在）：仍会重试，最多
  `MAX_STOP_SUBMIT_ATTEMPTS = 3` 次；每次重试都经 `submit_order` 生成一个**新的**
  `client_order_id`，但因为订单确定未被柜台接受，这不构成重复委托。
- **状态未知**（`failure == "unknown"`，超时/断连，报文可能已到柜台）：框架**不再自动
  重试**——重试会生成新的 `client_order_id`，柜台无从去重，一旦报文其实已经到达柜台，
  重试就是一笔真实的重复委托。因此该止损单会被直接放弃，打一条 CRITICAL 日志并触发
  `on_error(exc, "stop_trigger", order)`，需要人工介入核实。

**已知局限**：被放弃的这张止损单，如果报文其实已经到达柜台，其回报**无法**被自动关联
回本地止损 id——`_record_stop_remap`（`python/akquant/live/_runner.py`）只在提交
**成功**、拿到 `broker_order_id` 时才建立 `broker_order_id → local_id` 的映射；状态未知
时根本没有 `broker_order_id` 可供记录。这笔回报之后会以一笔"来源不明"的普通订单面貌
到达策略，而不会被识别成某张本地止损单的回报，需要人工在对账时核实。

**敏感脱敏（默认开启）。** 日志默认对密钥类字段（`password`/`token`/`api_key` 等）全掩码、对账户类字段（`user_id`/`account` 等）保留尾 4 位。这是 handler 层兜底——即便某处新增日志忘了脱敏，账号密钥也不会明文落盘；如需关闭设 `mask_sensitive=False`。

**日志语言。** 日志消息默认英文（可搜索、可协作、可被告警/日志系统消费的通用契约），结构化字段（`event`/`side`/`price`）也恒为英文。如偏好中文控制台，设 `language="zh"`——它只把**控制台的订单审计行**渲染成中文，**文件与 JSON 审计流仍是英文**，因此 grep/对账/告警不会因语言分裂。`CRITICAL` 用于交易前置断连、runner 崩溃等系统级致命事件，建议单独接告警通道。

### 15.5.3 代码示例：启动实盘

```python
--8<-- "examples/textbook/ch15_live_trading.py"
```

推荐进一步查看以下实盘脚本：

*   `examples/38_live_functional_strategy_demo.py`：函数式策略入口（paper / broker_live）。
*   `examples/39_live_broker_submit_order_demo.py`：`broker_live` 下最小下单闭环。
*   `examples/42_live_broker_event_audit_demo.py`：统一 broker 事件审计与策略归属追踪。
*   `examples/35_custom_broker_registry_demo.py`：自定义 broker 注册与工厂接入。

### 15.5.4 热启动与状态持久化 (Warm Start)

在准实盘/长会话回放场景中，系统可能会因网络波动或维护重启。为了保证策略状态（如指标缓存、持仓记录）不丢失，`AKQuant` 提供了**热启动**机制。

**1. 保存状态 (Checkpoint)**

在每日收盘后或定期调用 `save_checkpoint`：

```python
import akquant as aq
# 保存当前引擎状态和策略变量
aq.save_checkpoint(engine, strategy, "strategy_checkpoint.pkl")
```

**2. 恢复运行 (Restore)**

系统重启后，使用 `run_from_checkpoint` 加载快照并注入新的数据源：

```python
# 加载最新的数据源 (包含历史数据 + 今日新数据)
data_feed = aq.CSVFeedAdapter(path_template="latest_data_{symbol}.csv")

engine_result = aq.run_from_checkpoint(
    checkpoint_path="strategy_checkpoint.pkl",
    data=data_feed,
    symbols="rb2310",
)

# 获取恢复后的引擎和策略
engine = engine_result.engine
strategy = engine_result.strategy
```

`run_from_checkpoint` 会恢复 checkpoint 中的策略实例，不会通过 `strategy_source/strategy_loader` 重新加载策略实现。

### 15.5.5 动态策略加载 (Strategy Loader)

在实盘与准实盘场景中，策略实现有时需要按运行时配置动态加载，而不是在脚本中静态 `import`。`AKQuant` 支持通过 `strategy_source + strategy_loader` 机制完成策略注入。

下面示例演示了两种加载方式：

1.  `python_plain`：从源码文件按类名加载策略。
2.  `encrypted_external`：由外部回调解密并返回策略类。

```python
--8<-- "examples/textbook/ch15_strategy_loader.py"
```

当你需要将“策略参数 + 策略代码来源 + 运行模式”统一交给调度平台管理时，这条路径比手工改脚本更稳健。完整参数说明可结合《运行时配置指南》一起使用。

你也可以使用通用示例 `examples/44_strategy_source_loader_demo.py` 作为最小验证入口，先在回测中验证策略装载链路，再切换到实盘调度。

## 15.6 高可用架构 (High Availability)

实盘系统最怕的不是亏损，而是**宕机**。一旦系统崩溃，持仓状态丢失，正在进行的订单无法撤销，后果不堪设想。

### 15.6.1 主备切换 (Primary-Backup)

构建两套完全相同的系统：

1.  **主机 (Master)**：负责接收行情、计算信号、发送订单。
2.  **备机 (Slave)**：实时接收行情和主机状态，但不发单。
3.  **心跳 (Heartbeat)**：主机每秒向备机发送心跳包。
4.  **切换 (Failover)**：当备机连续 N 秒未收到心跳，判定主机宕机，自动接管交易权限，并报警通知人工介入。

### 15.6.2 状态持久化 (Persistence)

内存中的状态（持仓、订单、信号）必须实时落地到数据库（如 Redis AOF 或 SQLite）。

*   **Crash Recovery**：程序重启后，首先读取数据库恢复现场，确保“断点续传”。

## 15.7 低延迟优化 (Low Latency)

对于高频交易 (HFT)，速度就是利润。

低延迟优化往往沿着从物理链路到软件细节的层层下探。最外层是共置 (Co-location)，即把服务器托管在交易所机房（如上交所金桥数据中心），光纤直连，将物理距离缩短至米级，延迟可从毫秒级 (ms) 降至微秒级 ($\mu s$)。再往里是内核旁路 (Kernel Bypass)，它使用 Solarflare 网卡和 OpenOnload 技术绕过操作系统内核，直接在用户态处理网络包，从而减少上下文切换。继续深入到处理器层面，则可借助 CPU 亲和性 (CPU Affinity)，把交易进程绑定到特定的 CPU 核心，独占 L1/L2 缓存以避免缓存失效 (Cache Miss)。最后落到代码层面的是无锁编程 (Lock-free)，即在 C++ 或 Rust 中使用原子操作 (Atomic) 和无锁队列 (Ring Buffer) 替代互斥锁，避免线程阻塞。

## 15.8 监控体系 (Monitoring Stack)

仅仅有日志是不够的，我们需要可视化的仪表盘。

1.  **Prometheus**：时序数据库，采集系统指标。
    *   `strategy_latency`: 策略计算耗时。
    *   `order_latency`: 订单往返延时 (RTT)。
    *   `position_exposure`: 当前持仓敞口。
    *   `pnl_realtime`: 实时盈亏。
2.  **Grafana**：可视化展示。配置大屏，实时显示资金曲线、持仓分布、系统负载。
3.  **AlertManager**：报警中心。
    *   **P0 级报警**：程序崩溃、网络断开、资金不足。电话通知。
    *   **P1 级报警**：策略亏损超限、未成交订单过多。短信通知。
    *   **P2 级报警**：延迟抖动、CPU 高负载。邮件通知。

## 15.9 实盘事故复盘 (Post-Mortem)

前车之鉴，后事之师。

两起经典事故从不同侧面印证了前面所讲的原则。光大乌龙指 (2013) 是策略系统错误生成巨量市价单，且缺乏资金校验风控，导致瞬间买入 234 亿元股票，拉升上证指数 5%；它留下的教训是，风控系统必须独立于交易系统，且拥有最高权限（“熔断机制”）。骑士资本 (2012) 则是由于部署失误，旧代码被错误激活，在 45 分钟内疯狂买卖，亏损 4.4 亿美元，最终导致公司破产；它的教训在于，灰度发布和自动化部署是生命线，新代码上线前必须在模拟盘 (Paper Trading) 充分验证。

## 15.10 硬件加速 (Hardware Acceleration)

当通用 CPU 的性能达到瓶颈时，我们需要借助专用硬件。

### 15.10.1 FPGA (Field-Programmable Gate Array)

FPGA 允许直接在硬件电路层面编程，将网络包处理、行情解析、订单构建等逻辑烧录到芯片中。正因如此，它的延迟可达亚微秒级 (Sub-microsecond)，从接收行情到发出订单仅需 500ns，主要应用于做市商 (Market Making) 与高频套利等场景；但代价是开发成本极高，需要使用 Verilog/VHDL 语言，调试也相当困难。

### 15.10.2 GPU (Graphics Processing Unit)

GPU 擅长大规模并行计算，因此常用于深度学习训练 (Training) 与大规模期权定价 (Monte Carlo)。不过它也有明显限制：由于 PCIe 总线的延迟，GPU 不适合处理对延迟极度敏感的即时交易逻辑，更适合盘中实时计算复杂的因子或风险指标。

## 15.11 外部信号接入 (External Signal Ingestion)

前面各节假定信号在 AKQuant 内部由策略产生。但工业界另一种常见分工是：信号在**外部平台**生成（因子、择时、组合优化都在那边），执行系统只负责接收指令、风控、下单、审计。这一节讨论这种形态。

### 15.11.1 信号是指令，不是事件

理解这个区分是设计的关键。策略回调收到的 `Bar` 是**事件** —— 需要策略判断该不该交易；而信号平台推来的是**指令** —— 判断已经做完了。

因此外部信号**不经过策略回调**，直接成为委托。让它再过一次策略决策是多余的一跳，还会把「外部指令」与「自有信号」的责任边界搅混。

```python
from akquant import run_live
from akquant.signal import QueueSignalSource, Signal

source = QueueSignalSource()
source.put(Signal(
    signal_id="platform-0001",   # 幂等键
    symbol="000001.SZ",
    action="buy",
    quantity=100,
    price=10.5,
))

run_live(instruments=[...], broker="ctp",
         trading_mode="paper", signal_source=source)
```

### 15.11.2 幂等：分布式系统的必修课

信号平台重连、重启、网络重试都会重推同一条指令。**没有幂等键就意味着每次重推都是一笔新委托** —— 这是这类系统最常见的事故来源。

`signal_id` 就是幂等键。框架按它去重，其中两条边界值得推敲：

| 情形 | 是否放开去重标记 | 理由 |
|---|---|---|
| 投递抛异常（网关挂了） | ✅ 放开 | 那一笔并未真正下单，平台重推应被受理 |
| 同步返回未下单（风控拦下） | ❌ 不放开 | 指令已判定不可执行，重推只会再被拒一次 |

这个区分体现了一个通用原则：**幂等的边界应画在"副作用是否已发生"上**，而不是"调用是否成功"上。

### 15.11.3 回执：不告诉对端等于骗它

平台推完指令后需要知道它有没有生效。缺了回执，被风控拒掉的信号在平台侧看起来是"下成功了"，持仓账本随即不一致。

框架把四种结果回吐给信号源：`accepted`（已投递，**不代表已成交**）、`duplicate`、`rejected`、`error`。柜台的**异步**拒单也会经 `on_reject` 反查回原始 `signal_id` 后回执。

### 15.11.4 安全：能下单的网络端点

`HttpSignalSource` 是一个能触发真实下单的 HTTP 入口。它的安全约束是**硬性**的：token 必填（构造即校验）、默认只听 `127.0.0.1`（绑其他地址直接报错）、可选 HMAC 签名 + 时间戳窗口防重放、鉴权失败统一回 `401` 不做区分预言机。

这些取舍对标 Freqtrade：其 `force_entry_enable` 默认关闭、只听 localhost、官方明确警告勿暴露公网。**AKQuant 同样不承诺传输层安全** —— HTTPS 与公网暴露须在反向代理层解决。

### 15.11.5 ⚠️ 两种模式的风控覆盖面不同

这是**引擎架构决定的事实，不是配置项**，上线前必须知道：

| 限额 | `paper` | `broker_live` |
|---|---|---|
| `max_order_value` / `max_order_size` / `max_position_size` | ✅ | ✅ |
| `max_daily_loss` / `max_drawdown` / `strategy_risk_budget` | ✅ | ❌ **不生效** |

原因：引擎的实盘执行器不向柜台报单，故 broker_live 的订单走 Python 侧柜台通道，那条路上只有策略级三项限额做了前置校验。

**实践含义**：先在 `paper` 下用真实信号流验证限额配置，再切 `broker_live`，并在平台侧或柜台侧补上账户级熔断。这与 15.1.4 的切换清单是同一套纪律。

### 15.11.6 部署形态

对标 vn.py 的取舍（WebTrader 是独立进程 + RPC），生产推荐**进程分离**：

```
平台 ──HTTPS──> 反向代理 ──> 收单进程 ──Redis Stream──> 交易进程
```

HTTP 服务的故障与负载被进程边界隔开，不会波及交易主循环。开发单机可用 `HttpSignalSource` 同进程直收；测试用 `QueueSignalSource`，无网络依赖。

详见[外部信号平台接入指南](../advanced/signal_ingestion.md)。

## 15.12 量化团队协作 (Team Collaboration)

量化交易不再是单打独斗的时代，而是一个工业化的流水线。

1.  **基金经理 (PM)**：制定顶层投资逻辑，管理投资组合风险，对最终盈亏负责。
2.  **量化研究员 (Quant Researcher)**：挖掘因子，构建模型，撰写研究报告 (Jupyter Notebook)。
3.  **量化开发 (Quant Developer)**：
    *   **平台开发**：维护回测引擎 (`AKQuant`)、数据清洗管线。
    *   **策略开发**：将研究员的 Python 代码重构为高性能的 C++/Rust 实盘代码。
4.  **数据工程师 (Data Engineer)**：负责大数据的采集、存储和清洗。
5.  **运维 (SRE)**：负责服务器维护、网络监控、故障排查。

## 本章小结

### 必须掌握

- 实盘系统的核心不只是策略收益，而是网关、OMS、RMS 与运维闭环的稳定性。
- 热启动、审计日志和监控告警决定了策略是否能长期运行。

### 理解即可

- 高可用、低延迟、硬件加速与团队协作属于行业拓展，应放在主线之后理解。

### 实践提醒

- 先完成可复现的准实盘闭环，再讨论更高阶的生产级实盘话题。

## 主线推进

贯穿全书的那条最小多均线 / 趋势策略，到本章终于走出回测环境，迈向准实盘与实盘运维这一步。在此之前，它已经是一个事件驱动的标准策略类，经历了多资产扩展、严谨的绩效评价与风险约束；但它始终运行在历史数据重放之上，享受着零延迟与完美成交的理想假设。本章把这条主线接入真实的工程闭环：通过统一网关与 adapter 接口面对真实撮合与拒单，用 OMS 维护订单从 New 到 Filled/Rejected 的完整状态机，用前置风控 (RMS) 为它装上“乌龙指”的最后一道防线，再借助热启动与状态持久化让它能在重启后“断点续传”。与此同时，结构化日志、监控告警、主备切换与算法执行（TWAP/VWAP），把它从“一段能跑出收益的代码”推进为“一个可以被运维、被监控、可长期存活”的准实盘系统。至此，主线策略已经具备从模拟盘走向实盘的工程骨架，只待在真实柜台语义与生产级 broker adapter 补齐后完成最后一跃。

## 延伸阅读

**经典著作**

- Aldridge, I. *High-Frequency Trading: A Practical Guide to Algorithmic Strategies and Trading Systems*（第 2 版），John Wiley & Sons, 2013 —— 系统讲解高频交易的策略类型、交易系统架构、延迟与执行成本，是理解本章 15.4（算法交易）、15.7（低延迟）与 15.10（硬件加速）的实务读本。
- Johnson, B. *Algorithmic Trading and DMA: An Introduction to Direct Access Trading Strategies*，4Myeloma Press, 2010 —— 从市场微观结构出发，详尽讲解 TWAP、VWAP 等执行算法与直接市场接入 (DMA) 的机制，对应本章 15.1（网关接口）与 15.4（TWAP/VWAP）。
- Harris, L. *Trading and Exchanges: Market Microstructure for Practitioners*，Oxford University Press, 2003 —— 市场微观结构的权威实务专著，解释撮合、流动性供给与交易成本的来源，为本章 15.1.1（回测与实盘差异）与算法交易的冲击成本提供理论背景。
- Beyer, B., Jones, C., Petoff, J., & Murphy, N. R.（编）*Site Reliability Engineering: How Google Runs Production Systems*，O'Reilly Media, 2016 —— Google SRE 团队的实践合集，系统讨论监控、告警分级、心跳与故障恢复等运维原则，对应本章 15.5（部署与运维）、15.6（高可用）与 15.8（监控体系）。

**官方文档与工具**

- [AKQuant 实盘函数式指南](../advanced/live_functional_quickstart.md) —— `paper / broker_live` 运行模式、网关与下单闭环的权威说明，对应本章 15.1 与 15.5.3。
- [自定义 Broker 注册](../advanced/custom_broker_registry.md) 与 [自定义 Broker 生产接入清单](../advanced/custom_broker_production_checklist.md) —— 扩展自定义柜台接口与生产落地核对的官方文档，对应本章 15.1.2。
- [Prometheus 官方文档](https://prometheus.io/docs/) 与 [Grafana 官方文档](https://grafana.com/docs/) —— 时序指标采集与可视化大屏的工具文档，对应本章 15.8（监控体系）。

**本书相关**

- [第 4 章：事件驱动回测原理](04_backtest_engine.md) —— 本章 15.1.1 对比的“回测 vs 实盘”，正是建立在第 4 章事件驱动引擎之上的延伸。
- [第 10 章：策略评价体系与风险指标](10_analysis.md) —— 本章 15.3 的实盘前置风控 (RMS) 所约束的风险敞口，正是第 10 章风险指标在实盘链路上的工程化前置。

## 课后练习

### 基础题

1. 启动一个最小实盘或准实盘示例，记录关键日志字段。

### 应用题

1. 模拟一次异常中断并验证热启动恢复流程。

1. 用 `QueueSignalSource` 接一路外部信号，投递两次相同 `signal_id`，观察回执与实际成交笔数。

### 综合题

1. 设计一份包含网关、OMS、RMS 与监控项的最小上线检查表。
2. 你的信号平台把指令推给两个 AKQuant 实例（主备）。设计一套机制，保证主备切换时不出现重复下单，也不漏单。说明你依赖了哪些前提。

??? note "参考答案要点（先独立思考再展开）"

    **基础题**：以 `paper` 模式启动，调用 `configure_logging(..., file_json=True)`，记录 `order_id`、`client_order_id`、`strategy_id`、`symbol` 等结构化字段。

    **应用题**：`save_checkpoint` 落盘 → 模拟中断 → `run_from_checkpoint` 加载快照并注入新数据源，验证持仓与指标缓存恢复一致、无重复下单。

    **应用题 2**：两次投递后回执应为一条 `accepted` 与一条 `duplicate`，实际成交仅一笔。若两笔都成交，检查是否漏传 `signal_id`（缺幂等键时框架无法去重，只能按"每次都是新信号"处理）。

    **综合题 1**：参见 15.1.4 的切换清单——paper 验证、能力查询、CTP strict 终态、RMS 前置风控、热启动、监控告警、灰度发布。

    **综合题 2**：关键是把幂等状态从进程内移到共享存储：`signal_id` 已处理集合放 Redis（而非各自的进程内 `SignalDedup`），配合 `SETNX` 式的原子占位，使同一 id 只有一个实例能领走。不漏单则依赖 Redis Stream 消费组的 pending 语义——未 ack 的消息在实例崩溃后仍可被另一实例 `XCLAIM` 接管。须说明的前提：① 共享存储自身高可用，否则它成了新的单点；② 下单与"标记已处理"无法真正原子，故仍需对账兜底（比对柜台委托与信号台账），这是分布式系统的固有限制而非实现缺陷。

## 常见错误与排查

1. 订单状态不同步：检查本地状态与柜台回报对账流程。
2. 异常延迟增大：排查网络链路、消息积压和策略阻塞代码。
3. 实盘风险失控：核对仓位限制、熔断阈值和报警通道是否生效。
