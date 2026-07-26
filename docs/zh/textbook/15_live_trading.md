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
- 日志/审计示例（自包含，无需网关）：[examples/66_logging_audit_demo.py](https://github.com/akfamily/akquant/blob/main/examples/66_logging_audit_demo.py)
- 对应指南：[实盘函数式指南](../advanced/live_functional_quickstart.md)

## 快速运行与验收

```bash
python examples/textbook/ch15_live_trading.py
python examples/textbook/ch15_strategy_loader.py
python examples/66_logging_audit_demo.py
```

验收要点：

1. 示例可启动并完成最小实盘流程演示。
2. 日志中可观察到订单状态、网关事件或风控检查信息。
3. 调整风控参数后，策略行为变化符合预期。
4. `66_logging_audit_demo.py` 能看到：敏感字段被脱敏、订单审计单独落 JSON 文件、`language="zh"` 只改控制台审计行而文件恒英文。

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

建议结合以下文档落地：

*   [自定义 Broker 注册](../advanced/custom_broker_registry.md)
*   [自定义 Broker 生产接入清单](../advanced/custom_broker_production_checklist.md)

### 15.1.3 回测 → 实盘最小切换清单

从回测走向实盘，最稳妥的路径不是"一步到位接柜台"，而是按下面这份清单逐层确认。它也帮你把"哪些是 AKQuant 已就绪的能力、哪些仍需自己补齐"分清楚。

1. **先跑 paper（模拟盘）**：同一套 `Strategy` 代码先以 `paper` 模式运行，确认信号、下单与日志链路无误，再切 `broker_live`。切勿跳过这一步直接实盘。
2. **查询执行能力再下单**：实盘前用 `self.get_execution_capabilities()` 读取 `account_mode`、`supports_short_sell`、`position_effect` 等字段，据此决定是否启用做空、`close_today` 等语义，避免回测能跑、实盘被拒。
3. **数据源切换**：把历史重放的 `DataFeed` 换成实时行情源，由对应 broker gateway 驱动。
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

## 15.11 量化团队协作 (Team Collaboration)

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

### 综合题

1. 设计一份包含网关、OMS、RMS 与监控项的最小上线检查表。

??? note "参考答案要点（先独立思考再展开）"

    **基础题**：以 `paper` 模式启动，调用 `configure_logging(..., file_json=True)`，记录 `order_id`、`client_order_id`、`strategy_id`、`symbol` 等结构化字段。

    **应用题**：`save_checkpoint` 落盘 → 模拟中断 → `run_from_checkpoint` 加载快照并注入新数据源，验证持仓与指标缓存恢复一致、无重复下单。

    **综合题**：参见 15.1.3 的切换清单——paper 验证、能力查询、CTP strict 终态、RMS 前置风控、热启动、监控告警、灰度发布。

## 常见错误与排查

1. 订单状态不同步：检查本地状态与柜台回报对账流程。
2. 异常延迟增大：排查网络链路、消息积压和策略阻塞代码。
3. 实盘风险失控：核对仓位限制、熔断阈值和报警通道是否生效。
