# RFC:日志系统完善与实盘审计

> **状态**:全部落地(G1–G5) · **日期**:2026-07-20 · **范围**:AKQuant 核心(Python `log.py` + gateway 实盘链路 + Rust 桥接),偏增量演进、保持向后兼容
>
> **进度**:✅ G2 敏感信息脱敏(`SensitiveFilter`,默认开启) · ✅ G1 实盘订单审计(`gateway/order_audit.py` + `LogConfig.order_audit_file`) · ✅ G3 print→logging 收敛 · ✅ G4 级别语义(CRITICAL + Rust error 分级 + 级别语义表) · ✅ G5 贯穿式 trace_id
>
> 本文源于用户反馈「日志好像不够完善」。经全仓盘点,结论是:**日志基础设施(中央封装、结构化、Rust 桥接、多进程聚合)已相当成熟,真正的缺口集中在「实盘审计」这一量化特有场景**。本 RFC 把评估固化为可分期实施与评审的跟踪基线。对标实现:**nautilus_trader**(`Logger` + 每笔订单事件落盘审计)、**vnpy**(`OmsEngine` 订单/成交流水 + `LogEngine`)、**Zipline / backtrader**(标准 logging 分层)。

---

## 0. 背景与动机

一次针对「日志是否符合量化框架最佳实践」的全仓盘点得出两个基本事实:

1. **通用日志基础设施做得好**,不应整体推翻:自建中央模块 `python/akquant/log.py`(619 行)已覆盖库静默、分层命名、JSON 结构化、上下文字段、文件轮转、Rust→Python 桥接、多进程队列聚合、profile。
2. **实盘链路的可审计性缺失**,这才是「不够完善」的真实来源:订单的提交/回报/成交/撤单/拒单在**正常流转时没有任何 INFO 级审计日志**,进程停止后没有落盘凭证可供对账、复盘、追责。

量化框架与通用框架的分水岭恰恰在于此——**资金安全与合规复盘要求每一笔订单生命周期都可脱机重建**。因此本 RFC 的优先级排序以「实盘审计 + 敏感信息安全」为最高。

## 1. 现状盘点

### 1.1 已符合最佳实践(保留,不动)

| 实践 | 位置 |
|---|---|
| 库默认静默(`NullHandler`) | `log.py:339-346` |
| 统一根 logger + 分层命名(`akquant.*`) | `log.py:14` |
| 结构化日志(JSON 格式化 + 固定上下文字段集) | `log.py:158-183`、`log.py:16-25` |
| 文件轮转(`RotatingFileHandler`) | `log.py:441-451` |
| Rust→Python 桥接(`pyo3-log` + `[akq_ctx=...]` 上下文回传) | `src/lib.rs:47`、`src/log_context.rs` |
| 多进程(优化器)日志聚合(`QueueHandler`/`QueueListener`) | `optimize.py:380`、`optimize.py:718` |
| 三档 profile(research/optimize/live) | `log.py:186-205` |
| 构造结构化 `extra` 的辅助函数 `build_log_extra` | `log.py:221-242` |

### 1.2 缺口清单(本 RFC 的改造对象)

| 编号 | 缺口 | 严重性 | 现状位置 |
|---|---|---|---|
| **G1** | 实盘订单生命周期无 INFO 审计日志 | 🔴 P0 | `broker_event_bridge.py`、`order_submitter.py` |
| **G2** | 敏感信息明文 + 无统一脱敏层 | 🟠 P0 | `ctp/native.py:422-428,1007-1013` |
| **G3** | 一致性破损:`print` 与绕过中央封装 | 🟠 P1 | `ml/*`、`factor/engine.py`、`data.py`、`risk.py` |
| **G4** | 级别语义不完整(无 `critical`;Rust 热路径仅 `warn`) | 🟡 P1 | 全仓、`src/execution/*` |
| **G5** | 缺少贯穿全链路的关联 ID(trace/correlation id) | 🟡 P2 | `build_log_extra` 上下文字段 |

## 2. 目标与非目标

**目标**

- 实盘每一笔订单的**提交→回报→成交→撤单→拒单**都有一条带完整上下文的 INFO 审计日志,可脱机重建全生命周期。
- 敏感字段(账户号、密钥、token)有**格式化器级兜底脱敏**,新增日志语句默认安全。
- 全仓日志走**同一个中央入口**,`print` 从库代码内消除(报表/绘图的用户可见输出除外)。
- 关键级别语义补齐:系统级致命事件用 `CRITICAL`;Rust 执行路径能分级观测。
- 引入贯穿式 `trace_id`,把「信号→下单→回报→成交→持仓」串成一条可检索时间线。

**非目标**

- **不推翻现有 `log.py` 架构**,本 RFC 是增量演进,非重写。
- **不引入第三方日志库**(structlog/loguru):现有自建 JSON + 上下文已够用,新增依赖得不偿失。
- 不改变库默认静默契约(用户不配置则无输出)。
- 不做集中式日志采集/上报(ELK/Loki)——落盘审计文件为界,采集交由部署层。

## 3. 设计原则

| 原则 | 含义 |
|---|---|
| 审计优先 | 订单生命周期审计是实盘一等公民,默认开启、结构化、可落盘 |
| 安全兜底 | 脱敏在 formatter/filter 层生效,不依赖调用点自觉 |
| 单一入口 | 库内一律 `akquant.get_logger`,享受 profile 与上下文 |
| 增量兼容 | 不破坏现有 API 与默认静默;新能力经 `LogConfig` 开关 |
| 可关联 | 一个 `trace_id` 贯穿跨模块、跨 Rust/Python 边界 |

## 4. 分项改造点

### G1 — 实盘订单审计日志(🔴 P0)

**现状**:核心事件分发路径 `broker_event_bridge.py::_emit_observer_event`(`:106-133`)在 order/trade/execution_report/account 正常流转时**一行日志都不打**,仅 observer 回调抛异常才 `WARNING`(`:121`)。提交路径 `order_submitter.py` 同样只在「拒绝重复 client_order_id」时 `WARNING`(`:210`)。gateway 目录对 `submit/fill/trade/cancel` 的 `logger.info/debug` 检索零命中。订单审计当前只能靠内存事件流(`examples/42_live_broker_event_audit_demo.py`)观察,进程停止即丢失。

**问题**:无落盘凭证 → 无法事后对账、无法复盘成交时序、无法追责断连期间的订单状态。

**方案**:

1. 新增专用审计 logger 命名空间 `akquant.audit.order`,独立于 `gateway.live`,可单独配置更低阈值与独立文件。
2. 在订单状态机的关键跃迁点埋 INFO 审计埋点,每条携带完整上下文(`build_log_extra`):
   - 提交:`order_submitter.py` submit 成功路径 → `event=order_submit`
   - 回报:`broker_event_bridge.py::_dispatch_strategy_event` 的 `order` 分支 → `event=order_update`(带 broker 状态)
   - 成交:`trade` 分支 → `event=order_fill`(带成交价/量/`trade_id`)
   - 撤单:cancel 路径 → `event=order_cancel`
   - 拒单:已有的 WARNING 升格为审计事件 `event=order_reject`
3. 上下文字段统一含:`strategy_id`、`slot`、`symbol`、`client_order_id`、`order_id`(承载 broker_order_id)、`event`、`side`、`price`、`quantity`、`order_status`、`order_type`、`trade_id`、`reason`(`trace_id` 贯穿留待 G5)。
4. `LogConfig` 增 `order_audit_file` 字段(独立 JSON 轮转文件)。

**涉及文件**:`gateway/broker_event_bridge.py`、`gateway/order_submitter.py`、`gateway/broker_execution.py`、`gateway/order_audit.py`(新增)、`log.py`(审计字段 + `build_order_audit_extra` + 审计文件 handler)。

**验收**:审计文件按序产出 submit/update/fill/cancel 事件,每条含 `client_order_id`/`order_id`,可仅凭该文件重建订单生命周期。见 `tests/test_order_audit.py`。

**已落地(2026-07-20)**:审计走 `akquant.audit.order` 命名空间;`gateway/order_audit.py` 提供 `record_submit`/`record_reject`/`record_cancel`/`record_broker_event`,全部防御式(审计异常绝不中断交易主流程);埋点接入提交(`order_submitter` place_order 成功后)、拒单(重复 client_order_id)、回报/成交(`broker_event_bridge.drain_events`)、撤单(`broker_execution.cancel_order`);审计记录默认随主日志落盘,`order_audit_file` 另存纯审计 JSON 流。`trace_id` 全链路贯穿仍属 G5。

### G2 — 敏感信息脱敏(🟠 P0)

**现状**:登录成功日志明文记录 `user_id`(账户号)与 `broker_id`(`ctp/native.py:422-428`、`:1007-1013`)。密码目前未进日志(`ctp/native.py:400` 附近仅打 `req_id`),但**无任何统一脱敏层**——任何人新增一行 debug 即可能泄漏。

**问题**:账户号明文落盘;缺乏兜底,安全性依赖每个调用点的自觉。

**方案**:

1. 在 `log.py` 增 `SensitiveFilter`(logging.Filter),对 record 的 message 与 `extra` 字段按键名与值模式做掩码,格式如 `user_id=****1234`(保留尾 4 位)。
2. 该 filter 默认挂载到所有 handler,`LogConfig` 增 `mask_sensitive: bool = True` 开关。
3. 敏感字段清单集中定义为常量,便于扩展。

**涉及文件**:`log.py`(新增 filter + 挂载)、`gateway/brokers/ctp/native.py`(账户号改走掩码字段)。

**验收**:构造含 `password`/`user_id` 的日志记录,输出经掩码;单测覆盖键名匹配与尾位保留。见 `tests/test_log_sensitive_filter.py`。

**已落地(2026-07-20)**:两档字段清单——`FULL_MASK_KEYS`(password/secret/token/api_key/app_key/app_secret/auth_code/private_key 等,全掩码 `****`)与 `TAIL_MASK_KEYS`(user_id/account/investor_id/broker_id 等,保留尾 4 位);`mask_sensitive_value`/`mask_sensitive_text` 两个纯函数 + `SensitiveFilter` 挂在 handler 层,`LogConfig.mask_sensitive` 默认 `True`。这是**兜底**——任何调用点忘记脱敏也不泄漏。`ctp/native.py` 源头字段改造(可选强化)未做:登录处的 `user_id` 现由 filter 兜底掩码。

### G3 — print→logging 收敛与入口统一(🟠 P1)

**现状**:
- ML 模块完全用 `print`:`ml/model.py:129,139,141,273`、`strategy_ml.py:134,259,273`(无级别、无法关闭、多进程串行错乱)。
- `checkpoint.py:111`、`strategy_scheduler.py:83,97`、`strategy_trading_api.py:1065` 有裸 `print`。
- `factor/engine.py:9`、`data.py:12`、`risk.py:10` 用标准库 `logging.getLogger` **绕过中央 `get_logger`**,享受不到结构化上下文与 profile。

**问题**:一致性破损,部分库内输出无法被 `LogConfig` 统一管控。

**方案**:

1. ML/scheduler/checkpoint 的诊断性 `print` → `get_logger("ml"|"scheduler"|...).info/debug`。
2. `factor`/`data`/`risk` 的 `logging.getLogger("akquant.xxx")` → `get_logger("xxx")`,归并到中央封装。
3. **保留**报表/绘图的用户可见输出(`backtest/result.py`、`plot/*`)——这类是面向终端用户的呈现,不是日志。

**涉及文件**:`ml/model.py`、`strategy_ml.py`、`checkpoint.py`、`strategy_scheduler.py`、`strategy_trading_api.py`、`factor/engine.py`、`data.py`、`risk.py`、`gateway/brokers/plugins.py`。

**验收**:`grep -rn 'print(' python/akquant` 仅剩报表/绘图白名单;`grep -rn 'logging.getLogger' python/akquant` 归零(改走 `get_logger`)。

**已落地(2026-07-20)**:诊断/异常类 `print` 全部转 `logger`——ML 训练进度(verbose 门控)转 `info`、异常转 `warning`;`checkpoint` 快照确认转 `info`;`scheduler` 时间解析失败转 `warning`;`strategy_trading_api` 无法定价转 `warning`。`factor`/`data`/`risk`/`plugins` 的 `logging.getLogger` 全部改走中央 `get_logger`(库内 `logging.getLogger` 现仅存于实现模块 `log.py`)。报表(`backtest/result.py`)与绘图(`plot/*`)的用户可见 `print` 按约定保留。

### G4 — 级别语义补齐(🟡 P1)

**现状**:全仓无 `logger.critical`;Rust `engine/`、`execution/` 热路径**只有 `log::warn!`**,无 info/error/debug/trace(`src/execution/common.rs`、`simulated.rs` 等)。

**方案**:

1. 定义 `CRITICAL` 使用规范并落到:实盘断连、风控熔断、对账不平、结算失败等系统级致命事件。
2. Rust 执行路径:拒单/撤单失败等错误从 `warn!` 升格 `error!`;撮合关键节点补 `info!`(经 profile 控制默认阈值,避免热路径刷屏)。

**涉及文件**:`gateway/*`(critical 埋点)、`src/execution/*`、`src/engine/*`。

**验收**:文档 `docs/zh/guide/strategy.md` 增「日志级别语义表」;Rust 侧 error/info 有对应用例。

**已落地(2026-07-20)**:
- **CRITICAL 语义**:约定为「系统级致命,需人工立即介入」。落到 `live.py`(实盘 runner 因未捕获异常整体停止,`exception`→`critical`)与 `ctp/native.py::OnFrontDisconnected`(交易前置断连=无法下单,`warning`→`critical`)。市场行情断连保持 `warning`(数据路径,非执行致命)。
- **Rust error 升格**:`python.rs`(自定义撮合回调异常/无效结果→订单未执行)与 `parquet_stream.rs`(读取/解析失败→数据流静默截断)由 `warn!` 升 `error!`。data 侧的 invalid-numeric 回退、clock fallback、跳过异常拆股、IOC/FOK 未成交撤单等**保持 `warn!`**(可恢复降级,warn 语义正确)。
- **级别语义表**:落 `docs/zh/guide/strategy.md`(含审计与脱敏说明,一并覆盖收尾清单的文档同步项)。
- **延后**:「撮合关键节点补 `info!`」本轮**不做**——`pyo3-log` 会把 Rust `info!` 全量转发,用户在实盘/研究开 INFO 时热路径会刷屏,价值低风险高;待 Rust 侧有独立级别节流机制后再议。

### G5 — 贯穿式 trace_id(🟡 P2)

**现状**:已有 `order_id`/`client_order_id` 上下文字段(`log.py:16-25`),但无一个贯穿「信号→下单→回报→成交→持仓」的全局关联 ID。

**方案**:

1. `CONTEXT_FIELDS` 增 `trace_id`;`build_log_extra` 支持透传。
2. 下单入口生成 `trace_id`(或复用 `client_order_id` 根),经事件 payload 透传,回报/成交继承同一 `trace_id`。
3. Rust `AkqLogContext`(`src/log_context.rs:8-26`)增 `trace_id` 字段,桥接回 Python 保持一致。

**涉及文件**:`log.py`、`gateway/order_submitter.py`、`gateway/broker_event_bridge.py`、`src/log_context.rs`。

**验收**:一笔订单的 submit/fill 审计日志共享同一 `trace_id`,可用它 grep 出完整链路。见 `tests/test_order_audit.py::test_trace_id_shared_across_submit_and_fill`。

**已落地(2026-07-20)**:**trace_id = group/root client_order_id**(逻辑订单根 id)。
- Python:`CONTEXT_FIELDS` 增 `trace_id`;`build_log_extra`/`build_order_audit_extra` 均支持透传;文本 formatter 上下文后缀也带上。
- 贯穿链路:submit 直接用 `request_client_order_id`(group 根)作 trace_id;fill/回报经 `resolve_trace_id` 回调(接 `live.py::_lookup_group_id`,复用既有 `client_order_id→group_id` 映射)继承同一 trace_id——`broker_event_bridge`→`broker_runtime`→`live` 镜像 `resolve_owner_strategy_id` 的传递链;拒单用 group 根;撤单暂留空(broker_order_id 已足够关联)。
- Rust:`AkqLogContext` 增 `trace_id` 字段 + builder,使 `[akq_ctx=...]` 载荷与 Python schema 一致。因引擎 `Order` 尚无 trace 字段,执行链路暂不填充(标 `#[allow(dead_code)]`),待 `Order.trace_id` 落地后点亮。

## 5. 实施分期

| 阶段 | 内容 | 依赖 |
|---|---|---|
| **P0-a** | G2 脱敏 filter(先立安全兜底) | 无 |
| **P0-b** | G1 订单审计埋点 + 审计文件 + 测试 | G2(审计日志复用脱敏) |
| **P1-a** | G3 print→logging 收敛 | 无 |
| **P1-b** | G4 级别语义 + Rust error/info | 无 |
| **P2** | G5 trace_id 贯穿 | G1(审计埋点作为落点) |

P0 两项建议合入同一 PR;G5 因涉及 Rust 边界改动,单独 PR 评审。

## 6. 风险与兼容

- **性能**:审计埋点在实盘事件路径(非回测热路径),频率受限于真实成交,开销可忽略;Rust `info!` 经 profile 阈值控制,回测默认不启用。
- **兼容**:全部经 `LogConfig` 开关、默认值保守;不改现有 API 签名与默认静默契约。
- **误脱敏**:`SensitiveFilter` 值模式需避免误伤正常业务字段(如 symbol),以键名匹配为主、值模式为辅,并留白名单。
- **审计文件体积**:订单审计独立轮转,`order_audit_file` 复用 `RotatingFileHandler`。

## 7. 验收清单(总)

- [x] G1:审计文件按序产出 submit/update/fill/cancel,可仅凭该文件重建订单生命周期(`tests/test_order_audit.py`)。
- [x] G2:敏感字段输出经掩码,单测覆盖(`tests/test_log_sensitive_filter.py`)。
- [x] G3:库内 `print` 仅剩报表/绘图白名单;`logging.getLogger` 归零(仅存 `log.py` 实现)。
- [x] G4:级别语义表落文档;CRITICAL 落地(live/断连);Rust error 升格(python/parquet_stream)。`info!` 延后(见 G4 说明)。
- [x] G5:一笔订单全链路共享 `trace_id`(submit/fill,`tests/test_order_audit.py`)。
- [x] 文档:`docs/zh/guide/strategy.md` 同步级别语义表 + 审计与脱敏说明。(`docs/zh/advanced/live_functional_quickstart.md` 待随 G5 一并补)
