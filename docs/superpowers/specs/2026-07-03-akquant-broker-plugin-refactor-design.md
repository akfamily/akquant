# akquant Broker 插件化核心重构设计（Phase 1.5）

日期：2026-07-03
状态：待评审
前置：QMF Phase 1（证券）已完成（见 `2026-07-03-qmf-broker-gateway-design.md`）。
目标读者：在此之后做 Phase 2（期权）的实现者。

## 1. 目标

让 broker 成为**真正的、可表达自身语义的插件**：新增/维护一个 broker 不必修改 akquant 核心
文件,且能表达自己品种/字段的语义（证券/期权/期货路由、期权开平/备兑/行权等）。

本重构是 QMF 与 Phase 2（期权）的**地基**,独立成一阶段交付,要求：
- **向后兼容**:现有全部 gateway 测试（当前 83 passed）保持通过,`create_gateway_bundle` 公共
  API 与行为不变,现有策略无需改动。
- 本阶段**不实现期权业务**,只铺好核心接缝;期权在 Phase 2 基于这些接缝落地。

## 2. 背景：现状的三处摩擦（实现 QMF 时撞到，已核实）

1. **无法承载 broker/品种专属字段**:`UnifiedOrderRequest`（`broker_models.py:87`）字段极简、
   无品种判别;`extra` 旁路在 `strategy_trading_api.py:765` 与 `order_submitter.py:214` 两处
   被硬 `raise`。`BrokerCapability.broker_extra_fields`（`broker_models.py:38`）已声明却
   **无任何消费方**——即设计预留了扩展位但从未接通。
2. **内置 broker 绕过 registry**:`factory.py` 先查 registry,查不到再走硬编码 if/elif
   （ctp/miniqmt/ptrade/qmf）。两套接入机制并存;新 broker 必须改核心 `factory.py`。
3. **`GatewayBundle` 强制 market_gateway**:纯交易 broker（QMF）被迫写空转 MarketGateway;
   `live.py:312` 无条件调用 `bundle.market_gateway.start`（全仓仅此一处 deref）。

## 3. 决策（已选）

本阶段把**「未来任意 broker 的扩展性」当作一等目标**,共六项:

- **(A3) 扩展机制 = 混合**:核心加**强类型** `asset_type` 判别字段 + 重启**受能力矩阵约束**的
  `extra` dict（broker 用 `broker_extra_fields` 声明自己的字段）。兼顾类型安全与自由度,复用
  既有 `broker_extra_fields` 预留位。
- **(B) 注册插件化**、**(C) 行情可选** 一并做。
- **(D) `TraderGatewayBase` 提为核心**:按「给所有未来 broker 复用」设计（回调注册、None 安全
  emit、exec 由订单派生、通用 id 反查表助手）,本阶段仅迁移 QMF,不动 CTP/stub。
- **(E) 能力开放位**:`BrokerCapability` 加 `features: frozenset[str]`,broker 声明**任意**能力
  标志不改核心 —— 把能力矩阵从闭集变开集（消灭与 factory 硬编码同类的病）。
- **(F) `asset_type` 归一**:`normalize_asset_type()` + 文档化规范集,broker 映射别名,保多
  broker 一致;对规范集外的新品种**放行不报错**(保持可扩展)。
- 另出 **(G) 「如何新增 broker」契约文档**,把插件契约写成一等公民。

## 4. 组件设计

### (A3) 订单字段扩展 —— 让 broker 表达自身语义

**`broker_models.py`**
- `UnifiedOrderRequest` 增两个带默认值的字段（位置兼容,现有构造不受影响）：
  - `asset_type: str = "stock"` —— 品种判别；构造/提交时经 `normalize_asset_type()` 归一。
  - `extra: dict[str, Any] = field(default_factory=dict)` —— broker 专属字段（如期权
    `entrust_oc` / `covered_flag`）。
- 新增校验函数 `validate_broker_extra(capability, extra) -> None`：`extra` 的每个 key 必须在
  `capability.broker_extra_fields` 内,否则 `raise RuntimeError`（清晰列出未声明的 key）。
  空 `extra` 恒通过。
- **(F) 新增 `normalize_asset_type(value) -> str`**：`strip().lower()`;对规范集
  `{"stock","option","future","fund","bond","index","fx","crypto"}` 内的别名做映射
  （如 `"opt"→"option"`、`"stk"→"stock"`）,**规范集外的值原样放行（不报错）**——未来 broker
  可引入新品种而不改核心。区别于 `normalize_position_effect`（后者对未知值报错）。
- **(E) `BrokerCapability` 增 `features: frozenset[str] = frozenset()`**：broker 声明任意能力
  标志（如 `"iceberg"`、`"oco"`、`"combo"`、`"margin_short"`）。`as_execution_capabilities()`
  带出 `"features": sorted(features)`;`from_value()` 从 dict 的 `features` 还原（容错
  list/tuple/set）。核心不预设这些标志的含义,由 adapter/上层解释。

**`order_submitter.py`（broker_live 路径）**
- `BrokerOrderSubmitter.submit_order` 签名增 `asset_type: str = "stock"`、`extra: dict | None = None`。
- 用 `validate_broker_extra(capability, extra or {})` 替换第 214 行的硬 `raise`；
  校验通过后把 `asset_type`、`extra` 写入构造出的 `UnifiedOrderRequest`。

**`strategy_trading_api.py`（回测/paper 的 Rust 模拟路径）**
- 统一 `submit_order`（:740）签名同样增 `asset_type`、`extra`（保持策略面 API 一致）。
- 模拟撮合无法兑现 broker 专属语义,故：`extra` 非空或 `asset_type != "stock"` 时 `raise`,
  但**改为清晰信息**（如“期权/broker 专属订单需在 broker_live 模式下使用”）——替换现第
  765 行含糊的报错。现有策略不传这两个参数,行为完全不变。

> 说明：`submit_order` 两处实现（sim / broker_live）仍分处两文件,是既有 monkey-patch 架构的
> 产物;本阶段仅让二者签名一致、语义清晰,**不重构** monkey-patch 机制（风险大,冻结）。

### (B) 内置 broker 注册化 —— 真插件

- 新增 `python/akquant/gateway/brokers/builtins.py`：为 ctp/miniqmt/ptrade/qmf 各写一个 builder
  函数（签名 `(feed, symbols, use_aggregator, **kwargs) -> GatewayBundle`,函数体即把
  `factory.py` 对应 if 分支的逻辑**原样搬入**），并提供 `register_builtin_brokers()` 逐个
  `register_broker(...)`。
  - ctp/miniqmt/ptrade builder 可在模块顶部 import 各自 adapter（与现状一致,CTP 缺 SDK 时已
    优雅降级）。
  - qmf builder 在**函数体内**局部 import qmf 模块（保持 httpx 等为可选依赖）。
- `python/akquant/gateway/__init__.py`：在导入 adapters 之后调用 `register_builtin_brokers()`,
  确保导入 `akquant.gateway` 即完成注册。注册的是 builder **可调用对象**（不触发重依赖导入）。
- `factory.create_gateway_bundle`：删除 if/elif 链。流程 = 确保内置已注册 → 查 registry →
  命中则 build,否则 `raise ValueError(f"broker must be one of: {list_registered_brokers()}")`。
  **公共签名、行为、错误信息形态保持不变**;registry 覆盖内置的能力保留。
- 幂等：`register_builtin_brokers()` 可重复调用不报错（覆盖写入同名 builder）。

### (C) 行情可选

- `protocols.py`：`GatewayBundle.market_gateway: MarketGateway | None`（改为可选）。
- `live.py:312`：包一层 `if bundle.market_gateway is not None:` 再 `start`。
- qmf builder 返回 `market_gateway=None`；**删除** `QMFMarketGateway` 空类及其引用。

### (D) 核心基类 `TraderGatewayBase`（面向所有未来 broker）

按「每个未来 broker 都继承它、复用共享管件」来设计,而非仅 QMF 清理。
- 新增 `python/akquant/gateway/trader_base.py`：`TraderGatewayBase`
  - `__init__` 初始化回调槽与 **id 反查表** `_client_id_by_broker: dict[str,str]`。
  - 回调：`on_order/on_trade/on_execution_report` 存回调 + None 安全 `_emit_order/_emit_trade/
    _emit_exec`；`_emit_exec_from_order(snapshot)` 由订单快照派生执行回报（省去每个 broker 重写）。
  - **通用 id 反查表助手**：`record_broker_order(broker_order_id, client_order_id)`、
    `client_order_id_for(broker_order_id) -> str`（未命中回空串）—— 把 QMF 里手写的
    `entrust_no↔client_order_id` 逻辑升为所有 broker 可复用。
  - 默认实现 `heartbeat()->True`、`sync_open_orders()->[]`、`sync_today_trades()->[]`；
    `connect/disconnect/start/place_order/cancel_order/query_*/get_capabilities` 保持抽象
    （子类必实现）。
- `QMFTraderGateway` 继承它:删回调注册与自维护的 `_client_id_by_broker`,改用基类助手;
  `_dispatch_push` 用 `_emit_*` + `client_order_id_for`。
- CTP、miniqmt/ptrade **本阶段不迁**（降低风险；它们仍以 Protocol 直接实现,基类不强制）。

### (G) 「如何新增一个 broker」契约文档

- 新增 `docs/zh/advanced/adding_a_broker.md`：把插件契约写成一等公民 —— 
  ① 写 builder 并 `register_broker(name, builder)`（或放进 `brokers/builtins.py`）；
  ② 声明 `BrokerCapability`（`broker_extra_fields` 声明订单专属字段、`features` 声明能力标志）；
  ③ 继承 `TraderGatewayBase` 实现必需方法；④ 按 `asset_type` 路由、从 `req.extra` 取专属字段；
  ⑤ 行情可选（`market_gateway` 可为 `None`）。附最小可跑骨架片段,链接 QMF 作范例。

## 5. 数据流（broker_live 下单，重构后）

策略 `submit_order(symbol, side, ..., asset_type="option", extra={"entrust_oc":"O","covered_flag":"0"})`
→（broker_live 已 monkey-patch）`BrokerOrderSubmitter.submit_order` → `normalize_asset_type` +
`validate_broker_extra`（对照 `capability.broker_extra_fields`）→ 构造带归一 `asset_type`/`extra` 的
`UnifiedOrderRequest` → `trader_gateway.place_order(req)` → broker adapter 读 `req.asset_type` 选
endpoint、读 `req.extra` 取专属字段、`record_broker_order(...)` 记 id 反查 → 落到具体柜台请求。
回报经 WS/查询 → `client_order_id_for(broker_order_id)` 还原 → 基类 `_emit_*` 分发回策略。

## 6. 错误处理

- `extra` 含未声明 key → `validate_broker_extra` 抛清晰错误(列出违规 key 与已声明集合)。
- sim/backtest 模式收到 `extra`/非 stock `asset_type` → 抛“需 broker_live”清晰错误。
- 未知 broker → `ValueError` 列出 `list_registered_brokers()`（含全部内置 + 已注册）。
- 其余保持现状。

## 7. 向后兼容与测试

**兼容硬约束**：现有 gateway 测试全绿（当前 83 passed），`create_gateway_bundle(broker="ctp"/
"miniqmt"/"ptrade"/"qmf", ...)` 行为逐一不变；现有策略（不传 asset_type/extra）零变化。

**新增测试**：
- `broker_models`：`UnifiedOrderRequest` 默认 `asset_type=="stock"`、`extra=={}`；
  `validate_broker_extra` 合法通过 / 未声明 key 抛错 / 空 extra 通过。
- **(F) `normalize_asset_type`**：别名映射（`"opt"→"option"`）、大小写/空白归一、规范集外值原样
  放行不报错。
- **(E) `BrokerCapability.features`**：默认空 frozenset；`as_execution_capabilities()` 带出
  `features`；`from_value()` 从 list/tuple/set 还原。
- **(D) `TraderGatewayBase`**：回调注册 + `_emit_*` None 安全；`record_broker_order` /
  `client_order_id_for` 反查（命中/未命中）；`_emit_exec_from_order` 派生执行回报；
  默认 `heartbeat/sync_*`。
- `order_submitter`：broker_live 提交带合法 `extra` 时 `place_order` 收到含归一 `asset_type`/
  `extra` 的请求；未声明 key 抛错。
- `strategy_trading_api`：sim 模式传 `extra` 抛清晰错误；不传时行为不变。
- 注册化：导入 `akquant.gateway` 后 `list_registered_brokers()` 含 ctp/miniqmt/ptrade/qmf；
  `create_gateway_bundle` 四个内置各自可建;未知 broker 错误列出它们;registry 覆盖内置仍生效。
- 行情可选：`market_gateway=None` 的 bundle 不导致 LiveRunner 启动路径崩溃（对该守卫单测）。
- QMF：迁移到基类后 qmf 套件仍全绿;端到端对 `chibi_quant --mock` 仍 exit 0。

遵循 CLAUDE.md：`uvx ruff check` + `uvx ruff format --check` 通过（ruff `E,F,I,D`，公共函数含测试
需 imperative docstring）；示例实跑 exit 0；提交用 `--no-verify`（重型 pre-commit 会重编 Rust），
最终跑一次完整 pre-commit 门禁。

## 8. 不做（YAGNI / 冻结）

- 不实现期权业务（Phase 2）。
- 不重构 monkey-patch 下单机制与 `realtime.rs`（风险大,冻结）。
- 不迁移 CTP / miniqmt / ptrade 到 `TraderGatewayBase`（仅 QMF；基类不强制,Protocol 仍可直接实现）。
- `features` 只提供开放声明位,核心不预设/不校验各标志含义（由 adapter/上层解释）。
- `on_order` 仍单订阅（不引入多订阅观察者）。
- `asset_type` 不做闭集强校验（规范集外放行）——保持对新品种的可扩展。

## 9. 影响面清单（要改的核心文件）

- `python/akquant/gateway/broker_models.py`（加 `asset_type`/`extra` 字段、`validate_broker_extra`、
  `normalize_asset_type`(F)、`BrokerCapability.features`(E)）
- `python/akquant/gateway/order_submitter.py`（放开 extra + asset_type，归一）
- `python/akquant/strategy_trading_api.py`（签名一致 + 清晰报错）
- `python/akquant/gateway/protocols.py`（market_gateway 可选）
- `python/akquant/live.py`（判空守卫）
- `python/akquant/gateway/factory.py`（删 if/elif）
- `python/akquant/gateway/brokers/builtins.py`（新增，注册内置）
- `python/akquant/gateway/__init__.py`（调用注册）
- `python/akquant/gateway/trader_base.py`（新增，核心基类 D）+ `brokers/qmf/adapter.py`（迁移基类、
  删空转行情）
- `docs/zh/advanced/adding_a_broker.md`（新增，插件契约文档 G）

## 10. 待确认（本轮已随扩展性升级敲定，列出以便复核）

1. 扩展机制 = A3（强类型 `asset_type` + 能力矩阵约束的 `extra`）。
2. 六项全收：A3 + B 注册化 + C 行情可选 + D 核心基类 + E 能力开放位 `features` + F `asset_type`
   归一 + G 契约文档。
3. `asset_type` 采规范集 + 别名归一,**规范集外放行**（不做闭集强校验）。
