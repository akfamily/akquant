# 第 7 章：期货市场与衍生品策略

> ⏱️ 预计阅读 ~30 分钟 ｜ 🎯 难度 ★★★★☆（进阶）

期货 (Futures) 市场是现代金融体系中风险管理的核心场所。与股票市场不同，期货交易本质上是一种**零和博弈 (Zero-Sum Game)**，其核心机制——**保证金制度**、**双向交易**和**每日无负债结算**——为量化策略提供了丰富的操作空间，但也带来了更高的风险管理要求。本章将深入探讨期货市场的定价理论、期限结构、对冲策略以及在 `AKQuant` 引擎中的实现细节。

## 学习目标

- 理解保证金、杠杆、双向交易与每日结算在期货回测中的含义。
- 掌握中国期货配置、合约属性与风控参数的关键入口。
- 能够分析期货策略波动放大的来源。

## 前置知识

- 已掌握第 4 到 6 章的策略与市场制度基础。
- 了解期货合约、乘数与保证金的基本概念。

## 本章实践入口

- 主示例：[examples/textbook/ch07_futures.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch07_futures.py)
- 进阶示例：[examples/04_mixed_assets.py](https://github.com/akfamily/akquant/blob/main/examples/04_mixed_assets.py)
- 对应指南：[策略指南](../guide/strategy.md)

## 快速运行与验收

```bash
python examples/textbook/ch07_futures.py
```

验收要点：

1. 脚本能够完成期货回测并输出收益与风险指标。
2. 结果中可体现保证金与杠杆对权益波动的放大效应。
3. 调整合约参数或杠杆后，策略波动率变化符合预期。

## 7.0 AKQuant 中国期货配置速览

`AKQuant` 在 `BacktestConfig` 下提供了 `china_futures` 配置入口，可同时管理：

- 品种前缀模板（乘数、保证金、tick、手数）
- 品种前缀费率覆盖
- 品种前缀校验开关（tick 对齐、手数整数倍）
- 会话行为（是否强制交易时段）

示例：

```python
from akquant import (
    BacktestConfig,
    ChinaFuturesConfig,
    ChinaFuturesInstrumentTemplateConfig,
    ChinaFuturesValidationConfig,
    InstrumentConfig,
    StrategyConfig,
)

config = BacktestConfig(
    strategy_config=StrategyConfig(initial_cash=500_000),
    instruments_config=[
        InstrumentConfig(symbol="RB2310", asset_type="FUTURES")
    ],
    china_futures=ChinaFuturesConfig(
        enforce_sessions=False,
        instrument_templates_by_symbol_prefix=[
            ChinaFuturesInstrumentTemplateConfig(
                symbol_prefix="RB",
                multiplier=10.0,
                margin_ratio=0.1,
                tick_size=1.0,
                lot_size=1.0,
                commission_rate=0.0001,
            )
        ],
        validation_by_symbol_prefix=[
            ChinaFuturesValidationConfig(
                symbol_prefix="RB",
                enforce_tick_size=False,
                enforce_lot_size=True,
            )
        ],
    ),
)
```

构造校验（Fail Fast）：

- `symbol_prefix` 会自动去空格并大写；空值会直接报错
- 模板里 `multiplier / margin_ratio / tick_size / lot_size` 必须大于 0
- 模板和前缀规则里的 `commission_rate` 必须大于等于 0
- `validation_by_symbol_prefix` 每条规则至少设置一个开关
- 同一列表内前缀重复会报错，并定位到冲突项索引
- `session_profile` 支持 `CN_FUTURES_DAY`(商品日盘) / `CN_FUTURES_CFFEX_STOCK_INDEX_DAY` / `CN_FUTURES_CFFEX_BOND_DAY` / `CN_FUTURES_NIGHT_23 / CN_FUTURES_NIGHT_01 / CN_FUTURES_NIGHT_0230`

配置优先级矩阵（从高到低）：

| 场景 | 最高优先级 | 次级优先级 | 兜底 |
|---|---|---|---|
| 合约参数（乘数/保证金/tick/手数） | `InstrumentConfig` 显式字段 | `instrument_templates_by_symbol_prefix` | `run_backtest` 默认参数 |
| 前缀费率 | `fee_by_symbol_prefix` | 模板里的 `commission_rate` | `StrategyConfig.commission_rate` |
| 前缀校验开关 | `validation_by_symbol_prefix` | 模板里的 `enforce_tick_size / enforce_lot_size` | `ChinaFuturesConfig` 全局 `enforce_*` |
| 市场选择 | `use_china_futures_market=False` 或混合资产触发中国市场 | `use_china_futures_market=True` 且纯期货资产 | `use_simple_market` |

说明：

- 同一优先级层内按“同名单项覆盖”处理，后定义的显式规则覆盖模板默认。
- 在撮合校验路径中，前缀匹配采用更具体前缀优先（例如 `RB` 与 `RB2` 同时命中时优先 `RB2`）。

低层引擎 API 命名说明：

- 期货费率相关接口统一为 `set_futures_fee_rules` 与 `set_futures_fee_rules_by_prefix`。
- 旧命名单数形式 `set_future_fee_rules*` 已移除，迁移时请直接替换为复数接口。

## 7.1 期货市场机制 (Market Mechanisms)

### 7.1.1 标准化合约

期货合约是由交易所制定的、规定在将来某一特定时间和地点交割一定数量标的物的标准化合约。它的标准化主要体现在两个维度上。其一是**标的物 (Underlying)**，它既可以是实物商品（如螺纹钢、大豆），也可以是金融资产（如沪深300指数、国债）。其二是**合约乘数 (Multiplier)**，即一张合约代表的标的价值；以 IF（沪深300股指期货）为例，其乘数为 300元/点，若指数为 4000 点，则一张合约价值 $4000 \times 300 = 1,200,000$ 元。

### 7.1.2 保证金制度 (Margin System)

期货交易实行保证金制度，投资者只需缴纳合约价值一定比例的资金作为履约担保，即可进行交易。这引入了**杠杆 (Leverage)**。

$$ \text{Leverage} = \frac{1}{\text{Margin Ratio}} $$

若保证金率为 10%，则杠杆为 10 倍。这意味着标的资产价格波动 1%，账户权益波动 10%。

### 7.1.3 每日无负债结算 (Mark-to-Market)

A 股股票是 T+1 交收，而期货实行**T+0 交易**和**每日无负债结算 (Daily Mark-to-Market)**。交易所每日根据**结算价 (Settlement Price)**（通常为当日成交量加权平均价）计算盈亏，并划转资金。

在 `AKQuant` 回测引擎中，为了简化模型，通常采用**收盘价**近似结算价进行每日权益计算，但在实盘模块中必须严格对接交易所结算单。

## 7.2 期限结构与主力合约 (Term Structure)

### 7.2.1 期限结构理论

同一品种、不同到期月份的期货合约，其价格通常不同，形成**期限结构 (Term Structure)**。

根据持有成本理论 (Cost of Carry Model)：

$$ F_t = S_t e^{(r+u-y)(T-t)} $$

其中 $F_t$ 为期货价格，$S_t$ 为现货价格，$r$ 为无风险利率，$u$ 为仓储成本，$y$ 为便利收益 (Convenience Yield)。

期限结构通常呈现两种相反的形态。一种是**升水结构 (Contango / Normal Market)**，即 $F_{far} > F_{near}$，它通常发生在库存充足、仓储成本高或便利收益低的商品中（如黄金、铝）；在这种结构下，多头展期面临**负基差**（高买低卖），从而产生展期亏损 (Roll Yield Loss)。另一种是**贴水结构 (Backwardation / Inverted Market)**，即 $F_{far} < F_{near}$，它通常发生在现货短缺、便利收益极高的商品中（如现货紧张时的铜、大豆）；此时多头展期享受**正基差**（低买高卖），获得展期收益。

### 7.2.2 主力合约与移仓换月 (Rolling)

期货合约有生命周期，成交量和持仓量通常集中在某一个特定月份的合约上，称为**主力合约 (Dominant Contract)**。随着到期日临近，资金会迁移到下一个月份，形成**主力切换**。

为了进行长期回测，我们需要构建**连续合约 (Continuous Contract)**。`AKQuant` 建议采用以下逻辑进行移仓 (Rolling)：

1.  **信号触发**：当次主力合约的持仓量 (Open Interest) 超过当前主力合约时。
2.  **执行动作**：平掉旧合约，开仓新合约。
3.  **价格调整**：
    *   **价差调整 (Difference Adjustment)**：$P_{adj} = P_{raw} + (P_{new} - P_{old})$。保持绝对价差，适用于价差套利策略。
    *   **比例调整 (Ratio Adjustment)**：$P_{adj} = P_{raw} \times \frac{P_{new}}{P_{old}}$。保持百分比收益，适用于趋势策略。

## 7.3 对冲与套利策略 (Hedging & Arbitrage)

### 7.3.1 Beta 对冲 (Beta Hedging)

对于股票多头组合，可以通过做空股指期货来对冲系统性风险 (Beta Risk)，获取纯粹的 Alpha 收益。

所需期货合约数量 $N$ 计算公式：

$$ N = \beta_P \times \frac{V_P}{V_F} $$

其中：

*   $\beta_P$：投资组合相对于指数的 Beta 系数。
*   $V_P$：投资组合市值。
*   $V_F$：单张期货合约价值 ($Price \times Multiplier$)。

### 7.3.2 跨期套利 (Calendar Spread)

利用期限结构的异常变动获利。

*   **牛市套利 (Bull Spread)**：买入近月，卖出远月。适用于预期供应紧张（Backwardation 加深）的市场。
*   **熊市套利 (Bear Spread)**：卖出近月，买入远月。适用于预期供应过剩（Contango 加深）的市场。

## 7.4 引擎实现与配置

### 7.4.1 合约配置 (InstrumentConfig)

在 `AKQuant` 中，必须正确配置期货合约的乘数和保证金率，才能正确计算盈亏和风险。

```python
from akquant import InstrumentConfig

# 配置螺纹钢 (RB) 主力合约
rb_config = InstrumentConfig(
    symbol="RB.SHF",       # 交易所代码需规范
    asset_type="FUTURES",  # 资产类型
    multiplier=10.0,       # 1手 = 10吨
    margin_ratio=0.13,     # 保证金率 13%
    tick_size=1.0          # 最小变动价位
)
```

### 7.4.2 策略示例：双均线趋势策略 (Dual MA Trend)

下面的代码展示了一个经典的期货趋势跟踪策略。注意期货特有的**做空 (Short Selling)** 操作。

防踩坑提醒：

*   期货策略请显式配置 `InstrumentConfig(asset_type="FUTURES", multiplier=..., margin_ratio=...)`，不要依赖默认资产类型。
*   如果你的人工记录按“当根收盘信号、当根收盘附近成交”统计，请显式设置 `fill_policy={"price_basis":"close","bar_offset":0,"temporal":"same_cycle"}`。默认语义通常是“当根算信号、下一根开盘成交”。
*   `run_backtest(..., slippage=...)` 建议显式写成 policy，例如 `{"type":"percent","value":0.0002}`、`{"type":"fixed","value":0.2}` 或 `{"type":"ticks","value":1}`。其中百分比语义下，`0.0002 = 2 bps`，如果写成 `0.2`，表示 `20%` 滑点，而不是 `0.2` 个点。
*   需要表达固定点差时，优先使用订单级滑点配置，例如 `slippage={"type":"fixed","value":0.2}`。

```python
--8<-- "examples/textbook/ch07_futures.py"
```

**关键逻辑解析**：

这段策略的核心在于三点。在**做空开仓**上，优先使用 `self.short(symbol, quantity)`，因为它语义更清晰，也更适合向用户解释成交明细；与之对应，**平空仓**则优先使用 `self.cover(symbol, quantity)`，若当前持仓为 -1，回补 1 手后持仓归零。而在**杠杆管理**上，由于期货自带杠杆，策略在分配资金时需谨慎，因此建议按**名义本金 (Notional Value)** 而非保证金占用进行风控。

### 7.4.3 账户快照口径（期货保证金账户）

在 AKQuant 的期货保证金账户语义下，`get_account()` 与 `get_portfolio_value()` 建议按下面方式理解：

- `cash`: 账户现金余额。开仓期货时不会像股票买入那样扣减全额名义本金，只会反映手续费、已实现盈亏等现金流。
- `equity`: 账户权益，是期货保证金账户最重要的总资产口径；`get_portfolio_value()` 与它保持一致。
- `used_margin` / `margin`: 当前已占用保证金。
- `free_margin`: 可用保证金（`equity - used_margin`），即真正可用于新开仓的资金。下单因保证金不足被拒时，日志里的 `Available` 就是这个口径——不要拿 `cash` 去和它比较。
- `notional_value`: 当前期货名义敞口，用于观察杠杆暴露。
- `unrealized_pnl`: 按最新价格计算的浮动盈亏。
- `market_value`: 主要用于现货/持仓市值语义；对期货保证金账户，不应把它理解为“名义本金”。

一个常见误区是把期货账户按股票现货账户去理解，认为“开 1 手后现金应该减少整笔合约价值”。这并不符合保证金账户语义。更合理的观察方式是：

1. 看 `equity` 是否反映了浮动盈亏。
2. 看 `used_margin` 是否反映了保证金占用。
3. 看 `notional_value` 是否反映了期货敞口规模。

## 7.5 商品期限结构深入 (Deep Dive into Term Structure)

商品期货的定价核心在于**持有成本 (Cost of Carry)** 和 **便利收益 (Convenience Yield)**。

$$ F_t = S_t e^{(r + u - y)(T-t)} $$

公式中的两项力量方向相反。**仓储成本 ($u$)** 反映了持有现货需要仓库、保险，这会推高远期价格（Contango）；而**便利收益 ($y$)** 则源于持有现货能应对突发的需求冲击（如工厂停工），这种“期权价值”会压低远期价格（Backwardation）。

正因如此，**库存与价格的关系**也随之分化：当处于**高库存**时，便利收益低，仓储成本主导，价格趋于 Contango（远月升水）；当处于**低库存**时，便利收益高，现货溢价，价格趋于 Backwardation（近月升水）。

## 7.6 基差交易 (Basis Trading)

基差 ($Basis = Spot - Futures$) 最终会随着交割日的临近而收敛到 0（或接近交割成本）。

**期现套利 (Cash-and-Carry Arbitrage)**：

1.  **正向套利**：当期货价格大幅高于现货（$F >> S$），且基差能够覆盖资金成本和仓储成本时。
    *   买入现货，卖出期货。
    *   持有到期交割，锁定无风险利润。
2.  **反向套利**：当期货价格大幅低于现货（$F << S$）。
    *   卖出现货（如果手中有货），买入期货。
    *   到期接货，补回库存。

## 7.7 跨品种套利 (Cross-Variety Arbitrage)

利用相关品种之间的价差回归进行交易。

### 7.7.1 产业链套利 (Processing Spread)

最典型的是**大豆压榨套利 (Crush Spread)**：

$$ 1 \text{大豆} \approx 0.18 \text{豆油} + 0.8 \text{豆粕} $$

围绕这一压榨关系，可以构造方向相反的两类操作。**提油套利**是买入大豆、卖出豆油和豆粕，其逻辑在于：当压榨利润过低（甚至亏损）时，油厂会停工，减少豆油豆粕供给，从而推高产品价格。**反向提油套利**则是卖出大豆、买入豆油和豆粕，其逻辑相反：当压榨利润过高时，油厂开足马力，增加供给，进而压低产品价格。除大豆产业链外，黑色系（螺纹钢-铁矿石-焦炭）也是常见的产业链套利标的。

### 7.7.2 替代品套利 (Substitution Spread)

替代品套利利用的是品种之间的替代关系。例如**豆油 vs 棕榈油**，两者在食用油领域有替代关系，当价差过大时，可买入低价品种、卖出高价品种；**玉米 vs 小麦**则在饲料领域有替代关系，可循同样的思路构造价差。

## 7.8 交易所规则与实盘细节

在实盘交易中，必须严格遵守交易所规则，否则会导致废单或处罚。

1.  **夜盘交易 (Night Trading)**：
    *   大多数活跃品种（黄金、原油、螺纹钢）都有夜盘（21:00 - 23:00 或 02:30）。
    *   **策略启示**：外盘（如 COMEX 黄金、LME 铜）的波动主要发生在夜间，必须在夜盘时段保持策略在线。

2.  **涨跌停板与扩板**：
    *   期货的涨跌停板通常比股票窄（如 4%-8%）。
    *   **扩板规则**：当出现单边市（连续涨跌停）时，交易所会提高保证金率并扩大涨跌停幅度（D1: 4% $\rightarrow$ D2: 7% $\rightarrow$ D3: 9%）。

3.  **交割月限制**：
    *   **个人户**：通常不能进入交割月（如 2305 合约，个人户必须在 4 月底前平仓）。
    *   **持仓限额**：交易所对单品种持仓有严格限制，超过限制会被强平。

在 AKQuant 中，如果你为期货合约配置了 `expiry_date`，并且引擎在日终实际执行了到期结算或到期移除，策略可通过 `on_expiry(event)` 接收通知。这个回调非常适合实现换月记录、到期后自动选取下月主力合约、或把到期结算事件推送到监控系统。最小可运行示例见：`examples/49_on_expiry_demo.py`。

## 7.9 经典 CTA 策略解析

### 7.9.1 R-Breaker

这是一个经典的日内反转+趋势策略，曾连续多年被评为全美前十最赚钱的策略。

*   **逻辑**：根据昨日的 $(H, L, C)$ 计算出 6 个枢轴点 (Pivot Points)。
    *   **观察区**：$S_{setup} < P < B_{setup}$。
    *   **反转区**：突破 $S_{enter}$ 或 $B_{enter}$ 后回调。
    *   **趋势区**：突破 $S_{break}$ 或 $B_{break}$。

### 7.9.2 Aberration

一个基于布林带的长线趋势策略。

*   **做多**：价格突破布林带上轨。
*   **做空**：价格跌破布林带下轨。
*   **平仓**：价格回归中轨。
*   **组合**：通常同时交易 8 个以上的低相关品种（如铜、棉花、大豆、股指）来分散风险。

## 7.10 期货高频交易 (HFT)

期货市场是 HFT 的主战场。

### 7.10.1 做市商 (Market Making)

做市商的**业务**是同时挂出买单 (Bid) 和卖单 (Ask)，赚取价差 (Spread)。但这门生意的核心**风险**是**存货风险 (Inventory Risk)**：如果单边行情剧烈，手中的存货会产生巨额亏损。为应对这一风险，做市策略的**核心模型**是 Avellaneda-Stoikov 模型，它会根据当前存货调整报价位置。

### 7.10.2 订单流策略 (Order Flow)

通过分析 L2 逐笔委托数据，识别大户的拆单行为 (Iceberg Orders)。

## 本章小结

### 必须掌握

- 期货回测中最重要的变量不是方向判断，而是杠杆、保证金与风险暴露。
- 合约参数、交易时段与校验规则决定了回测结果是否可信。

### 理解即可

- 期限结构、套保与跨品种扩展是期货策略进一步演进的研究方向。

### 实践提醒

- 先把合约属性和风险约束设清楚，再讨论收益率高低。

## 主线推进

贯穿全书的那条最小双均线 / 趋势策略，在前几章里一直运行在 A 股股票这类单向、全额资金占用的市场上。本章把它迁移到了期货这一全新的市场场景：同样的双均线趋势逻辑被重写为期货版本（见 7.4.2），但执行语义发生了关键变化——策略第一次显式面对保证金、杠杆、双向交易与每日无负债结算，下单接口也从只做多的 `buy` / `sell` 扩展到 `short` / `cover`，并通过 `InstrumentConfig` 与 `china_futures` 配置补齐了乘数、保证金率、tick 与手数等合约属性。至此，主线策略不再假设“一手等于一笔全额现金”，而是学会用 `equity`、`used_margin`、`notional_value` 去观察一个保证金账户的真实风险暴露，为后续把它推广到期权、跨品种与实盘场景做好了市场制度上的准备。

## 延伸阅读

**经典著作**

- Hull, J. C. *Options, Futures, and Other Derivatives*（《期权、期货及其他衍生产品》），Pearson —— 衍生品定价与对冲的权威教材，系统讲解持有成本模型、期限结构与 Beta 对冲，可与本章 7.1（市场机制）、7.2（期限结构）、7.3（对冲与套利）、7.5（持有成本与便利收益）对照精读。
- Kaufman, P. J. *Trading Systems and Methods*（第 6 版），John Wiley & Sons, 2020 —— 系统化交易与 CTA 策略设计的综合性参考，涵盖趋势跟随、突破系统与风险建模，对应本章 7.9（经典 CTA 策略解析）。
- Avellaneda, M., & Stoikov, S. *High-frequency Trading in a Limit Order Book*（Quantitative Finance, 2008）—— 做市报价与存货风险管理的奠基性论文，直接对应本章 7.10.1（做市商）所述的 Avellaneda-Stoikov 模型。

**官方文档与工具**

- [AKQuant 策略指南](../guide/strategy.md) —— `on_expiry` 等到期/换月回调的触发条件与用法说明，对应本章 7.8（交易所规则与实盘细节）。
- [AKShare 官方文档](https://akshare.akfamily.xyz/) —— 期货行情、主力合约与持仓量数据的获取来源，支撑本章 7.2.2（主力合约与移仓换月）的连续合约构建。

**本书相关**

- [第 1 章：量化投资概述与环境搭建](01_foundations.md) —— 本章期货回测沿用第 1 章建立的回测全流程与时间心智模型。
- [第 5 章：策略开发实战](05_strategy.md) —— 本章 7.4.2 的 `short` / `cover` 下单与开平语义承接第 5.3.5 节的 `position_effect` 说明，策略层风控承接第 5.6 节。

## 课后练习

### 基础题

1. 调整保证金率或乘数，观察权益波动如何变化。

### 应用题

1. 增加一个简单的移仓换月规则并记录策略表现差异。

### 综合题

1. 对同一策略分别在低杠杆与高杠杆配置下做风险收益对比。

??? note "参考答案要点（先独立思考再展开）"

    **基础题**：保证金率下降或乘数上升都会抬高杠杆，使权益波动放大——收益与回撤同比放大。

    **应用题**：按持仓量信号在主力切换时平旧开新，用价差调整或比例调整拼接连续合约；记录换月前后的价格跳变与盈亏影响。

    **综合题**：高杠杆放大夏普分子的同时也放大回撤，常导致卡玛比率下降甚至爆仓风险上升，说明杠杆并非越高越好。

## 常见错误与排查

1. 盈亏异常放大：检查合约乘数和保证金参数是否配置正确。
2. 换月跳变明显：确认主力合约切换与连续合约处理逻辑。
3. 风控频繁触发：核对仓位规模与杠杆设置是否过高。
