# 第 8 章：期权定价与波动率策略

> ⏱️ 预计阅读 ~30 分钟 ｜ 🎯 难度 ★★★★☆（偏难）

期权 (Options) 是金融工程皇冠上的明珠。它不仅是一种非线性 (Non-linear) 的衍生品，更是交易“波动率 (Volatility)”和“时间 (Time)”的工具。本章将从经典的 Black-Scholes-Merton (BSM) 定价模型出发，深入剖析希腊字母 (Greeks) 的数学含义与风控应用，并展示如何在 `AKQuant` 中构建专业的期权策略。

## 学习目标

- 理解期权定价、时间价值、隐含波动率与 Greeks 的核心作用。
- 掌握期权策略回测中的主要配置入口与风险暴露来源。
- 能够解释执行价、到期日与波动率变化如何影响策略表现。

## 前置知识

- 建议先掌握第 7 章中的衍生品与风险控制基础。
- 了解看涨、看跌、到期日和执行价等基本术语。

## 本章实践入口

- 主示例：[examples/textbook/ch08_options.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch08_options.py)
- 进阶示例：[examples/07_option_test.py](https://github.com/akfamily/akquant/blob/main/examples/07_option_test.py)
- 函数式孪生示例：[examples/textbook/ch08_options_functional.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch08_options_functional.py)
- 对应指南：[量化基础](../guide/quant_basics.md)

## 快速运行与验收

```bash
python examples/textbook/ch08_options.py
python examples/textbook/ch08_options_functional.py
```

验收要点：

1. 脚本可完成期权策略回测并输出核心统计指标。
2. 输出中可观察到 Greeks 或波动率变化对策略表现的影响。
3. 修改执行价或到期参数后，结果变化符合期权定价直觉。

## 8.0 AKQuant 中国期权配置速览

`AKQuant` 提供 `BacktestConfig.china_options` 用于中国期权费率配置：

- `fee_per_contract`: 全局每张合约手续费
- `fee_by_symbol_prefix`: 按品种前缀覆盖手续费
- `use_china_market`: 中国市场路由开关

与中国期货配置能力的详细对照可参考 API 文档中的“期货 vs 期权配置能力对照”。

示例：

```python
from akquant import (
    BacktestConfig,
    ChinaOptionsConfig,
    ChinaOptionsFeeConfig,
    InstrumentConfig,
    StrategyConfig,
)

config = BacktestConfig(
    strategy_config=StrategyConfig(initial_cash=500_000),
    instruments_config=[
        InstrumentConfig(
            symbol="RB2310-C-3800",
            asset_type="OPTION",
            option_type="CALL",
            strike_price=3800.0,
            underlying_symbol="RB2310",
        )
    ],
    china_options=ChinaOptionsConfig(
        fee_per_contract=5.0,
        fee_by_symbol_prefix=[
            ChinaOptionsFeeConfig(
                symbol_prefix="RB",
                commission_per_contract=8.0,
            )
        ],
    ),
)
```

## 8.1 期权基础与定价理论 (Pricing Theory)

### 8.1.1 核心要素

期权赋予买方在未来特定时间 ($T$) 以特定价格 ($K$) 买入或卖出标的资产 ($S$) 的**权利**。

*   **Call (看涨)**：$Payoff = \max(S_T - K, 0)$
*   **Put (看跌)**：$Payoff = \max(K - S_T, 0)$
*   **Moneyness (实虚值状态)**：
    *   **ITM (实值)**：具有内在价值 ($Call: S > K$)。
    *   **ATM (平值)**：$S \approx K$。
    *   **OTM (虚值)**：无内在价值 ($Call: S < K$)。

### 8.1.2 Black-Scholes-Merton (BSM) 模型

BSM 模型是现代期权定价的基石。对于欧式看涨期权 (European Call)，其定价公式为：

$$ C = S N(d_1) - K e^{-rT} N(d_2) $$

其中：

*   $N(\cdot)$：标准正态分布累积分布函数。
*   $d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$
*   $d_2 = d_1 - \sigma\sqrt{T}$
*   $\sigma$：标的资产收益率的波动率。

**模型洞察**：期权价格取决于五个变量：$S, K, T, r, \sigma$。其中前四个是市场可观测的，唯独**波动率 ($\sigma$)** 是未知的，需要估计。

## 8.2 希腊字母 (The Greeks)

希腊字母是期权价格关于各变量的偏导数，量化了期权的风险暴露。

### 8.2.1 Delta ($\Delta$)：方向风险

$$ \Delta = \frac{\partial C}{\partial S} $$

*   **含义**：标的价格变化 1 单位，期权价格变化多少。
*   **特性**：Call $\Delta \in (0, 1)$, Put $\Delta \in (-1, 0)$。ATM Call $\Delta \approx 0.5$。
*   **应用**：**Delta Neutral Hedging**。通过持有 $-N \times \Delta$ 份标的资产，使组合的总 Delta 为 0，从而免疫小幅价格波动风险，纯粹赚取时间价值或波动率收益。

### 8.2.2 Gamma ($\Gamma$)：凸性风险

$$ \Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{\partial \Delta}{\partial S} $$

*   **含义**：Delta 随标的价格的变化率。Gamma 越大，Delta 变化越快，对冲越困难。
*   **特性**：ATM 期权 Gamma 最大。临近到期时，ATM Gamma 会急剧飙升 (Pin Risk)。

### 8.2.3 Theta ($\Theta$)：时间衰减

$$ \Theta = \frac{\partial C}{\partial T} $$

*   **含义**：时间每流逝一天，期权价值损失多少。
*   **特性**：期权买方通常是 Theta 负值（消耗时间），卖方是 Theta 正值（赚取时间）。

### 8.2.4 Vega ($\nu$)：波动率风险

$$ \nu = \frac{\partial C}{\partial \sigma} $$

*   **含义**：波动率变化 1%，期权价格变化多少。
*   **特性**：长期限 (Long-term) 期权的 Vega 更大。

## 8.3 波动率曲面 (Volatility Surface)

### 8.3.1 隐含波动率 (Implied Volatility, IV)

如果我们把市场上的期权价格 $C_{market}$ 代入 BSM 公式，反推出的 $\sigma$ 即为**隐含波动率 (IV)**。IV 代表了市场对未来波动的预期。

### 8.3.2 波动率微笑 (Volatility Smile)

BSM 模型假设 $\sigma$ 为常数，但实际上，不同行权价 ($K$) 的期权 IV并不同：

*   **Smile/Skew**：通常 OTM Put 的 IV 高于 ATM，形成“偏斜 (Skew)”，反映了市场对暴跌风险的恐惧（黑天鹅定价）。
*   **Term Structure**：不同到期日的 IV 也不同。

## 8.4 策略示例：备兑看涨 (Covered Call)

这是一种最基础的收益增强策略，适合长期看好但认为短期横盘的标的。

**构建**：

1.  **持有标的** (Long Underlying)。
2.  **卖出 OTM Call** (Short Call)。

**逻辑**：备兑看涨的损益取决于标的走向。若标的上涨，收益会被行权价封顶 (Capped Upside)，因为卖出的 Call 让出了向上的超额空间；若标的横盘，策略则赚取 Call 的权利金 (Theta Income)，从而降低了持仓成本；若标的下跌，这笔权利金又提供了一定的安全垫 (Downside Buffer)，部分缓冲了标的的浮亏。

### 8.4.1 代码实现

```python
--8<-- "examples/textbook/ch08_options.py"
```

## 8.5 希腊字母深入 (Advanced Greeks)

除了 Delta, Gamma, Theta, Vega 这四个主要风险维度，专业交易员还需要关注二阶甚至三阶导数。

1.  **Vanna ($\frac{\partial \Delta}{\partial \sigma}$)**：
    *   Delta 对波动率的敏感度。
    *   **应用**：当波动率上升时，OTM Call 的 Delta 会增加（变得更有可能变为 ITM），而 ITM Call 的 Delta 会减小。做市商需要根据 Vanna 调整 Delta 对冲头寸。

2.  **Vomma ($\frac{\partial \nu}{\partial \sigma}$)**：
    *   Vega 对波动率的敏感度（Vega 的凸性）。
    *   **应用**：买入 Vomma（通常是买入 OTM 期权）可以在波动率飙升时获得加速度收益。

3.  **Charm ($\frac{\partial \Delta}{\partial T}$)**：
    *   Delta 对时间的敏感度。
    *   **应用**：随着到期日临近，OTM 期权的 Delta 会加速衰减至 0，ITM 期权的 Delta 会加速收敛至 1。周末效应（Weekend Effect）往往会导致 Charm 风险暴露。

## 8.6 常见期权策略组合

期权的魅力在于通过组合构建出任意形状的损益曲线 (Payoff)。

### 8.6.1 跨式组合 (Straddle)

跨式组合通过买入 ATM Call + 买入 ATM Put（相同 $K$, 相同 $T$）来构建。它表达的观点是**做多波动率**：交易者认为市场即将发生大行情（如财报发布、重大政策），但不确定方向，因而押注波动幅度而非涨跌。其风险则在于，如果市场横盘，就会损失全部权利金，因为此时 Theta 损耗极大。

### 8.6.2 宽跨式组合 (Strangle)

宽跨式组合改用买入 OTM Call + 买入 OTM Put 来构建。它的观点与 Straddle 相同，但因为用的是虚值期权，所以成本更低，相应地需要的波动幅度也更大。这使它常被用作一种彩票型策略，博取黑天鹅事件。

### 8.6.3 垂直价差 (Vertical Spread)

垂直价差通过一买一卖同类期权来表达方向观点。其中**牛市价差 (Bull Spread)** 买入低行权价 Call ($K_L$)、卖出高行权价 Call ($K_H$)；**熊市价差 (Bear Spread)** 则买入高行权价 Put ($K_H$)、卖出低行权价 Put ($K_L$)。这类组合的特点是收益有限、风险也有限：由于通过卖出期权降低了权利金成本，它成为方向性交易的首选。

### 8.6.4 铁鹰组合 (Iron Condor)

铁鹰组合由卖出 OTM Put Spread + 卖出 OTM Call Spread 构建而成，其中：

*   卖出 Put $K_1$ (低)，买入 Put $K_2$ (更低) 保护。
*   卖出 Call $K_3$ (高)，买入 Call $K_4$ (更高) 保护。

它表达的观点是**做空波动率**，即认为市场将在 $[K_1, K_3]$ 区间内震荡。因此它本质上是一种收租策略：只要标的不大涨大跌，就能稳赚权利金。

## 8.7 动态对冲：Gamma Scalping

这是一种利用 Gamma 属性，通过不断调整 Delta 对冲来获利的策略。

1.  **构建**：买入跨式组合 (Long Straddle)，保持 Delta 中性。
2.  **上涨时**：Gamma $> 0$，Delta 变大（如 $0 \rightarrow 0.2$）。为了保持中性，**卖出** 0.2 份标的。
3.  **下跌时**：Gamma $> 0$，Delta 变小（如 $0 \rightarrow -0.2$）。为了保持中性，**买入** 0.2 份标的。

**结果**：在对冲过程中，我们其实一直在**“高抛低吸”**标的资产，而这套机制的盈亏取决于波动是否足够。如果市场波动足够大，Gamma Scalping 赚取的利润将超过 Theta 损耗（权利金的时间衰减）；反之，如果市场死水一潭，Gamma 利润不足以覆盖 Theta 成本，策略就会亏损。

## 8.8 引擎配置与实战细节

### 8.8.1 合约配置

在 `AKQuant` 中，配置期权合约需指定 `option_type`, `strike_price` 和 `expiry_date`。

```python
from akquant import InstrumentConfig, OptionType

# 配置某个月份的购 4000 合约
opt_config = InstrumentConfig(
    symbol="MO2309-C-4000",
    asset_type="OPTION",
    option_type=OptionType.CALL,
    strike_price=4000.0,
    expiry_date="2023-09-15"
)
```

如果你的期权策略需要在到期后执行额外逻辑，例如记录行权/到期结算结果、移除失效合约后重建候选池，推荐实现 `on_expiry(event)`。该回调仅在引擎实际执行 `expiry_date` 驱动的到期结算/移除后触发。最小可运行示例见：`examples/49_on_expiry_demo.py`。

### 8.8.2 保证金计算

期权卖方（义务方）需要缴纳保证金。`AKQuant` 支持交易所标准的保证金计算公式：

$$ Margin = \text{权利金} + \max(12\% \times S - \text{虚值额}, 7\% \times S) $$

这意味着卖出期权的杠杆并不是固定的，而是随着标的价格变化而动态变化的。策略必须预留足够的现金以防**追加保证金 (Margin Call)**。

## 8.9 波动率套利 (Volatility Arbitrage)

波动率套利的核心在于交易**隐含波动率 (IV)** 与**已实现波动率 (RV)** 之间的差价。

$$ Profit \approx \text{Vega} \times (IV_{sold} - IV_{bought}) + \frac{1}{2} \text{Gamma} \times (RV^2 - IV^2) $$

一种典型做法是**做空波动率**：当 $IV > RV$ 时，卖出跨式组合 (Short Straddle) 并进行 Delta 对冲，只要市场实际波动小于 IV 预示的波动，就能赚取 Vega 差价。另一种是**Vanna-Volga 方法**，它利用市场上的三个主要报价（ATM, 25-Delta Call, 25-Delta Put）来构建整个波动率曲面，进而寻找定价错误的期权。

## 8.10 尾部风险对冲 (Tail Risk Hedging)

黑天鹅事件（如 2020 年疫情熔断）虽然罕见，但足以摧毁整个投资组合。

常见的尾部对冲手段有两类。其一是 **Put Buying**，即定期买入深度虚值 (Deep OTM) Put：它虽然长期亏损权利金（像买保险一样），但在崩盘时能获得百倍回报，对冲股票多头的亏损。其二是 **VIX Call**，即买入 VIX 看涨期权，其逻辑在于 VIX 指数通常与股市负相关。

## 本章小结

### 必须掌握

- 期权不是单纯押方向，而是在交易价格、时间与波动率的联合暴露。
- Greeks 是连接期权理论、风险管理与策略解释的核心桥梁。

### 理解即可

- 卖方策略、波动率策略与组合 Greeks 管理属于更高阶的实务延伸。

### 实践提醒

- 每次调参时优先记录执行价、期限与波动率口径，避免只看收益结果。

## 主线推进

贯穿全书的那条最小双均线 / 趋势策略，到这一章迎来了它最不一样的一次迁移：从“押方向”转向“押波动率与时间”。前几章里，这条主线在股票与衍生品市场上始终交易的是价格方向——金叉买入、死叉卖出，盈亏几乎只由标的涨跌决定。进入期权市场后，本章把同一套“信号 + 下单 + 风控”的框架接到了非线性资产上：原本的趋势判断可以转化为方向性的垂直价差或备兑看涨，而原本无处安放的“横盘观点”，现在能通过跨式、铁鹰这类组合直接表达成对波动率的多空。更重要的是，主线策略第一次需要在 Delta、Gamma、Theta、Vega 这组风险维度上同时管理暴露，而不再只盯收益曲线。至此，这条最小策略已经具备了在期权场景下被改写、被对冲、被组合的能力，为后续把它接入更复杂的多资产与波动率管理打下了基础。

## 延伸阅读

**经典著作**

- Hull, J. C. *Options, Futures, and Other Derivatives*（第 11 版），Pearson, 2022 —— 期权与衍生品定价的权威教科书，系统讲解 BSM 模型、Greeks 与波动率曲面，可与本章 8.1（定价理论）、8.2（希腊字母）、8.3（波动率曲面）对照精读。
- Natenberg, S. *Option Volatility & Pricing: Advanced Trading Strategies and Techniques*（第 2 版），McGraw-Hill, 2014 —— 从交易员视角讲解波动率交易与组合策略，直接对应本章 8.4（备兑看涨）、8.6（策略组合）、8.9（波动率套利）。
- Taleb, N. N. *Dynamic Hedging: Managing Vanilla and Exotic Options*，John Wiley & Sons, 1997 —— 做市商与套利者视角的动态对冲与高阶 Greeks 实务专著，延伸本章 8.5（希腊字母深入）、8.7（Gamma Scalping）与 8.10（尾部风险对冲）。

**官方文档与工具**

- [AKQuant 量化基础指南](../guide/quant_basics.md) —— 本章期权配置与回测的对应指南，配合 8.0（中国期权配置速览）与 8.8（引擎配置与实战细节）阅读。
- [AKShare 官方文档](https://akshare.akfamily.xyz/) —— 期权合约、行情与波动率相关数据的获取来源，支撑本章 8.3 与 8.9 的实证分析。

**本书相关**

- [第 7 章：期货市场与衍生品策略](07_futures.md) —— 本章 8.8.2（保证金计算）与卖方风险承接第 7 章的衍生品与风控基础。
- [第 5 章：策略开发实战](05_strategy.md) —— 本章 8.4 备兑看涨与 8.7 动态对冲沿用第 5 章建立的策略类结构与下单接口心智模型。

## 课后练习

### 基础题

1. 修改执行价或到期日，比较策略结果与 Greeks 变化。

### 应用题

1. 为示例策略增加一个基于 Delta 或 Vega 的风险约束。

### 综合题

1. 对比两种不同波动率环境下同一策略的收益风险特征。

??? note "参考答案要点（先独立思考再展开）"

    **基础题**：行权价越高（越虚值），权利金越低、Delta 越小；越临近到期，Theta 时间衰减越快，且 ATM 期权的 Gamma 会急剧上升。

    **应用题**：当组合 `|Delta|` 或 `Vega` 超过阈值时减仓或对冲（如加一条期货腿中和 Delta），把方向/波动率暴露控制在限额内。

    **综合题**：高 IV 环境下卖方权利金更厚但风险更大；低波动环境中备兑/卖方更易稳赚 Theta，而跨式买方易因 Theta 损耗亏损。

## 常见错误与排查

1. 定价偏差过大：检查无风险利率、到期时间和波动率输入。
2. 保证金不足：核对卖方策略的仓位规模与资金占用。
3. 风险暴露失控：优先检查 Delta、Gamma、Vega 是否超出阈值。
