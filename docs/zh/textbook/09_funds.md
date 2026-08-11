# 第 9 章：基金投资与资产配置理论

> ⏱️ 预计阅读 ~30 分钟 ｜ 🎯 难度 ★★★★☆（偏难）

本章将视野从单一资产（股票、期货、期权）扩展到**投资组合 (Portfolio)**。对于大资金管理而言，**资产配置 (Asset Allocation)** 决定了 90% 以上的长期收益波动。本章将深入探讨场内基金 (ETF/LOF) 和可转债 (Convertible Bonds) 的微观机制，并基于现代投资组合理论 (MPT) 构建科学的配置策略。

## 学习目标

- 理解 ETF、可转债与资产配置策略在教学与实务中的典型定位。
- 掌握固定比例、风险平价等组合构建思路。
- 能够解释基金类策略为什么适合做跨资产配置实验。

## 前置知识

- 已掌握股票与回测基础。
- 了解基金净值、再平衡与组合波动的基本概念。

## 本章实践入口

- 主示例：[examples/textbook/ch09_funds.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch09_funds.py)
- 进阶示例：[examples/textbook/ch09_portfolio.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch09_portfolio.py)
- 函数式孪生示例：[examples/textbook/ch09_funds_functional.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch09_funds_functional.py)（主示例孪生）、[examples/textbook/ch09_portfolio_functional.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch09_portfolio_functional.py)（进阶示例孪生）
- 对应指南：[策略指南](../guide/strategy.md)

## 快速运行与验收

```bash
python examples/textbook/ch09_funds.py
python examples/textbook/ch09_funds_functional.py
python examples/textbook/ch09_portfolio.py
python examples/textbook/ch09_portfolio_functional.py
```

验收要点：

1. 两个脚本都可独立运行并输出组合或策略结果。
2. 可对比单资产策略与组合配置策略的回撤差异。
3. 修改资产权重后，收益风险结构变化符合分散化预期。

## 9.1 场内基金 (Exchange Traded Funds)

### 9.1.1 交易机制与套利

**ETF (Exchange Traded Fund)** 是在交易所上市交易的开放式基金，其核心特征是**一级市场申赎**与**二级市场买卖**并存。

*   **IOPV (Intraday Indicative Optimized Portfolio Value)**：基金份额参考净值，由交易所实时计算并发布（通常每 15 秒一次）。
*   **折溢价套利 (Arbitrage)**：
    *   **溢价套利 (Premium)**：$Price > IOPV$。买入一篮子股票 $\rightarrow$ 申购 ETF 份额 $\rightarrow$ 卖出 ETF 份额。
    *   **折价套利 (Discount)**：$Price < IOPV$。买入 ETF 份额 $\rightarrow$ 赎回一篮子股票 $\rightarrow$ 卖出股票。

这种套利机制保证了 ETF 价格紧贴净值波动，使 ETF 成为跟踪指数最有效的工具。

### 9.1.2 常用策略：网格交易 (Grid Trading)

网格交易是一种利用行情震荡获利的自动化策略，特别适合波动率高且长期不退市的宽基 ETF。它的核心逻辑是将价格区间划分为若干网格，执行动作则相应地遵循“下跌触网买入，上涨触网卖出”的纪律。从数学本质上看，它是在均值回归 (Mean Reversion) 的过程中，通过高频低吸高抛，赚取波动率收益 (Volatility Harvesting)。

```python
--8<-- "examples/textbook/ch09_funds.py"
```

## 9.2 可转债 (Convertible Bonds)

可转债是一种兼具**债性 (Bond-like)** 和**股性 (Equity-like)** 的混合融资工具。持有者有权在特定时期内按转股价将其转换为公司股票。

### 9.2.1 定价模型

可转债价格 $P_{CB}$ 可以分解为：

$$ P_{CB} = P_{Bond} + P_{Option} $$

其中**纯债价值 ($P_{Bond}$)** 是未来现金流（利息+本金）的贴现值，它构成了转债价格的**债底 (Bond Floor)**，因而提供了极高的安全边际；**期权价值 ($P_{Option}$)** 则是看涨期权 (Call Option) 的价值，当正股上涨时，转债价格便随之上涨。

### 9.2.2 核心条款 (Clauses)

中国市场的可转债具有独特的“博弈条款”：

1.  **下修条款 (Downward Revision)**：当正股价持续低迷时，公司可下调转股价。这相当于降低了 Call Option 的行权价，提升了期权价值（送钱条款）。
2.  **强赎条款 (Redemption)**：当正股价持续高于转股价 130% 时，公司可强制赎回。这迫使持有人转股，实现“债转股”的融资目的。
3.  **回售条款 (Put)**：当正股价持续低迷且临近到期时，持有人可将债券按面值加利息回售给公司。

### 9.2.3 策略：双低轮动 (Double-Low Rotation)

**双低**指“低价格 + 低溢价率”，这两个维度恰好对应了转债的攻守两端：**低价格**意味着贴近债底，防守性强；**低转股溢价率**意味着股性强，进攻性好。基于这一点的轮动逻辑是，构建双低转债组合，定期剔除不再双低的标的，买入新的双低标的，从而实现“高抛低吸”。

## 9.3 现代投资组合理论 (Modern Portfolio Theory, MPT)

MPT 由 Harry Markowitz 提出，其核心思想是通过分散化 (Diversification) 降低非系统性风险。换言之，把资金分散到彼此并不完全同涨同跌的资产上，组合整体的波动会低于各资产波动的简单加权，这正是分散化带来的“免费午餐”。

### 9.3.1 均值-方差优化 (Mean-Variance Optimization)

假设投资组合由 $N$ 个资产组成，权重向量为 $w = [w_1, w_2, ..., w_N]^T$。

*   **组合预期收益**：$E(R_p) = w^T \mu$
*   **组合方差**：$\sigma_p^2 = w^T \Sigma w$
    其中 $\mu$ 为资产预期收益率向量，$\Sigma$ 为资产收益率的协方差矩阵 (Covariance Matrix)。

**优化目标**：
在给定风险水平 $\sigma_{target}$ 下，最大化预期收益：

$$ \max_w w^T \mu \quad \text{s.t.} \quad w^T \Sigma w = \sigma_{target}^2, \sum w_i = 1 $$

### 9.3.2 有效前沿 (Efficient Frontier)

所有最优组合在“风险-收益”平面上构成的曲线称为**有效前沿**。在这条曲线上有两个具有特殊意义的点：一个是**全局最小方差组合 (GMVP)**，即风险最低的点；另一个是**切点组合 (Tangency Portfolio)**，即夏普比率 (Sharpe Ratio) 最高的点。

## 9.4 资产配置策略实战

### 9.4.1 固定比例配置 (Fixed Weight)

最经典的策略是 **60/40 股债平衡**：其中 60% 配置股票 ETF (如沪深300) 负责进攻，40% 配置债券 ETF (如国债指数) 负责防守。这一策略的关键在于**再平衡 (Rebalancing)**——定期（如每季度）将仓位恢复至 60/40，从而强制实现了“卖高买低”的纪律。

### 9.4.2 风险平价 (Risk Parity)

风险平价是桥水基金 (Bridgewater) 全天候策略的核心。它不追求资金等权，而是追求**风险贡献等权 (Equal Risk Contribution)**。由于债券波动率远低于股票，为了实现风险平价，通常需要对债券加杠杆。

## 9.5 风险平价深入 (Risk Parity Deep Dive)

风险平价的提出，正是为了纠正传统 60/40 组合的一个隐蔽缺陷：它看似分散，实则 90% 的风险都来自于股票（因为股票波动率远大于债券）。这意味着当股市崩盘时，债券的那点收益根本无法对冲股票的亏损。

### 9.5.1 风险贡献 (Risk Contribution, RC)

组合的总风险 $\sigma_p$ 可以分解为各资产的风险贡献之和：

$$ \sigma_p = \sum_{i=1}^N RC_i = \sum_{i=1}^N w_i \frac{\partial \sigma_p}{\partial w_i} $$

风险平价的目标是找到一组权重 $w$，使得所有资产的 $RC_i$ 相等。

$$ RC_1 = RC_2 = ... = RC_N = \frac{\sigma_p}{N} $$

### 9.5.2 优化求解

求解风险平价权重本质上是一个非线性凸优化问题。但在资产相关性为 0 的简化假设下，可以得到一个直观的近似结论——权重与波动率成反比：

$$ w_i \propto \frac{1}{\sigma_i} $$

这一结论的实战意义是：对**低波动资产（债券）**配以高权重（或加杠杆），对**高波动资产（股票）**配以低权重。这样一来，无论哪类资产发生波动，对组合净值的影响都是相同的。

## 9.6 Black-Litterman 模型

Markowitz 的均值-方差模型在实战中非常敏感：稍微修改一下预期收益率 $\mu$，优化出来的权重就会剧烈变化（Corner Solution，即全仓某一个资产）。Black-Litterman 模型通过贝叶斯框架解决了这个问题，其思路是把市场均衡与投资者观点逐步融合。

首先是**先验 (Prior)**，即市场均衡状态下的预期收益（Implied Returns）——假设当前市场权重就是最优的，反推大家的预期收益是多少。其次是**观点 (Views)**，即投资者的主观观点（如“我看好科技股跑赢大盘 5%”），并附带置信度（Confidence）。最后是**后验 (Posterior)**，将先验与观点融合，得到新的预期收益向量 $\mu_{BL}$ 和协方差矩阵 $\Sigma_{BL}$。

再用新的 $\mu_{BL}$ 和 $\Sigma_{BL}$ 进行均值-方差优化，得到的权重既尊重了市场，又体现了个人观点，且非常稳健。

## 9.7 分层风险平价 (Hierarchical Risk Parity, HRP)

López de Prado 指出，传统的风险平价依赖于协方差矩阵的逆矩阵 $\Sigma^{-1}$，这在样本外极不稳定。HRP 的应对之道是借用机器学习中的**聚类 (Clustering)** 算法，分三步构建权重。

第一步是**聚类**：将相关性高的资产聚为一类（如所有科技股聚在一起，所有债券聚在一起），构建一棵层次树 (Dendrogram)。第二步是**递归二分**：从树的根节点开始，将资金在两个子类之间分配，分配比例由两个子类的方差决定。第三步是**自上而下**地逐层分配，直到每个叶子节点（具体资产）都分到了权重。

**优势**：HRP 不需要对协方差矩阵求逆，因此对噪音不敏感，样本外表现通常优于 Markowitz 和传统风险平价。

## 9.8 代码实现：股债轮动

下面的代码展示了如何在 `AKQuant` 中构建一个包含股票和债券 ETF 的投资组合，并定期进行再平衡。

```python
--8<-- "examples/textbook/ch09_portfolio.py"
```

## 9.9 FOF 组合管理 (Fund of Funds)

FOF 策略不是直接买卖股票，而是配置一篮子基金。

1.  **核心/卫星策略 (Core-Satellite)**：
    *   **核心**：配置宽基指数 ETF（如沪深300），获取市场平均收益 (Beta)。
    *   **卫星**：配置行业主题 ETF（如半导体、医药）或主动管理基金，获取超额收益 (Alpha)。
2.  **基金优选**：
    *   **4P 分析法**：People (团队), Philosophy (理念), Process (流程), Performance (业绩)。
    *   **量化筛选**：夏普比率、最大回撤、Calmar 比率持续稳定。

## 9.10 流动性管理 (Liquidity Management)

对于大资金，流动性是必须考虑的约束。

*   **冲击成本**：买入成交量过小的 ETF 会导致巨大的滑点。
*   **变现能力**：在市场恐慌时，能否迅速将资产变现？
    *   **Tier 1**：现金、货币基金。
    *   **Tier 2**：国债 ETF、沪深300 ETF。
    *   **Tier 3**：小盘股、信用债、私募股权 (PE)。

## 本章小结

### 必须掌握

- 基金与资产配置策略的重点在于组合层风险收益权衡，而非单一择时信号。
- 再平衡规则、资产相关性与流动性约束会共同决定组合表现。

### 理解即可

- Black-Litterman、HRP 与 FOF 管理提供了更高阶的配置框架。

### 实践提醒

- 先验证简单可复现的配置框架，再逐步叠加更复杂的优化模型。

## 主线推进

贯穿全书的那条最小多均线 / 趋势策略，到本章迎来了一次视角的跃迁：此前各章始终把它当作作用于**单一资产**的择时信号来打磨——第 1 章跑通回测闭环，第 4、5 章把它重写成事件驱动的标准策略类并补上止损与风控。本章则把舞台从单一标的扩展到**投资组合**，让主线策略不再孤军作战。具体而言，我们把它放进一个多资产配置框架：一方面，它可以作为 ETF 网格或股债轮动中“进攻腿”的择时引擎；另一方面，组合层的固定比例、风险平价乃至 Black-Litterman、HRP 等配置规则，又在它之上加了一层关于“在不同资产间如何分配资金、如何再平衡”的决策。至此，主线从“一个资产上的进出场判断”推进到了“多资产组合的风险预算与配置纪律”，为后续把它接入评价、优化与实盘环节铺好了从单标的到组合的过渡。

## 延伸阅读

**经典著作**

- Markowitz, H. M. "Portfolio Selection," *The Journal of Finance*, 7(1), 1952, 77–91 —— 现代投资组合理论的奠基论文，首次以“均值-方差”框架刻画分散化与风险-收益权衡，直接对应本章 9.3（MPT）。
- Black, F., & Litterman, R. "Global Portfolio Optimization," *Financial Analysts Journal*, 48(5), 1992, 28–43 —— Black-Litterman 模型的原始文献，以市场均衡先验融合投资者观点，对应本章 9.6。
- López de Prado, M. "Building Diversified Portfolios that Outperform Out of Sample," *The Journal of Portfolio Management*, 42(4), 2016, 59–69 —— 提出分层风险平价 (HRP)，用层次聚类替代协方差矩阵求逆，对应本章 9.7。
- López de Prado, M. *Advances in Financial Machine Learning*，John Wiley & Sons, 2018 —— 第 16 章“Machine Learning Asset Allocation”系统给出 HRP 算法与实现，延伸本章 9.5（风险平价）与 9.7（HRP）。

**官方文档与工具**

- [AKShare 官方文档](https://akshare.akfamily.xyz/) —— 获取 ETF、可转债、基金净值等本章所需数据的主要来源，对应本章 9.1、9.2。
- [AKQuant 策略指南](../guide/strategy.md) —— 组合调仓、再平衡与 `order_target_percent` 等接口的权威说明，对应本章 9.4、9.8。

**本书相关**

- [第 1 章：量化投资概述与环境搭建](01_foundations.md) —— 本章 9.3 的 MPT 与第 1 章 1.1.2 的 CAPM/多因子理论基础一脉相承。
- [第 5 章：策略开发实战](05_strategy.md) —— 本章 9.8 的股债轮动组合复用第 5 章建立的策略类结构与下单接口。

## 课后练习

### 基础题

1. 修改固定权重配置比例，比较组合波动率变化。

### 应用题

1. 将风险平价与固定权重策略做一次样本内外对比。

### 综合题

1. 设计一个包含 ETF 与可转债的简化多资产配置实验。

??? note "参考答案要点（先独立思考再展开）"

    **基础题**：股票权重越高，组合波动率越大（股票主导风险）——传统 60/40 组合约 90% 的风险其实来自股票那 60%。

    **应用题**：风险平价在样本外通常更稳（风险被真正分散），固定权重样本内可能收益更高但波动更大；用夏普与最大回撤对比。

    **综合题**：用 ETF 作进攻/防守腿、双低可转债兼顾攻守，按固定比例或风险平价配置并定期再平衡。

## 常见错误与排查

1. 组合波动异常：检查权重是否归一化且无超范围配置。
2. 结果不稳定：确认样本区间是否覆盖不同市场状态。
3. 流动性不足：核对标的成交额与策略交易规模匹配度。
