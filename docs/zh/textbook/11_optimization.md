# 第 11 章：参数优化与稳健性检验

> ⏱️ 预计阅读 ~25 分钟 ｜ 🎯 难度 ★★★★☆（偏难）

量化策略通常包含若干参数（如均线周期、止损阈值）。参数的选择对策略表现有着决定性影响。本章将探讨如何通过科学的**参数优化 (Parameter Optimization)** 寻找最优解，并深入分析**过拟合 (Overfitting)** 的成因与防范措施。

## 学习目标

- 掌握参数优化、样本内外切分与稳健性检验的基本流程。
- 理解过拟合、偏差-方差权衡与多重检验问题。
- 能够解释为什么“最优参数”并不等于“可上线参数”。

## 前置知识

- 已完成第 10 章的回测评价与指标解读。
- 了解参数搜索和交叉验证的基础概念。

## 本章实践入口

- 主示例：[examples/textbook/ch11_optimization.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch11_optimization.py)
- 进阶示例：[examples/02_parameter_optimization.py](https://github.com/akfamily/akquant/blob/main/examples/02_parameter_optimization.py)
- 对应指南：[优化指南](../guide/optimization.md)

## 快速运行与验收

```bash
python examples/textbook/ch11_optimization.py
```

验收要点：

1. 脚本可完成参数搜索并输出最优参数组合。
2. 输出中可比较样本内与样本外表现差异。
3. 改变搜索范围后，最优参数与结果变化具有一致性。

## 11.1 参数优化理论

### 11.1.1 目标函数 (Objective Function)

参数优化的本质是一个数学规划问题：

$$ \max_{\theta \in \Theta} f(\theta | D_{train}) $$

其中：

*   $\theta$：策略参数向量。
*   $\Theta$：参数搜索空间。
*   $D_{train}$：训练集数据（样本内数据）。
*   $f$：目标函数，通常为夏普比率、卡玛比率或净利润。

### 11.1.2 搜索算法

在确定目标函数之后，剩下的问题是如何在参数空间里寻找最优解，常见的搜索算法可以按“计算代价”与“搜索效率”的权衡来理解。最朴素的是**网格搜索 (Grid Search)**，它穷举所有可能的参数组合，优点是能在离散网格上找到全局最优解，缺点则是计算量随参数数量指数级增长，也就是所谓的维数灾难。为缓解这一问题，**随机搜索 (Random Search)** 转而在参数空间内随机采样，由于不必逐格扫描，它在高维空间中的效率通常高于网格搜索。当参数空间更巨大且呈现非凸结构时，则可借助**遗传算法 (Genetic Algorithm)**，它模拟生物进化过程，通过变异和交叉寻找最优解，因而适合非凸优化问题。

## 11.2 过拟合：量化交易的隐形杀手

**过拟合 (Overfitting)** 指策略在样本内 (In-Sample) 表现极佳，但在样本外 (Out-of-Sample) 迅速失效的现象。

### 11.2.1 统计学原理

过拟合的本质是**多重假设检验 (Multiple Hypothesis Testing)** 的谬误。
假设我们在随机生成的噪声数据上测试 100 组参数，即使没有任何真实规律，我们也大概率能找到一组在 95% 置信水平下“显著有效”的参数。

$$ P(\text{至少一次伪显著}) = 1 - (1 - 0.05)^{100} \approx 99.4\% $$

这意味着，尝试的参数越多，找到“伪规律”的概率就越大。这被称为**数据窥探偏差 (Data Snooping Bias)**。

### 11.2.2 偏差-方差权衡 (Bias-Variance Tradeoff)

过拟合并非孤立现象，它处在偏差-方差权衡的一端。当模型太简单、无法捕捉市场规律时（如买入持有），就会出现**欠拟合 (High Bias)**；反过来，当模型太复杂、以至于记住了历史数据的噪音时（如用 10 个参数拟合 100 天的数据），则会陷入**过拟合 (High Variance)**。优化的目标因此不是一味追求拟合得更好，而是在这两个极端之间找到恰当的平衡点。

## 11.3 稳健性检验 (Robustness Testing)

为了检验策略是否过拟合，我们需要进行严格的稳健性测试。

### 11.3.1 样本外测试 (Out-of-Sample Testing)

样本外测试的做法是将历史数据分为**训练集 (Train Set)** 和**测试集 (Test Set)**，并对两者赋予截然不同的职责。其基本原则是：训练集用于优化参数，测试集仅用于验证，因此测试集数据在优化过程中必须严格不可见。据此可以给出一条判断标准——如果测试集夏普比率显著低于训练集（如衰减超过 50%），则存在过拟合。

### 11.3.2 参数敏感性分析 (Parameter Sensitivity)

优秀的策略应该落在**参数平原 (Parameter Plateau)** 上，而不是**参数尖峰 (Parameter Peak)** 上，二者的区别正在于对参数扰动的反应。所谓参数平原，是指参数发生微小变化（如均线从 20 变 21）时，绩效指标仍保持稳定；而参数尖峰则相反，参数微小变化就会引起绩效断崖式下跌，这通常意味着过拟合。在实践中，可以通过绘制**参数热力图 (Heatmap)** 来可视化参数敏感性，从而直观地判断策略究竟身处平原还是尖峰。

### 11.3.3 滚动回测 (Walk-Forward Analysis)

模拟真实交易中“定期重新优化”的过程。

1.  在 $T_0 \sim T_1$ 优化参数，在 $T_1 \sim T_2$ 使用该参数交易。
2.  在 $T_1 \sim T_2$ 重新优化参数，在 $T_2 \sim T_3$ 使用新参数交易。
3.  拼接所有测试段的资金曲线。

这是检验策略真实生命力的“金标准”。

## 11.4 AKQuant 优化实战

`AKQuant` 提供了 `run_grid_search` 函数，支持并行计算。

### 11.4.0 Windows 并行运行前置条件

当 `max_workers > 1` 时，Windows 使用多进程 `spawn` 模式，需满足以下条件：

*   策略类必须位于可导入模块中，不能直接定义在 `__main__`。
*   脚本入口必须加 `if __name__ == "__main__":`。
*   这类报错属于多进程序列化限制，不是成交策略语义错误。

示例：

```python
from akquant import run_grid_search
from my_strategy_module import TailTradingStrategy


def main() -> None:
    results = run_grid_search(
        strategy=TailTradingStrategy,
        param_grid=param_grid,
        data=all_data,
        max_workers=4,
    )
    print(results.head())


if __name__ == "__main__":
    main()
```

### 11.4.1 参数模型驱动（适合页面化）

在面向页面配置、策略市场、研究平台等场景中，推荐直接用**内联参数字段**声明
参数：

*   **参数声明层**：在策略类体内用 `IntParam` / `FloatParam` / ... 内联声明字段。
    *   用于参数类型约束、默认值管理、前端 JSON Schema 导出；经
        `self.params.<name>` 只读访问，`self.params` 在实例构造期即已就绪。
*   **优化搜索层**：继续使用 `run_grid_search(param_grid=...)`。
    *   `param_grid` 只负责离散候选值，键名须与内联字段名一致，不承担复杂对象校验。

推荐这样做的核心原因是：**运行参数校验**与**参数组合搜索**在职责上是不同问题，
但共享同一套字段声明，拆开后更清晰、更稳健，也不需要再单独维护一个
`ParamModel` 子类。

策略示意（节选）：

```python
from akquant import IntParam, Strategy


class SmaStrategy(Strategy):
    fast_period = IntParam(10, ge=2, le=200)
    slow_period = IntParam(30, ge=3, le=500)
```

### 11.4.2 代码示例

```python
--8<-- "examples/textbook/ch11_optimization.py"
```

### 11.4.2A 新参数（并行日志与严格参数校验）

为提升优化可观测性与结果可靠性，推荐关注以下参数：

*   `forward_worker_logs`（`run_grid_search`）：
    *   `False`（默认）：性能优先，子进程日志可能在主进程不可见；
    *   `True`：将子进程 `self.log()` 聚合回主进程，适合排障与教学演示。
*   `strict_strategy_params`（`run_backtest`，默认 `True`）：
    *   严格校验策略内联参数字段（`IntParam`/`FloatParam`/... 声明）；
    *   当 `param_grid` 中存在策略未声明的参数、或取值越界时，立即抛错，避免静默回退导致“看似跑完但结果无效”。
*   `run_walk_forward` 也支持通过 `**kwargs` 透传这两个参数：
    *   `forward_worker_logs` 作用于样本内优化阶段（内部 `run_grid_search`）；
    *   `strict_strategy_params` 在样本内优化与样本外验证阶段都生效。

示例：

```python
results = run_grid_search(
    strategy=TailTradingStrategy,
    param_grid=param_grid,
    data=all_data,
    max_workers=4,
    forward_worker_logs=True,
)
```

WFO 传导示例：

```python
wfo_results = run_walk_forward(
    strategy=TailTradingStrategy,
    param_grid=param_grid,
    data=all_data,
    train_period=250,
    test_period=60,
    max_workers=4,
    forward_worker_logs=True,
    strict_strategy_params=True,
)
```

### 11.4.3 结果分析

运行上述代码后，我们会得到一个按夏普比率排序的参数表。

*   **观察前 10 名**：如果前 10 名参数比较集中（如均线都在 20-25 之间），说明策略比较稳健。
*   **观察分布**：如果最优参数东一榔头西一棒子，说明策略可能在拟合噪音。

## 11.5 组合净化交叉验证 (Combinatorial Purged Cross-Validation, CPCV)

这是由 De Prado 提出的目前最先进的回测框架。

### 11.5.1 为什么需要 CPCV？

要理解 CPCV 的价值，不妨先看既有方法各自的局限。传统的 **K-Fold** 交叉验证用在金融场景时，会因为时间序列的相关性而导致信息泄露。**Walk-Forward** 虽然避免了信息泄露，却只测试了一条历史路径——一旦历史重演的方式略有不同，策略可能就失效了。CPCV 正是为了同时克服这两点不足而提出的。

### 11.5.2 CPCV 原理

CPCV 将数据切分为 $N$ 组，每次选取 $k$ 组作为测试集（共有 $C_N^k$ 种组合）。在训练集和测试集之间进行**净化 (Purging)** 和 **隔离 (Embargo)**。

通过这种方式，我们可以生成大量可能的“历史路径”。

*   **路径生成**：将所有测试集的预测结果按时间拼接，可以重组出 $C_N^k$ 条完整的资金曲线。
*   **概率分布**：我们可以得到策略夏普比率的分布，而不是单一的数值。这让我们能回答：“在 95% 的概率下，该策略的夏普比率大于 1.0 吗？”

## 11.6 调整后的夏普比率 (Deflated Sharpe Ratio, DSR)

如果你尝试了 1000 组参数，终于找到了一组夏普比率为 2.0 的参数。这个 2.0 是真实的吗？

**Deflated Sharpe Ratio (DSR)** 用于修正**多重测试偏差 (Multiple Testing Bias)**。它在概率夏普比率 (PSR) 的基础上，进一步考虑了**尝试次数 (Number of Trials)** 的影响。

$$ DSR = PSR(\widehat{SR}, SR_{benchmark}) $$

其中基准夏普 $SR_{benchmark}$ 不再是 0，而是根据尝试次数 $K$ 和由于尝试次数增多而导致的预期最大夏普比率 $E[\max(SR)_K]$ 计算得出的。

$$ E[\max(SR)_K] \approx E[SR] + \sigma_{SR} \sqrt{2 \ln K} $$

这意味着：**尝试的次数越多，你就应该要求越高的夏普比率，才能确信这不是运气。**

## 11.7 高级优化算法

除了简单的网格搜索，量化领域还常用以下高级算法：

### 11.7.1 遗传算法 (Genetic Algorithm, GA)

模拟生物进化过程，适用于参数空间巨大且非凸的优化问题。

1.  **种群初始化**：随机生成 $N$ 个策略个体（参数组合）。
2.  **适应度评估**：回测每个个体，计算夏普比率作为适应度。
3.  **选择 (Selection)**：优胜劣汰，保留高夏普的个体。
4.  **交叉 (Crossover)**：交换两个父代个体的参数片段，生成子代。
5.  **变异 (Mutation)**：随机微调某些参数，引入多样性，防止陷入局部最优。
6.  **迭代**：重复步骤 2-5，直到满足停止条件。

### 11.7.2 贝叶斯优化 (Bayesian Optimization)

适用于计算昂贵的优化问题（例如每次回测需要 1 小时）。

它通过构建一个**代理模型 (Surrogate Model)**（通常是高斯过程 Gaussian Process）来拟合目标函数 $f(\theta)$。

在这一框架中，真正驱动搜索的是**采集函数 (Acquisition Function)**：它根据代理模型的预测值和不确定性，决定下一步去哪里采样、尝试哪组参数。其精妙之处在于通过算法在**开发 (Exploitation)**（去已知表现好的地方）和 **探索 (Exploration)**（去未知的地方）之间进行平衡。正因如此，贝叶斯优化通常能比网格搜索快 10-100 倍找到全局最优解。

## 11.8 目标函数的选择

优化什么指标，决定了你会得到什么样的策略。

1.  **最大化夏普比率**：追求风险调整后收益。最常用。
2.  **最大化净利润**：追求绝对收益。容易导致高波动、大回撤的策略。
3.  **最大化卡玛比率**：追求低回撤。适合风险厌恶型资金。
4.  **多目标优化 (Multi-Objective Optimization)**：寻找**帕累托前沿 (Pareto Frontier)**。例如，同时追求高收益和低回撤。这种方法不会给出一个“最优解”，而是一组“非劣解”，供基金经理根据当前的市场观点和风险偏好进行选择。

## 11.9 聚类分析 (Cluster Analysis)

De Prado 提出了一种名为 **ONC (Optimal Number of Clusters)** 的算法，用于从一堆策略中筛选出真正互补的策略。

1.  **相关性矩阵**：计算 $N$ 个策略回测收益率的相关性矩阵。
2.  **聚类**：将高相关性的策略聚为一类（例如所有的趋势策略聚在一起，所有的反转策略聚在一起）。
3.  **筛选**：在每一类中只保留表现最好的一个策略。

这样构建出来的投资组合，其夏普比率通常远高于简单的等权组合，因为我们真正实现了**风险分散**。

## 11.10 回测长度与显著性 (Minimum Backtest Length)

你需要多少年的数据才能证明策略有效？

$$ MinTRL \approx \frac{1.25}{\widehat{SR}^2} \left( \frac{\Phi^{-1}(1-\alpha) - \Phi^{-1}(\beta) \sqrt{1-\rho}}{1-\rho} \right)^2 $$

简单来说，策略的夏普比率越低，所需的验证时间就越长。

*   夏普 0.5 的策略，可能需要 50 年的数据才能通过统计检验。
*   夏普 2.0 的策略，可能只需要 2-3 年的数据。

这意味着，对于低频策略（通常夏普较低），你需要极长的历史数据；而对于高频策略（通常夏普较高），短期的验证可能就足够了。

## 11.11 “抽屉问题” (The File Drawer Problem)

学术界和业界都存在一种严重的**发表偏差 (Publication Bias)**。

*   研究员尝试了 100 个策略，其中 95 个失败了，被扔进了“抽屉”。
*   只有 5 个成功的策略被写进了论文或展示给客户。
*   由于读者看不到那 95 个失败的案例，他们会误以为这 5 个策略非常完美。

**Deflated Sharpe Ratio** 正是为了解决这个问题，它要求你在评估那 5 个成功策略时，必须考虑到背后还有 95 个失败的尝试。

## 本章小结

### 必须掌握

- 优化的目标不是找到最好看的回测，而是找到更稳健的参数区域。
- 样本外验证、参数敏感性与 DSR/CPCV 是防过拟合的重要工具。

### 理解即可

- 遗传算法、贝叶斯优化等高级方法只有在验证框架可靠时才有意义。

### 实践提醒

- 先定义好验证流程与保存规范，再启动大规模参数扫描。

## 主线推进

贯穿全书的那条最小多均线 / 趋势策略，到本章经历了一次从“能用”到“可信”的检验。第 1 章让它跑通回测闭环，第 4、5 章把它重写成标准策略类并补齐止损与风控，第 9 章又把它放进多资产配置框架，第 10 章则为它建立了一整套评价与可视化口径。本章承接这条线索，正面回应一个此前被悬置的问题：那条策略里反复出现的均线周期、阈值等参数，究竟该如何选、选出来的“最优”又是否可信。我们因此先用 `run_grid_search` 在参数空间里搜索它的候选组合，再用样本外测试、参数敏感性热力图与滚动回测 (Walk-Forward) 审视它是落在参数平原还是参数尖峰；更进一步，CPCV 把单一历史路径扩展为夏普比率的概率分布，DSR 与最小回测长度则提醒我们在多次尝试之后必须抬高对夏普的要求。至此，主线从“在一组参数上得到一条漂亮的资金曲线”推进到了“为这条策略找到稳健的参数区域并量化其可信度”，为下一步把它接入因子化、机器学习增强与实盘部署打下了方法论基础。

## 延伸阅读

**经典著作**

- Bailey, D. H., & López de Prado, M. "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality," *The Journal of Portfolio Management*, 40(5), 2014, 94–107 —— DSR 的原始文献，在 PSR 基础上引入尝试次数修正多重测试偏差，直接对应本章 11.6（DSR）与 11.11（抽屉问题）。
- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance," *Notices of the American Mathematical Society*, 61(5), 2014, 458–471 —— 证明尝试足够多的参数即可轻易“制造”出高夏普回测，并给出最小回测长度 (MinBTL)，对应本章 11.2（过拟合）与 11.10（回测长度与显著性）。
- López de Prado, M. *Advances in Financial Machine Learning*，John Wiley & Sons, 2018 —— 第 7、12 章系统给出净化 K-Fold 与组合净化交叉验证 (CPCV)，第 8 章讨论特征重要性与多重检验，覆盖本章 11.3.3（滚动回测）、11.5（CPCV）与 11.9（ONC 聚类）。
- Bishop, C. M. *Pattern Recognition and Machine Learning*，Springer, 2006 —— 第 1、3 章对偏差-方差权衡、模型复杂度与过拟合给出经典统计学习视角，对应本章 11.2.2（偏差-方差权衡）。

**官方文档与工具**

- [AKQuant 优化指南](../guide/optimization.md) —— `run_grid_search`、`run_walk_forward` 等优化接口与并行参数的权威说明，对应本章 11.4。
- [scikit-learn: Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html) —— 网格搜索、随机搜索与交叉验证的标准实现参考，对应本章 11.1.2 与 11.3。
- [scikit-optimize 文档](https://scikit-optimize.github.io/) —— 基于高斯过程的贝叶斯优化与采集函数实现，对应本章 11.7.2。

**本书相关**

- [第 10 章：策略评价体系与风险指标](10_analysis.md) —— 本章 11.1.1 目标函数所用的夏普、卡玛等指标，其定义与解读承接第 10 章。
- [第 5 章：策略开发实战](05_strategy.md) —— 本章 11.4 的参数搜索复用第 5 章建立的标准策略类结构与生命周期回调。

## 课后练习

### 基础题

1. 修改参数网格范围，观察最优参数是否稳定。

### 应用题

1. 对一个策略增加一次滚动回测并记录样本外表现。

### 综合题

1. 写出一份简化优化报告，说明哪些结果可信、哪些结果需要保留意见。

??? note "参考答案要点（先独立思考再展开）"

    **基础题**：稳健策略的最优参数应落在"参数平原"（邻域绩效相近）；若最优参数东一榔头西一棒，多半是在拟合噪声。

    **应用题**：用 `run_walk_forward` 分段优化 + 样本外验证，重点看样本外夏普相对样本内的衰减幅度（衰减超 50% 要警惕）。

    **综合题**：可信 = 样本外稳定 + 处于参数平原 + 通过 DSR；需保留意见 = 尝试次数多却未做多重检验校正、或样本长度不足（参见最小回测长度）。

## 常见错误与排查

1. 样本内过高收益：优先检查是否发生参数数据泄漏。
2. 样本外断崖下滑：确认是否存在过窄参数空间或过拟合。
3. 结果不可复现：固定随机种子并记录数据与代码版本。
