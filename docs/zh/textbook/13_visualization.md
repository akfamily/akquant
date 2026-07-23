# 第 13 章：策略可视化与报表分析

> ⏱️ 预计阅读 ~20 分钟 ｜ 🎯 难度 ★★★☆☆（进阶）

数据可视化 (Data Visualization) 不仅是展示结果的手段，更是**探索性数据分析 (Exploratory Data Analysis, EDA)** 的核心工具。通过高质量的图表，我们可以直观地识别策略的风险特征、收益来源以及潜在的过拟合迹象。本章将介绍如何使用 `AKQuant` 及第三方工具生成专业的量化回测报告。

## 学习目标

- 掌握权益曲线、回撤、收益分布与交易行为图的基本读法。
- 理解可视化在策略诊断、复盘与沟通中的作用。
- 能够把报告图表与风险解释对应起来。

## 前置知识

- 已完成第 10 章的指标分析基础。
- 了解 Matplotlib、Plotly 或基础图表阅读方式。

## 本章实践入口

- 主示例：[examples/textbook/ch13_visualization.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch13_visualization.py)
- 进阶示例：[examples/11_plot_visualization.py](https://github.com/akfamily/akquant/blob/main/examples/11_plot_visualization.py)
- 对应指南：[可视化指南](../guide/visualization.md)

## 快速运行与验收

```bash
python examples/textbook/ch13_visualization.py
```

验收要点：

1. 脚本可生成至少一组回测图表或可视化产物。
2. 图表包含收益曲线与回撤维度信息。
3. 修改参数后图表形态变化可用于解释策略风险特征。

## 13.1 可视化原则与核心图表

### 13.1.1 权益曲线 (Equity Curve)

权益曲线是最基础的图表，展示账户总资产随时间的变化，而坐标系的选择会直接影响解读视角。**线性坐标 (Linear Scale)** 适合短期回测；**对数坐标 (Logarithmic Scale)** 则更适合长期回测，因为在对数坐标下，直线的斜率代表复利增长率，且能清晰展示早期的波动，避免它被后期的指数增长所掩盖。

### 13.1.2 水下曲线 (Underwater Plot)

水下曲线专门用于展示**回撤 (Drawdown)** 的深度和持续时间。其 **Y 轴**表示当前净值距离历史最高点的百分比跌幅（0% ~ -100%），而**阴影面积**则反映了投资者承受痛苦的“时间和空间”。分析时应重点关注**回撤修复期 (Recovery Period)**：如果修复期过长（如超过 1 年），就说明策略可能已经失效。

### 13.1.3 收益分布图 (Return Distribution)

收益分布图展示日收益率的直方图 (Histogram) 和核密度估计 (KDE)，从中可以读出两个关键特征。其一是**尖峰肥尾 (Fat Tails)**，这是金融数据的典型特征，分析时尤其要关注分布的**左尾 (Left Tail)**，那正是“黑天鹅”藏身之处。其二是**偏度 (Skewness)**：**正偏 (Positive Skew)** 对应“小亏大赚”，是趋势策略的典型形态；**负偏 (Negative Skew)** 则对应“小赚大亏”，常见于套利策略或卖出期权。

## 13.2 高级分析图表

### 13.2.1 月度热力图 (Monthly Heatmap)

月度热力图将收益率按年份和月份排列成矩阵，用颜色深浅表示收益高低，主要**用途**是识别**季节性 (Seasonality)** 和**策略衰退**。从图形**特征**上看，如果某一年份全是绿色（亏损），就可能意味着市场风格发生了根本性转变。

### 13.2.2 滚动指标 (Rolling Metrics)

静态指标（如全周期夏普）可能掩盖局部的剧烈波动，因此需要借助滚动指标观察其随时间的变化。其中，**滚动波动率**用于观察市场恐慌时策略的风险暴露，**滚动夏普**则用于观察策略表现的稳定性。

## 13.3 AKQuant 内置绘图工具

`AKQuant` 提供了统一的 Plotly 报告接口，可直接输出交互式 HTML，并内置基准对比分析模块。

### 13.3.1 基础绘图

```python
import akquant as aq

# 运行回测
result = aq.run_backtest(...)

# 生成交互式仪表盘
fig = result.viz.dashboard(show=False, title="Strategy Dashboard")
fig.write_html("dashboard.html")
```

### 13.3.2 策略报告与基准对比 (Plotly)

`result.viz.report` 会生成整合版策略报告，默认包含：

1. 核心指标与权益回撤图
2. 收益分布与滚动指标
3. 交易复盘图（提供行情数据时）
4. 基准对比模块（传入 `benchmark` 时）

```python
benchmark_returns = (
    benchmark_df.set_index("date")["close"].pct_change().fillna(0.0)
)

result.viz.report(
    title="Alpha Strategy Report",
    filename="akquant_report.html",
    show=False,
    benchmark=benchmark_returns,
)
```

基准对比模块会展示以下相对指标：

- 累计超额收益 (Total Excess)
- 年化超额收益 (Annual Excess)
- 跟踪误差 (Tracking Error)
- 信息比率 (Information Ratio)
- Beta / Alpha

### 13.3.3 结构化 Benchmark Analysis 与前端复用

长期项目里，更推荐把 benchmark analysis 当成结构化分析资产，而不是只存在于 HTML 报告中。

```python
benchmark_returns = (
    benchmark_df.set_index("date")["close"].pct_change().fillna(0.0)
)

payload = result.benchmark_analysis(
    benchmark=benchmark_returns,
    curve_freq="D",
)

print(payload["summary"])
print(payload["series"][:2])
```

推荐做法：

1. 回测结束后由后端调用 `result.benchmark_analysis(...)`
2. 将 `summary + series + meta` 直接返回给前端
3. `result.viz.report(..., benchmark=...)` 与前端页面复用同一套 benchmark analysis 逻辑

这样可以避免前端再次计算超额收益、IR、Beta、Alpha，减少口径漂移。

如需把 benchmark analysis 作为回测产物保存：

```python
result.export_benchmark_analysis(
    path="artifacts/benchmark_analysis.json",
    benchmark=benchmark_returns,
    format="json",
    curve_freq="D",
)
```

### 13.3.4 信用账户强平审计视图

在融资/融券回测中，如果发生维持担保比例触发的强平，`BacktestResult` 会产出结构化审计表：

```python
liq_audit = result.liquidation_audit_df
print(liq_audit.head())
```

使用内置报告：

```python
result.viz.report(filename="report_margin.html", show=False)
```

报告会自动包含：

1. 强平审计明细表（日期、当日计息、强平标的、强平顺序）
2. 风险图表区中的按日强平统计图（有数据时展示）

## 13.4 第三方工具集成：QuantStats

`AKQuant` 完美支持 `QuantStats`，这是一个强大的 Python 量化分析库，能生成媲美专业基金的 Tearsheet。

### 13.4.1 安装与使用

```bash
pip install quantstats
```

### 13.4.2 生成综合报告

```python
result.viz.quantstats(
    benchmark="000300.SH",
    filename="qs_stats.html",
    title="Alpha Strategy QuantStats Report",
)
```

## 13.5 完整示例代码

下面的代码演示了如何运行策略，并使用 `AKQuant` 报告接口生成包含基准对比的可视化报告。

```python
--8<-- "examples/textbook/ch13_visualization.py"
```

## 13.6 专业 K 线图绘制 (Professional Candlestick Charts)

虽然折线图能展示大致趋势，但量化交易员更习惯看 K 线图 (Candlestick)。Python 中最专业的库是 `mplfinance`。

### 13.6.1 基础 K 线与成交量

```python
import mplfinance as mpf

# 准备数据 (必须包含 Open, High, Low, Close, Volume 列)
df.index.name = 'Date'

# 绘制蜡烛图 + 成交量
mpf.plot(df, type='candle', volume=True, style='yahoo')
```

### 13.6.2 叠加买卖点信号

在 K 线图上标注买入 (Buy) 和卖出 (Sell) 信号，直观复盘交易逻辑。

```python
# 生成买卖点标记
buys = [price if sig == 1 else np.nan for price, sig in zip(df['low'], df['buy_signal'])]
sells = [price if sig == -1 else np.nan for price, sig in zip(df['high'], df['sell_signal'])]

# 添加到副图
apds = [
    mpf.make_addplot(buys, type='scatter', markersize=100, marker='^', color='r'),
    mpf.make_addplot(sells, type='scatter', markersize=100, marker='v', color='g')
]

mpf.plot(df, addplot=apds)
```

## 13.7 3D 波动率曲面 (3D Volatility Surface)

对于期权交易员，仅仅看 IV 曲线是不够的，我们需要看到整个曲面（Strike x Expiry）。

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# X: 行权价, Y: 到期时间, Z: 隐含波动率
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('Strike Price')
ax.set_ylabel('Time to Expiry')
ax.set_zlabel('Implied Volatility')
```
**观察要点**在于两个方向上的形态：沿 Strike 轴的弯曲程度反映了 **Smile/Skew**，而沿 Time 轴的倾斜程度则反映了波动率的期限结构 (**Term Structure**)。

## 13.8 交易分析图表 (Trade Analysis)

### 13.8.1 MAE/MFE 散点图

最大不利偏离 (MAE) 与最大有利偏离 (MFE) 的散点图是优化止盈止损的神器，它以 **X 轴**表示 MAE（最大浮亏）、**Y 轴**表示最终盈亏 (PnL)。从这张图可以读出入场与离场的质量：如果大量盈利交易的 MAE 都很小（如 < 1%），说明入场点非常精准；反之，如果亏损交易的 MAE 很大，则说明止损设置过宽，或者执行拖沓。它背后的**黄金法则**是截断亏损 (Cut Loss)、让利润奔跑 (Let Profit Run)——这就意味着在图上，左下角的点（止损单）应该密集且受控，右上角的点（盈利单）应该发散且无上限。

## 本章小结

### 必须掌握

- 可视化不是装饰页面，而是发现风险、漂移和过拟合迹象的诊断工具。
- 收益、回撤、分布与交易行为需要联动解读，而不是单图下结论。

### 理解即可

- 同一策略在不同坐标、频率和基准下会呈现不同的解释视角。

### 实践提醒

- 统一图表口径、时间轴与色彩规范，避免图形与指标互相打架。

## 主线推进

贯穿全书的那条最小多均线 / 趋势策略，在前几章先后跑通了回测闭环、重写为标准策略类、扩展到多资产组合，并在上一阶段补齐了评价与优化口径。到本章，它进入“可视化诊断”的环节：我们不再只用一串汇总数字去判断它好坏，而是把它的回测产物摊开成一组相互印证的图表——用权益曲线（线性与对数）看复利轨迹，用水下图看回撤的深度与修复期，用收益分布看左尾与偏度，用月度热力图和滚动夏普看它在不同年份、不同市场状态下的稳定性，再借基准对比模块算出它相对沪深 300 的超额收益、信息比率与 Beta/Alpha。这样一来，主线策略从“能算出一个夏普”推进到了“能被一组图表交叉诊断”，为下一步把它因子化、并最终接入实盘前的复盘检查打好了观察基础。

## 延伸阅读

**经典著作**

- Tufte, E. R. *The Visual Display of Quantitative Information*, Graphics Press, 1st ed. 1983 / 2nd ed. 2001 —— 统计图形设计的奠基之作，提出“数据墨水比 (data-ink ratio)”与小型多重图等原则，为本章 13.1（可视化原则与核心图表）提供方法论基础。
- Bailey, D. H., & López de Prado, M. "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality," *The Journal of Portfolio Management*, 40(5), 2014, 94–107 —— 指出报表中的夏普比率会被多重检验与非正态分布系统性高估，呼应本章 13.2（滚动指标）与“识别过拟合迹象”的诊断目标。
- Bacon, C. R. *Practical Portfolio Performance Measurement and Attribution*, John Wiley & Sons, 1st ed. 2004 / 3rd ed. 2021 —— 系统讲解绩效度量与归因的实务计算，对应本章 13.3（基准对比：超额收益、跟踪误差、信息比率、Beta/Alpha）。

**官方文档与工具**

- [QuantStats（Ran Aroussi）](https://github.com/ranaroussi/quantstats) —— 本章 13.4 集成的第三方组合分析库，提供 stats / plots / reports 三大模块与 Tearsheet 报告。
- [mplfinance](https://github.com/matplotlib/mplfinance) —— 本章 13.6 用于绘制专业 K 线与成交量、叠加买卖点信号的 Matplotlib 金融图表库。
- [AKQuant 可视化指南](../guide/visualization.md) —— `result.viz.dashboard`、`result.viz.report`、`benchmark_analysis` 等内置绘图与报告接口的权威说明，对应本章 13.3。

**本书相关**

- [第 10 章：策略评价体系与风险指标](10_analysis.md) —— 本章 13.1、13.2 中权益曲线、回撤、滚动夏普等图表，是第 10 章指标分析的可视化呈现。
- [第 9 章：基金投资与资产配置理论](09_funds.md) —— 本章 13.3 的基准对比与超额收益分析，可直接用于复盘第 9 章的股债轮动与组合配置结果。

## 课后练习

### 基础题

1. 为同一策略分别绘制线性与对数权益曲线并比较解读差异。

### 应用题

1. 增加一个收益分布图，分析左尾风险变化。

### 综合题

1. 为交易记录生成 MAE/MFE 图并据此调整止损阈值。

??? note "参考答案要点（先独立思考再展开）"

    **基础题**：对数坐标下直线斜率代表复利增长率，能清晰展示早期波动；线性坐标在后期指数增长时会掩盖早期细节。

    **应用题**：关注收益分布左尾的肥厚程度与偏度；负偏（小赚大亏）形态需警惕黑天鹅风险。

    **综合题**：若盈利交易的 MAE 普遍很小，可适当收紧止损；若亏损交易 MAE 很大，说明止损过宽或执行拖沓。

## 常见错误与排查

1. 图表结论偏差：确认时间轴与收益频率是否一致。
2. 指标与图形不一致：核对绘图数据源和统计口径是否统一。
3. 图表过于拥挤：拆分子图并统一色彩与尺度规范。
