# 第 14 章：高性能因子挖掘与表达式引擎

> ⏱️ 预计阅读 ~20 分钟 ｜ 🎯 难度 ★★★☆☆（进阶）

## 学习目标

- 掌握因子表达式的基本写法、分区语义与批量运行方式。
- 理解 TS、CS、EL 三类算子的角色边界与调试路径。
- 能够把表达式研究扩展到可解释、可复现的因子实验。

## 前置知识

- 已掌握第 3 章的数据处理与基础时间序列操作。
- 了解滚动窗口、排序与相关系数等常见统计概念。

## 本章实践入口

- 主示例：[examples/textbook/ch14_factor.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch14_factor.py)
- 进阶示例：[examples/16_adj_returns_signal.py](https://github.com/akfamily/akquant/blob/main/examples/16_adj_returns_signal.py)
- 对应指南：[因子指南](../guide/factor.md)

## 快速运行与验收

```bash
python examples/textbook/ch14_factor.py
```

验收要点：

1. 脚本可完成至少一个表达式因子的计算与输出。
2. 能对同一数据集运行多个因子并比较结果差异。
3. 拆分复杂表达式后，结果可逐步复现并便于排障。

## 14.1 本章你会得到什么

这一章聚焦一件事：让你能稳定写出**可解释、可调试、可批量运行**的因子表达式。

学完后你应该能做到：

1. 用 1 行表达式快速验证一个 Alpha 想法。
2. 区分 TS / CS / EL 三类算子并避免语义误用。
3. 对复杂表达式进行拆步调试，而不是“盲猜哪里错了”。
4. 用 `run_batch` 批量计算并理解它的性能取舍。

> 本章讲“怎么思考与实战”，算子清单与排障速查请看 [因子表达式引擎指南](../guide/factor.md)。

## 14.2 为什么表达式模式更适合因子研究

在 Alpha 研究中，研究员真正需要的是“低成本试错”。如果沿用传统方式，每个因子都要写一段 DataFrame 逻辑，重复处理分组、对齐、窗口和缺失值，试错成本会随候选因子数量线性累积；而表达式方式则把研究员的工作前移到“先写出数学结构，再交给引擎执行”，例如 `Rank(Ts_Mean(Close, 5))`。

这种写法的核心收益是解耦：研究员只需描述“算什么”，引擎则负责“怎么算”——也就是解析、执行计划与并行优化。正因为“描述意图”与“执行细节”被分到了两侧，因子想法才得以被快速验证，而不必每次都重写一遍底层的数据处理流程。

## 14.3 十分钟上手：从 0 到可运行

### 14.3.1 准备最小数据

```python
import akshare as ak
import pandas as pd
from akquant.data import ParquetDataCatalog

catalog = ParquetDataCatalog("./data_catalog")

symbols = ["sh600000", "sz300750"]
for symbol in symbols:
    df = ak.stock_zh_a_hist_sina(
        symbol=symbol,
        start_date="20230101",
        end_date="20230601",
        adjust="hfq",
    )
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    catalog.write(symbol, df)
```

### 14.3.2 跑通三个层次

```python
from akquant.factor import FactorEngine

engine = FactorEngine(catalog)

# 单层表达式：先确认基础计算正确
df_ts = engine.run("Ts_Mean(Close, 5)")

# 嵌套表达式：再确认跨分区语义
df_nested = engine.run("Rank(Ts_Mean(Close, 5))")

# 批量表达式：最后再进行规模化
df_batch = engine.run_batch(
    [
        "Ts_Mean(Close, 5)",
        "Rank(Volume)",
        "Rank(Ts_Corr(Close, Volume, 10))",
    ]
)
```

### 14.3.3 结果先看三列

先只看这三件事，再看绩效：

1. `date` 是否连续且顺序正确。
2. `symbol` 是否完整覆盖样本池。
3. `factor_value` 是否存在异常常数、全 NaN 或极端离群。

## 14.4 表达式写作三板斧

### 14.4.1 第一板斧：先单层，后嵌套

不要一开始就写五层嵌套，建议按顺序：

1. 先验证内层（例如 `Ts_Mean(Close, 5)`）。
2. 再包外层（例如 `Rank(...)`）。
3. 最后再加条件或组合项（例如 `If(...)`）。

### 14.4.2 第二板斧：按分区语义写

- **TS（时序）**：按 `symbol` 分组滚动。
- **CS（截面）**：按 `date` 分组横截面。
- **EL（元素级）**：逐元素变换，不引入分组窗口。

经验规则：

- 你在问“过去 d 根 K 线”时，优先 TS。
- 你在问“同一天谁强谁弱”时，优先 CS。
- 你在问“单点映射关系”时，优先 EL。

### 14.4.3 第三板斧：复杂式子拆成步骤

例如表达式：

```python
Rank(Ts_Corr(Close, Volume, 10))
```

建议先验证：

```python
Ts_Corr(Close, Volume, 10)
```

再验证外层 `Rank(...)`。拆步后，定位错误和性能问题都更快。

## 14.5 常用因子模板（可直接改参数）

### 14.5.1 趋势类

- 均线突破：

```python
If(Close > Ts_Mean(Close, 20), 1, -1)
```

- 新高强度：

```python
Close / Ts_Max(Close, 60)
```

### 14.5.2 反转类

- 短期反转：

```python
-1 * Rank(Delta(Close, 6))
```

- 乖离回归：

```python
-1 * (Close - Ts_Mean(Close, 20)) / Ts_Mean(Close, 20)
```

### 14.5.3 波动率与量价类

- 低波偏好：

```python
-1 * Ts_Std(Close, 20)
```

- 量价相关：

```python
-1 * Ts_Corr(Close, Volume, 10)
```

### 14.5.4 组合类

- 动量反转：

```python
Rank(Ts_Mean(Close, 5)) - Rank(Ts_Mean(Close, 20))
```

- 量价背离：

```python
If((Close == Ts_Max(Close, 20)) & (Volume < Ts_Mean(Volume, 20)), 1, 0)
```

## 14.6 调试与性能：最实用的工作流

### 14.6.1 排错顺序（建议固定）

1. 列名是否可映射（`Close`/`close` 可以，`ClosePrice` 需要真实存在）。
2. 窗口 `d` 是否大于可用历史长度。
3. 数据是否有大量 NaN 或停牌空洞。
4. 是否一次写了过深嵌套导致难以定位。

### 14.6.2 为什么嵌套表达式会慢

当出现 `CS(TS(...))` 或 `TS(CS(...))`，引擎会拆成多步并物化中间结果，换来正确语义与可调试性。

这不是额外负担，而是对“结果可解释”的必要成本。遇到慢查询时，先拆步验证再考虑并行/批量策略。

### 14.6.3 `run_batch` 的正确使用场景

`run_batch` 适合：

- 多个候选因子同批计算。
- 统一样本池、统一时间段对比。

`run_batch` 不适合：

- 单个复杂表达式的微观调试（先用 `run` 更清晰）。

## 14.7 数据质量与时区注意事项

1. 默认时区为 `Asia/Shanghai`。
2. 非 A 股场景需要显式设置 `timezone`。
3. 时间列必须显式 `tz_localize`，避免隐式时区偏移。
4. 停牌或缺失日期建议先做交易日对齐，再做滚动窗口计算。

> 时区细节请参考 [时区处理指南](../advanced/timezone.md)。

## 14.8 从原理到实践：引擎到底做了什么

当你调用：

```python
engine.run("Rank(Ts_Mean(Close, 5))")
```

内部过程可以理解为三步：

1. **Parser**：把字符串转为抽象语法结构。
2. **Planner**：识别 TS/CS/EL，必要时自动拆步。
3. **Executor**：按步骤执行并在关键节点物化中间结果。

这一机制直接决定了两个实践建议：

- 调试优先拆步。
- 优化优先减少不必要的跨分区嵌套。

## 本章小结

### 必须掌握

- 因子研究的关键不只是“算出来”，而是确保语义正确、结果可解释。
- 先单层、后嵌套、再批量，是最稳妥的表达式调试路径。

### 理解即可

- 高性能执行与表达式引擎只是手段，稳健检验与解释性仍是研究核心。

### 实践提醒

- 先检查日期、标的覆盖与异常值，再讨论因子优劣。

## 主线推进

贯穿全书的那条最小多均线 / 趋势策略，到本章被重新审视为一个“因子”而非仅仅一套买卖规则。此前各章把它当作完整的择时信号来打磨——第 1 章跑通回测闭环，第 4、5 章改造成事件驱动的标准策略类，第 9 章又把它放进多资产组合。本章则提供了一个更细的视角：均线突破 `If(Close > Ts_Mean(Close, 20), 1, -1)` 本身就可以写成一行表达式，主线策略的“信号”由此被拆解为可独立计算、可批量比较、可截面排序的因子单元。这样一来，那条最小策略不再是一个不可分的黑箱，而是能被 TS/CS/EL 算子重组、能与其他候选因子在统一样本池上 `run_batch` 对比的研究对象。至此，主线从“一个资产上的进出场判断”进一步推进到“可表达、可解释、可批量评估的因子化形态”，为后续用更系统的评价与优化方法筛选和增强它铺平了道路。

## 延伸阅读

**经典著作**

- Kakushadze, Z. "101 Formulaic Alphas," *Wilmott Magazine*, 2016(84), 2016, 72–80（亦见 arXiv:1601.00991） —— 公开给出 101 个可直接当作代码的公式化 Alpha，平均持有期约 0.6–6.4 天、两两平均相关性仅 15.9%，是本章 14.5 因子模板与表达式算子设计的直接源头。
- Bailey, D. H., & López de Prado, M. "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality," *The Journal of Portfolio Management*, 40(5), 2014, 94–107 —— 指出在 `run_batch` 式的多因子批量试验中，最优夏普会因多重检验而被系统性高估，为本章 14.6 的批量计算与因子筛选提供必要的统计纠偏视角。
- López de Prado, M. *Advances in Financial Machine Learning*，John Wiley & Sons, 2018 —— 系统讨论特征工程、回测过拟合与因子稳健检验，呼应本章小结强调的“稳健检验与解释性仍是研究核心”。

**官方文档与工具**

- [Polars 官方文档](https://docs.pola.rs/) —— 表达式（Expression）与惰性执行（Lazy API）是本章 14.2「表达式模式」与 14.8「Parser/Planner/Executor」机制的工程参照，其谓词下推、投影下推等查询优化思想正对应本章 14.6 关于嵌套表达式性能取舍的讨论。
- [AKShare 官方文档](https://akshare.akfamily.xyz/) —— 本章 14.3.1 准备最小数据所用行情接口（如 `stock_zh_a_hist_sina`）的权威来源。
- [AKQuant 因子表达式引擎指南](../guide/factor.md) —— 算子清单、分区语义与排障速查的完整参考，配合本章 14.4「写作三板斧」与 14.6「调试与性能」使用。

**本书相关**

- [第 3 章：金融数据获取与处理](03_data.md) —— 本章 14.3 的滚动窗口、排序与缺失值处理依赖第 3 章建立的数据基础。
- [第 16 章：AKQuant 技术指标体系与应用](16_rust_indicators.md) —— 本章 14.8 所述「引擎到底做了什么」与第 16 章的高性能执行思路一脉相承，可对照理解 Python 描述层与 Rust 执行层的分工。

## 课后练习

### 基础题

1. 为同一数据集分别运行一个 TS 因子和一个 CS 因子，比较输出差异。

### 应用题

1. 把一个三层嵌套表达式拆成多步运行并记录每步结果。

### 综合题

1. 设计一个包含因子计算、异常检查和结果解释的最小研究流程。

??? note "参考答案要点（先独立思考再展开）"

    **基础题**：TS 因子按 `symbol` 做时序滚动（如 `Ts_Mean`），CS 因子按 `date` 做横截面（如 `Rank`），二者的分区语义与输出含义不同。

    **应用题**：先验证最内层（如 `Ts_Corr(Close, Volume, 10)`）再逐层包外层（`Rank(...)`），分步物化中间结果，定位错误与性能瓶颈都更快。

    **综合题**：最小研究流程 = 计算因子 → 检查 date 连续性 / symbol 覆盖 / 异常值（全 NaN、常数、离群）→ 再评估绩效与可解释性。

## 常见错误与排查

1. 表达式报错：优先核对字段名、窗口长度与函数入参类型。
2. 结果异常噪声：检查停牌缺失、时区和交易日对齐是否正确。
3. 运行性能退化：拆分深层嵌套并减少不必要中间物化。
