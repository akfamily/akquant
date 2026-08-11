# 第 16 章：AKQuant 技术指标体系与应用

> ⏱️ 预计阅读 ~20 分钟 ｜ 🎯 难度 ★★★☆☆（进阶）

在前面的章节中，我们已经学习了数据、回测、策略、分析与实盘主线。本章把视角重新拉回策略内部最常用的一类工具：**技术指标 (Indicators)**。与“知道某个指标名字”相比，更重要的是学会把指标当作**可迁移、可验证、可工程化复用**的组件来使用。

## 学习目标

- 理解 AKQuant Rust 指标体系的使用边界，以及 `python -> rust` 迁移的推荐路径。
- 掌握指标输入、输出、warmup 与多输出解包的统一阅读方法。
- 能将趋势、动量、波动率、量价指标组合成最小可复现实验。

## 前置知识

- 已掌握第 5 章中的策略开发基础，知道指标会被放进什么样的交易逻辑中。
- 已掌握第 10 章中的结果分析方法，能够解释指标变化如何影响策略表现。

## 本章实践入口

- 主示例：[examples/textbook/ch16_indicators.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch16_indicators.py)
- 进阶示例：[examples/45_talib_indicator_playbook_demo.py](https://github.com/akfamily/akquant/blob/main/examples/45_talib_indicator_playbook_demo.py), [examples/60_custom_indicator_demo.py](https://github.com/akfamily/akquant/blob/main/examples/60_custom_indicator_demo.py), [examples/62_indicator_streaming_demo.py](https://github.com/akfamily/akquant/blob/main/examples/62_indicator_streaming_demo.py), [examples/61_indicator_visualization_export_demo.py](https://github.com/akfamily/akquant/blob/main/examples/61_indicator_visualization_export_demo.py)
- 函数式孪生示例：[examples/textbook/ch16_indicators_functional.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch16_indicators_functional.py)
- 对应指南：[AKQuant 指标全量说明](../guide/rust_indicator_reference.md)

## 快速运行与验收

```bash
python examples/textbook/ch16_indicators.py
python examples/textbook/ch16_indicators_functional.py
# 可选：接入真实 A 股数据
python examples/textbook/ch16_indicators.py --symbol sh600000
```

验收要点：

1. 脚本依次输出三课演示：读指标（输入/输出/warmup）、用指标（EMA+ADX+NATR 角色分工回测）、迁移指标（python/rust 后端对齐）。
2. 切换 `backend="python"` 与 `backend="rust"` 后，收敛尾段数值基本一致且可解释。
3. 能识别 warmup 导致的空值区段，并在计算指标前剔除前部 NaN 填充，避免把无效值喂给策略信号。

## 16.1 为什么单独学习 Rust 指标体系

AKQuant 的指标接口并不只是“换一个更快的 TA-Lib 后端”。它更像一个统一层：

1. **接口统一**：同一套函数签名可切换 `backend="python"` 与 `backend="rust"`。
2. **迁移友好**：你可以先让旧策略在 Python 后端上对齐，再切换到 Rust 后端提速。
3. **工程化复用**：指标可以作为主信号、过滤器、风控尺度，甚至可视化与特征工程输入。

因此，本章关注的不是“把 103 个指标背下来”，而是学会三件真正重要的事：

- 指标**吃什么输入**；
- 指标**吐什么输出**；
- 指标在前若干根 K 线上的 **warmup 行为**是什么。

## 16.2 全量覆盖范围（当前 103 个）

- Momentum：8 个
- Moving Average & Transforms：59 个
- Trend：20 个
- Volatility：11 个
- Volume：5 个

完整逐项解释、参数口径与返回说明请始终以参考页为准：

- [Rust 指标全量说明（103 个）](../guide/rust_indicator_reference.md)

这意味着本章的定位不是“把字典搬进正文”，而是帮你建立**读指标、选指标、迁移指标**的方法。

## 16.3 学会“读指标”的统一方法

无论你看到的是 `EMA`、`MACD` 还是 `ATR`，都建议按同一模板理解：

1. **输入是什么**：是 `close`，还是 `high, low, close`，还是需要 `volume`。
2. **输出是什么**：返回单序列、双序列还是三序列。
3. **warmup 多长**：前多少根 K 线不能直接用于交易判断。
4. **策略角色是什么**：主信号、过滤器、风控尺度，还是确认信号。

### 16.3.1 输入结构

最常见的输入结构有三类：

- **单输入**：例如 `EMA(close)`、`RSI(close)`。
- **三输入**：例如 `ATR(high, low, close)`、`ADX(high, low, close)`。
- **四输入**：少数量价指标还会显式使用 `volume`。

如果输入结构看错，后面的结果解释几乎一定会偏掉。

### 16.3.2 输出结构

指标的输出并不总是“一列数字”。

- **单输出**：`EMA`、`RSI`、`ATR`
- **双输出**：`AROON`
- **三输出**：`MACD`、`BollingerBands`、`STOCH`

因此，多输出指标不能只记名字，还要记**解包顺序**。

### 16.3.3 warmup 不是小细节

指标在窗口未满时，结果往往是空值或不稳定值。对策略开发而言，这不是边缘问题，而是最常见的误用来源之一。

如果你在 warmup 阶段就直接发出交易信号，往往会出现两类假象：

1. 回测“看起来能跑”，但信号并不可靠；
2. Python 与 Rust 后端的对齐误差被误判成实现错误。

## 16.4 五大类指标怎么教、怎么用

### 16.4.1 趋势类：先判断方向，再过滤强度

- 方向判别：`EMA` / `SMA` / `TEMA` / `KAMA` / `SAR`
- 强度过滤：`ADX` / `ADXR` / `DX`
- 典型组合：`EMA + ADX + NATR`

教学上可以先让学生完成一件最小任务：**均线给方向，ADX 过滤弱趋势，NATR 控制仓位尺度**。这样一来，趋势、强度、风险三个角色都被引入了。

### 16.4.2 动量类：捕捉变化速度与过热过冷

- 速度/斜率：`ROC` / `ROCP` / `ROCR` / `ROCR100` / `MOM`
- 过热过冷：`RSI` / `CMO` / `WILLR`
- 典型组合：`BBANDS + RSI + MOM`

动量类指标最容易出现的问题是“只看信号方向，不看市场状态”。把它与波动率或趋势过滤器联用，通常比单独使用更稳。

### 16.4.3 波动率类：先定风险尺度，再谈仓位

- 范围与波动：`ATR` / `NATR` / `TRANGE` / `STDDEV` / `VAR`
- 通道类：`BollingerBands`
- 价格派生：`MEDPRICE` / `TYPPRICE` / `WCLPRICE` / `AVGPRICE` / `MIDPRICE`

波动率类指标特别适合做两类任务：

1. 给止损、止盈、移动保护定义动态阈值；
2. 把“行情变快了”这件事转化为可量化的仓位调整规则。

### 16.4.4 量价类：确认而不是替代主信号

- 趋势确认：`OBV` / `AD`
- 动量确认：`MFI` / `ADOSC`
- K 线力量：`BOP`

量价类指标常见的正确用法是“做确认层”，而不是直接替代价格主信号。这样更容易解释，也更适合教学。

### 16.4.5 数学变换类：为特征工程做准备

- 对数/指数：`LN` / `LOG10` / `LOG1P` / `EXP` / `EXPM1`
- 三角与双曲：`SIN` / `COS` / `TAN` / `ASIN` / `ACOS` / `ATAN` / `SINH` / `COSH` / `TANH`
- 代数运算：`ADD` / `SUB` / `MULT` / `DIV` / `MOD` / `POW` / `MAX2` / `MIN2`
- 规整变换：`ABS` / `SIGN` / `ROUND` / `CLIP` / `CLAMP01` / `SQ` / `CUBE` / `RECIP` / `INV_SQRT` / `DEG2RAD`

这一类指标在传统技术分析里不一定显眼，但在第 12 章和第 14 章那种“把信号送入模型或表达式引擎”的任务里很有价值。

## 16.5 三个最常见的工程坑位

### 16.5.1 warmup 区段误用

- 窗口不足时的结果不能直接拿来发单。
- 教学示例和作业里应明确要求先做空值过滤。

### 16.5.2 多输出解包顺序错误

- `MACD -> (macd, signal, hist)`
- `BollingerBands -> (upper, middle, lower)`
- `STOCH -> (slowk, slowd)`
- `AROON -> (aroondown, aroonup)`

如果顺序拿错，结果通常不会报错，但策略语义会悄悄失真，这比直接报错更危险。

### 16.5.3 迁移时直接切 Rust

推荐流程不是“一上来就换 Rust”，而是：

1. 先用 `backend="python"` 与旧策略结果对齐；
2. 再切到 `backend="rust"` 观察数值与绩效差异；
3. 最后再做性能与大规模批量实验。

## 16.6 标准教学脚手架

下面这段脚手架适合直接放进实验课，用于演示“同一套输入上并行计算多类指标”的最小流程：

```python
import numpy as np
from akquant import talib as ta

close = np.asarray(df["close"], dtype=float)
high = np.asarray(df["high"], dtype=float)
low = np.asarray(df["low"], dtype=float)
volume = np.asarray(df["volume"], dtype=float)

ema_fast = np.asarray(ta.EMA(close, timeperiod=20, backend="rust"), dtype=float)
ema_slow = np.asarray(ta.EMA(close, timeperiod=60, backend="rust"), dtype=float)
adx = np.asarray(ta.ADX(high, low, close, timeperiod=14, backend="rust"), dtype=float)
natr = np.asarray(ta.NATR(high, low, close, timeperiod=14, backend="rust"), dtype=float)
rsi = np.asarray(ta.RSI(close, timeperiod=14, backend="rust"), dtype=float)

# warmup 区段不可直接参与信号判断
if np.isnan([ema_fast[-1], ema_slow[-1], adx[-1], natr[-1], rsi[-1]]).any():
    return
```

你可以把这段模板理解为一个最小实验台：

- `EMA` 负责方向；
- `ADX` 负责强度；
- `NATR` 负责风险尺度；
- `RSI` 负责状态确认。

## 16.7 三类推荐实验

### 16.7.1 趋势实验：`EMA + ADX + NATR`

目标：构建一套最小趋势框架。

- `EMA(20)` 与 `EMA(60)` 判断方向；
- `ADX(14)` 过滤掉趋势强度过弱的区间；
- `NATR(14)` 用于决定仓位大小或止损距离。

这一实验最适合作为“指标不是堆砌，而是角色分工”的第一课。

### 16.7.2 震荡实验：`BBANDS + RSI`

目标：理解区间震荡与过热过冷。

- `BollingerBands` 给出上下轨；
- `RSI` 确认是否进入超买/超卖区域；
- 观察单独使用与联合使用时的信号差异。

### 16.7.3 迁移实验：`python -> rust`

目标：验证迁移流程，而不是直接追求更快。

建议步骤：

1. 先固定参数与数据集；
2. 用 `backend="python"` 记录一份基线结果；
3. 切到 `backend="rust"` 对比数值、信号点位与最终绩效；
4. 仅在结果口径一致后再讨论性能收益。

## 16.8 推荐教学路径

1. 第 1 周：先教 `EMA` / `RSI` / `ATR` 的输入、输出与 warmup。
2. 第 2 周：加入 `MACD` / `BBANDS` / `STOCH` 的多输出处理。
3. 第 3 周：引入 `ADX` / `NATR` / `SAR` 做风险过滤。
4. 第 4 周：做一次 `python -> rust` 迁移实验与回归验证。
5. 第 5 周：把数学变换类指标接入简单特征工程实验。

## 本章小结

### 必须掌握

- Rust 指标体系已经覆盖策略开发中最常用的核心技术面能力。
- 真正决定教学效果的不是指标数量，而是输入、输出、warmup 与角色分工的结构化理解。

### 理解即可

- 数学变换类指标、流式指标处理与自定义指标开发，是从技术分析过渡到特征工程的重要桥梁。

### 实践提醒

- 做指标迁移时先对齐结果，再切后端提速。
- 完整指标字典、参数说明与返回结构请始终以参考页为准：[Rust 指标全量说明（103 个）](../guide/rust_indicator_reference.md)。

## 主线推进

贯穿全书的那条最小多均线 / 趋势策略，在本章被反过来拆解成它最底层的零件——技术指标，并被纳入一套可迁移、可验证、可工程化复用的使用规范。前面各章里，均线只是“一根线穿过价格”的直觉；本章则要求把它当作有明确输入、输出与 warmup 行为的组件来对待：`EMA` 负责给方向，`ADX` 负责过滤弱趋势，`NATR` 负责把仓位与止损放回波动率尺度上，三者各司其职，正是“指标不是堆砌，而是角色分工”的最小示范。更关键的是，本章为这条主线补上了从 Python 后端到 Rust 后端的迁移纪律：先用 `backend="python"` 与既有结果对齐，再切到 `backend="rust"` 比对数值、信号点位与绩效，最后才谈性能与批量提速。至此，主线策略不再依赖某一处临时写就的指标计算，而是站在一个统一、可复现、可加速的指标工程层之上——这也为全书从研究走向准实盘、再走向规模化实验，留下了可靠的复用基座。

## 延伸阅读

**经典著作**

- Kaufman, P. J. *Trading Systems and Methods*（第 6 版），John Wiley & Sons, 2019（ISBN 9781119605355）—— 交易系统与技术指标的权威工具书，系统覆盖移动平均、ATR、布林带、MACD、动量振荡器与趋势系统的数学口径与风险用法，可作为本章 16.4「五大类指标」逐项含义与组合方式的纸面参照。
- Murphy, J. J. *Technical Analysis of the Financial Markets*，New York Institute of Finance, 1999 —— 技术分析的标准教科书，对趋势、动量、波动率与量价四类指标的市场含义有完整阐述，呼应本章 16.4.1–16.4.4 关于「指标承担什么角色」的讨论。
- Pardo, R. *The Evaluation and Optimization of Trading Strategies*（第 2 版），John Wiley & Sons, 2008 —— 强调先固定数据与参数、再做基线对齐与回归验证的严谨流程，与本章 16.5.3、16.7.3 提出的「先对齐结果、再切后端提速」迁移纪律一脉相承。

**官方文档与工具**

- [AKQuant 指标全量说明（103 个）](../guide/rust_indicator_reference.md) —— 本章正文反复强调的权威参考页，给出全部指标的输入、输出、参数口径与返回结构，对应本章 16.2 的覆盖范围与 16.3 的「读指标」方法。
- [TA-Lib Functions List](https://ta-lib.org/functions/) —— TA-Lib 官方函数清单，逐一列出 ADX、MACD、BBANDS、ATR、OBV 等全部指标及其英文全称，可对照本章 16.4 的五大类划分与 16.4.5 的数学变换类。
- [TA-Lib Python 文档](https://ta-lib.github.io/ta-lib-python/) —— TA-Lib 的 Python 封装文档，说明函数返回数组、lookback（warmup）区段会以 NaN 填充等行为，呼应本章 16.3.3 与 16.5.1 对 warmup 空值的处理要求。

**本书相关**

- [第 5 章：策略开发实战](05_strategy.md) —— 本章 16.4 各指标最终要放进第 5 章那样的事件驱动策略类里承担主信号、过滤器或风控尺度的角色。
- [第 10 章：策略评价体系与风险指标](10_analysis.md) —— 本章 16.7.3 的 `python -> rust` 迁移实验，最终要靠第 10 章的数值、信号点位与绩效对比来判定口径是否一致。
- [第 14 章：高性能因子挖掘与表达式引擎](14_factor.md) —— 本章 16.4.5 的数学变换类指标，正是把信号送入第 14 章那种因子表达式与模型流程时的常用预处理算子。

## 课后练习

### 基础题

1. 任选 `EMA`、`RSI`、`ATR` 三个指标，写出它们各自的输入、输出和 warmup 口径。

### 应用题

1. 用 `EMA + ADX + NATR` 设计一个最小趋势策略过滤框架，并解释每个指标承担的角色。

### 综合题

1. 对同一数据集分别运行 `backend="python"` 与 `backend="rust"`，比较数值、信号点位与回测结果差异，并写出迁移结论。

??? note "参考答案要点（先独立思考再展开）"

    **基础题**：`EMA(close)` 单输入单输出，warmup ≈ `timeperiod-1`；`RSI(close)` 单输入单输出，warmup ≈ `timeperiod`；`ATR(high, low, close)` 三输入单输出，warmup ≈ `timeperiod`。

    **应用题**：EMA 给方向、ADX 过滤弱趋势、NATR 定风险尺度（决定仓位大小或止损宽度）——三者角色分工，而非指标堆砌。

    **综合题**：EMA 这类简单递推指标全程基本一致；RSI/ADX/NATR 这类 Wilder 平滑指标在 warmup 区段有初值差异、收敛尾段趋于一致。结论：先用 python 后端对齐基线，再切 rust 提速。

## 常见错误与排查

1. 指标结果全是空值：优先检查窗口长度、输入数组长度和 warmup 区段是否被误当成有效值。
2. 多输出指标结果看起来“能跑但不对”：检查解包顺序是否与文档一致。
3. 切换到 Rust 后结果变化很大：先回到 Python 后端做基线对齐，再排查数据口径、空值处理和参数设置。

---

**全书结语**：
恭喜你完成了《量化投资：从理论到实战》的全部课程！
我们从 Python/Rust 基础出发，构建了高性能回测引擎，探讨了股票、期货、期权等全资产类别的策略，落地到实盘交易系统，并在本章把视角收束回策略内部最常用的指标工程。
量化投资是一场没有终点的马拉松。市场在变，对手在变，唯一不变的是我们要保持**对数据的敬畏**和**对逻辑的执着**。

**愿 Alpha 与你同在！**
