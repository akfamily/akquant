# AKQuant 指标全量说明（104 个）

本页是 `akquant.talib` 的完整指标说明，面向“查指标含义 + 判断输入输出 + 了解 warmup 预期”三个核心需求。AKQuant 的底层高性能计算由 Rust 实现，对外保持统一易用的 Python 接口。

## 使用约定

- 所有函数均可直接在 `akquant.talib` 中使用，`backend="auto"` 默认走 `rust`。
- 需要做历史策略基线对齐时，建议临时显式设置 `backend="python"`。
- 如需全局覆盖 `auto` 行为，可设置环境变量 `AKQUANT_TALIB_AUTO_BACKEND=python|rust`。
- 单输出指标返回单序列；多输出指标返回 tuple。
- 绝大多数窗口类指标在 warmup 区段会返回 `NaN` 或空位，这属于正常行为。
- 流式指标类（如 `aq.EMA`）在预热期 `update` 返回 `None`、`is_ready` 为 `False`，不再返回编造值；输入含 NaN/±inf 的坏数据行同样返回 `None` 且不推进内部状态。
- 数值口径以 TA-Lib 0.8.2 为准，TA-Lib 里有的函数两个后端逐值一致，有意偏离项见[指标组合实战手册](talib_indicator_playbook.md)的“数值口径”一节。
- “warmup 参考”列给出默认口径下前多少个输出为空（`NaN`），与 TA-Lib 的 lookback 一致（akquant 独有的函数按实际行为标注）。
- 本页统计口径来自 AKQuant 当前指标注册表，当前总计 **104 个指标**。
- `HeikinAshi`（平均 K 线）只提供增量指标类 `aq.HeikinAshi()`，没有 `akquant.talib` 批量函数；配合 `self.I(..., outputs=("open","high","low","close"), render_type="candle")` 画成 K 线，见[自定义指标](./custom_indicator.md)。

## 全量索引总表（单表）

| 类别 | 指标 | 常见输入 | 输出 |
| :--- | :--- | :--- | :--- |
| Momentum | CMO | `close` | 单序列 |
| Momentum | MOM | `close` | 单序列 |
| Momentum | ROC | `close` | 单序列 |
| Momentum | ROCP | `close` | 单序列 |
| Momentum | ROCR | `close` | 单序列 |
| Momentum | ROCR100 | `close` | 单序列 |
| Momentum | RSI | `close` | 单序列 |
| Momentum | WILLR | `high,low,close` | 单序列 |
| Moving Average | ABS | `close` | 单序列 |
| Moving Average | ACOS | `close` | 单序列 |
| Moving Average | ADD | `x,y` | 单序列 |
| Moving Average | APO | `close` | 单序列 |
| Moving Average | ASIN | `close` | 单序列 |
| Moving Average | ATAN | `close` | 单序列 |
| Moving Average | AVGDEV | `close` | 单序列 |
| Moving Average | CEIL | `close` | 单序列 |
| Moving Average | CLAMP01 | `close` | 单序列 |
| Moving Average | CLIP | `value,min,max` | 单序列 |
| Moving Average | COS | `close` | 单序列 |
| Moving Average | COSH | `close` | 单序列 |
| Moving Average | CUBE | `close` | 单序列 |
| Moving Average | DEG2RAD | `close` | 单序列 |
| Moving Average | DEMA | `close` | 单序列 |
| Moving Average | DIV | `x,y` | 单序列 |
| Moving Average | EMA | `close` | 单序列 |
| Moving Average | EXP | `close` | 单序列 |
| Moving Average | EXPM1 | `close` | 单序列 |
| Moving Average | FLOOR | `close` | 单序列 |
| Moving Average | HT_TRENDLINE | `close` | 单序列 |
| Moving Average | INV_SQRT | `close` | 单序列 |
| Moving Average | KAMA | `close` | 单序列 |
| Moving Average | LN | `close` | 单序列 |
| Moving Average | LOG1P | `close` | 单序列 |
| Moving Average | LOG10 | `close` | 单序列 |
| Moving Average | MACD | `close` | 三序列 |
| Moving Average | MAMA | `close` | 双序列 |
| Moving Average | MAX | `close` | 单序列 |
| Moving Average | MAX2 | `x,y` | 单序列 |
| Moving Average | MAXINDEX | `close` | 单序列 |
| Moving Average | MIDPOINT | `close` | 单序列 |
| Moving Average | MIN | `close` | 单序列 |
| Moving Average | MIN2 | `x,y` | 单序列 |
| Moving Average | MININDEX | `close` | 单序列 |
| Moving Average | MINMAX | `close` | 双序列 |
| Moving Average | MINMAXINDEX | `close` | 双序列 |
| Moving Average | MOD | `x,y` | 单序列 |
| Moving Average | MULT | `x,y` | 单序列 |
| Moving Average | POW | `x,y` | 单序列 |
| Moving Average | PPO | `close` | 单序列 |
| Moving Average | RANGE | `close` | 单序列 |
| Moving Average | RECIP | `close` | 单序列 |
| Moving Average | ROUND | `close` | 单序列 |
| Moving Average | SIGN | `close` | 单序列 |
| Moving Average | SIN | `close` | 单序列 |
| Moving Average | SINH | `close` | 单序列 |
| Moving Average | SMA | `close` | 单序列 |
| Moving Average | SQ | `close` | 单序列 |
| Moving Average | SQRT | `close` | 单序列 |
| Moving Average | SUB | `x,y` | 单序列 |
| Moving Average | SUM | `close` | 单序列 |
| Moving Average | T3 | `close` | 单序列 |
| Moving Average | TAN | `close` | 单序列 |
| Moving Average | TANH | `close` | 单序列 |
| Moving Average | TEMA | `close` | 单序列 |
| Moving Average | TRIMA | `close` | 单序列 |
| Moving Average | TRIX | `close` | 单序列 |
| Moving Average | WMA | `close` | 单序列 |
| Trend | ADX | `high,low,close` | 单序列 |
| Trend | ADXR | `high,low,close` | 单序列 |
| Trend | AROON | `high,low` | 双序列 |
| Trend | AROONOSC | `high,low` | 单序列 |
| Trend | BETA | `x,y` | 单序列 |
| Trend | CCI | `high,low,close` | 单序列 |
| Trend | CORREL | `x,y` | 单序列 |
| Trend | COVAR | `x,y` | 单序列 |
| Trend | DX | `high,low,close` | 单序列 |
| Trend | LINEARREG | `close` | 单序列 |
| Trend | LINEARREG_ANGLE | `close` | 单序列 |
| Trend | LINEARREG_INTERCEPT | `close` | 单序列 |
| Trend | LINEARREG_R2 | `close` | 单序列 |
| Trend | LINEARREG_SLOPE | `close` | 单序列 |
| Trend | MINUS_DI | `high,low,close` | 单序列 |
| Trend | PLUS_DI | `high,low,close` | 单序列 |
| Trend | SAR | `high,low` | 单序列 |
| Trend | STOCH | `high,low,close` | 双序列 |
| Trend | TSF | `close` | 单序列 |
| Trend | ULTOSC | `high,low,close` | 单序列 |
| Volatility | ATR | `high,low,close` | 单序列 |
| Volatility | AVGPRICE | `open,high,low,close` | 单序列 |
| Volatility | BollingerBands | `close` | 三序列 |
| Volatility | HeikinAshi | `open,high,low,close` | 四序列 |
| Volatility | MEDPRICE | `high,low` | 单序列 |
| Volatility | MIDPRICE | `high,low` | 单序列 |
| Volatility | NATR | `high,low,close` | 单序列 |
| Volatility | STDDEV | `close` | 单序列 |
| Volatility | TRANGE | `high,low,close` | 单序列 |
| Volatility | TYPPRICE | `high,low,close` | 单序列 |
| Volatility | VAR | `close` | 单序列 |
| Volatility | WCLPRICE | `high,low,close` | 单序列 |
| Volume | AD | `high,low,close,volume` | 单序列 |
| Volume | ADOSC | `high,low,close,volume` | 单序列 |
| Volume | BOP | `open,high,low,close` | 单序列 |
| Volume | MFI | `high,low,close,volume` | 单序列 |
| Volume | OBV | `close,volume` | 单序列 |

## 中文补充说明（用途与推荐起始参数）

| 类别 | 典型用途 | 推荐起始参数（教学/实盘起步） |
| :--- | :--- | :--- |
| Momentum | 判断速度、强弱、超买超卖 | `RSI(14)`、`ROC(10)`、`MOM(10)`、`WILLR(14)` |
| Moving Average | 主趋势判断、平滑降噪、信号交叉 | `EMA(20/60)`、`SMA(20/60)`、`MACD(12,26,9)`、`KAMA(10)` |
| Trend | 趋势强度过滤、回归/角度分析、跟踪止损 | `ADX(14)`、`SAR(0.02,0.2)`、`STOCH(14,3,3)`、`LINEARREG(14)` |
| Volatility | 风险尺度、仓位控制、通道边界 | `ATR(14)`、`NATR(14)`、`STDDEV(20,1.0)`、`BBANDS(20,2.0)` |
| Volume | 量价确认、过滤假突破、资金动量验证 | `OBV`、`MFI(14)`、`AD`、`ADOSC(3,10)`、`BOP` |

说明：

- “推荐起始参数”是默认起步值，不是最优值；应在你的品种与周期上再做回测微调。
- 教学建议按“主信号 + 过滤器 + 风控”三层结构组合，而不是单指标独立使用。
- 若要做迁移对齐，先用 `backend="python"` 对齐历史策略，再切到高性能后端。

## 1) Momentum（8）

| 指标 | 解释 | 常见输入 | 输出 | warmup 参考 |
| :--- | :--- | :--- | :--- | :--- |
| RSI | 相对强弱指数，衡量涨跌强弱比 | `close` | 单序列 | `period` |
| ROC | 价格变化率（百分比） | `close` | 单序列 | `period` |
| ROCP | 价格变化率（比例） | `close` | 单序列 | `period` |
| ROCR | 价格比率（当前/过去） | `close` | 单序列 | `period` |
| ROCR100 | 价格比率×100 | `close` | 单序列 | `period` |
| MOM | 动量（当前减过去） | `close` | 单序列 | `period` |
| WILLR | 威廉指标，区间位置振荡器 | `high,low,close` | 单序列 | `period-1` |
| CMO | Chande 动量振荡器 | `close` | 单序列 | `period` |

## 2) Moving Average & Transforms（59）

### 2.1 平滑与趋势类

| 指标 | 解释 | 常见输入 | 输出 | warmup 参考 |
| :--- | :--- | :--- | :--- | :--- |
| SMA | 简单移动平均 | `close` | 单序列 | `period-1` |
| EMA | 指数移动平均 | `close` | 单序列 | `period-1` |
| WMA | 加权移动平均 | `close` | 单序列 | `period-1` |
| TRIMA | 三角移动平均 | `close` | 单序列 | `period-1` |
| DEMA | 双指数均线 | `close` | 单序列 | `2*(period-1)` |
| TEMA | 三指数均线 | `close` | 单序列 | `3*(period-1)` |
| TRIX | 三重 EMA 的变化率 | `close` | 单序列 | `3*(period-1)+1` |
| KAMA | 自适应均线 | `close` | 单序列 | `period` |
| T3 | Tillson T3 平滑 | `close` | 单序列 | `6*(period-1)` |
| HT_TRENDLINE | 近似 Hilbert 趋势线 | `close` | 单序列 | 63 |
| MAMA | 自适应双线 `(mama,fama)` | `close` | 双序列 | 32 |
| MACD | `(macd,signal,hist)` 三输出 | `close` | 三序列 | `slow+signal-2` |
| APO | 绝对价格振荡器（快慢均线差） | `close` | 单序列 | `slow-1` |
| PPO | 百分比价格振荡器 | `close` | 单序列 | `slow-1` |

### 2.2 窗口统计与区间类

| 指标 | 解释 | 常见输入 | 输出 | warmup 参考 |
| :--- | :--- | :--- | :--- | :--- |
| MIDPOINT | 窗口中点 `(max+min)/2` | `close` | 单序列 | `period-1` |
| MAX | 窗口最大值 | `close` | 单序列 | `period-1` |
| MIN | 窗口最小值 | `close` | 单序列 | `period-1` |
| MAXINDEX | 窗口最大值对应索引 | `close` | 单序列 | `period-1` |
| MININDEX | 窗口最小值对应索引 | `close` | 单序列 | `period-1` |
| MINMAX | 同时输出 `(min,max)` | `close` | 双序列 | `period-1` |
| MINMAXINDEX | 输出 `(minidx,maxidx)` | `close` | 双序列 | `period-1` |
| SUM | 窗口求和 | `close` | 单序列 | `period-1` |
| AVGDEV | 平均绝对偏差 | `close` | 单序列 | `period-1` |
| RANGE | 窗口振幅 `max-min` | `close` | 单序列 | `period-1`（akquant 独有） |

### 2.3 代数、三角、对数与变换类

| 指标 | 解释 | 常见输入 | 输出 |
| :--- | :--- | :--- | :--- |
| LN | 自然对数 | `close` | 单序列 |
| LOG10 | 常用对数 | `close` | 单序列 |
| SQRT | 平方根 | `close` | 单序列 |
| CEIL | 向上取整 | `close` | 单序列 |
| FLOOR | 向下取整 | `close` | 单序列 |
| SIN | 正弦 | `close` | 单序列 |
| COS | 余弦 | `close` | 单序列 |
| TAN | 正切 | `close` | 单序列 |
| ASIN | 反正弦 | `close` | 单序列 |
| ACOS | 反余弦 | `close` | 单序列 |
| ATAN | 反正切 | `close` | 单序列 |
| SINH | 双曲正弦 | `close` | 单序列 |
| COSH | 双曲余弦 | `close` | 单序列 |
| TANH | 双曲正切 | `close` | 单序列 |
| EXP | 指数 `e^x` | `close` | 单序列 |
| EXPM1 | `e^x-1` | `close` | 单序列 |
| LOG1P | `ln(1+x)` | `close` | 单序列 |
| DEG2RAD | 角度转弧度 | `close` | 单序列 |
| RECIP | 倒数 `1/x` | `close` | 单序列 |
| INV_SQRT | 逆平方根 `1/sqrt(x)` | `close` | 单序列 |
| ABS | 绝对值 | `close` | 单序列 |
| SIGN | 符号函数 | `close` | 单序列 |
| ROUND | 四舍五入 | `close` | 单序列 |
| SQ | 平方 | `close` | 单序列 |
| CUBE | 立方 | `close` | 单序列 |
| CLAMP01 | 截断到 `[0,1]` | `close` | 单序列 |
| ADD | 加法 | `x,y` | 单序列 |
| SUB | 减法 | `x,y` | 单序列 |
| MULT | 乘法 | `x,y` | 单序列 |
| DIV | 除法 | `x,y` | 单序列 |
| MOD | 取模 | `x,y` | 单序列 |
| POW | 幂运算 | `x,y` | 单序列 |
| MAX2 | 双输入取大 | `x,y` | 单序列 |
| MIN2 | 双输入取小 | `x,y` | 单序列 |
| CLIP | 按 `[min,max]` 截断 | `value,min,max` | 单序列 |

## 3) Trend（20）

| 指标 | 解释 | 常见输入 | 输出 | warmup 参考 |
| :--- | :--- | :--- | :--- | :--- |
| ADX | 趋势强度，不分方向 | `high,low,close` | 单序列 | `2*period-1` |
| ADXR | ADX 平滑再平滑 | `high,low,close` | 单序列 | `3*period-2` |
| DX | 方向动量差异度 | `high,low,close` | 单序列 | `period` |
| PLUS_DI | 正向方向指标 | `high,low,close` | 单序列 | `period` |
| MINUS_DI | 负向方向指标 | `high,low,close` | 单序列 | `period` |
| CCI | 典型价格偏离度 | `high,low,close` | 单序列 | `period-1` |
| AROON | `(_down,_up)` 趋势新高新低时距 | `high,low` | 双序列 | `period` |
| AROONOSC | AROON 振荡值 | `high,low` | 单序列 | `period` |
| STOCH | 随机指标 `(slowk,slowd)` | `high,low,close` | 双序列 | `fastk+slowk+slowd-3` |
| SAR | 抛物线转向止损 | `high,low` | 单序列 | 1 |
| ULTOSC | 终极振荡器 | `high,low,close` | 单序列 | `period3` |
| LINEARREG | 线性回归末端值 | `close` | 单序列 | `period-1` |
| LINEARREG_SLOPE | 回归斜率 | `close` | 单序列 | `period-1` |
| LINEARREG_INTERCEPT | 回归截距 | `close` | 单序列 | `period-1` |
| LINEARREG_ANGLE | 回归角度 | `close` | 单序列 | `period-1` |
| LINEARREG_R2 | 决定系数 `R²` | `close` | 单序列 | `period-1`（akquant 独有） |
| TSF | 时间序列预测值 | `close` | 单序列 | `period-1` |
| CORREL | 相关系数 | `x,y` | 单序列 | `period-1` |
| BETA | Beta（协方差/方差） | `x,y` | 单序列 | `period` |
| COVAR | 协方差 | `x,y` | 单序列 | `period-1`（akquant 独有；rust 后端口径，python 后端晚 4 根） |

## 4) Volatility（12）

| 指标 | 解释 | 常见输入 | 输出 | warmup 参考 |
| :--- | :--- | :--- | :--- | :--- |
| BollingerBands | 布林带 `(upper,middle,lower)` | `close` | 三序列 | `period-1` |
| ATR | 平均真实波幅 | `high,low,close` | 单序列 | `period` |
| NATR | 归一化 ATR（%） | `high,low,close` | 单序列 | `period` |
| TRANGE | 真实波幅（当期） | `high,low,close` | 单序列 | 1 |
| STDDEV | 标准差（可带倍数） | `close` | 单序列 | `period-1` |
| VAR | 方差（忽略 `nbdev`，与 TA-Lib 一致） | `close` | 单序列 | `period-1` |
| MEDPRICE | 中间价 `(high+low)/2` | `high,low` | 单序列 | 0 |
| TYPPRICE | 典型价 `(h+l+c)/3` | `high,low,close` | 单序列 | 0 |
| WCLPRICE | 加权收盘价 | `high,low,close` | 单序列 | 0 |
| AVGPRICE | 平均价 `(o+h+l+c)/4` | `open,high,low,close` | 单序列 | 0 |
| HeikinAshi | 平均 K 线 `(ha_open,ha_high,ha_low,ha_close)`；仅增量类 `aq.HeikinAshi()` | `open,high,low,close` | 四序列 | 0 |
| MIDPRICE | 区间中价 `(maxH+minL)/2` | `high,low` | 单序列 | `period-1` |

## 5) Volume（5）

| 指标 | 解释 | 常见输入 | 输出 | warmup 参考 |
| :--- | :--- | :--- | :--- | :--- |
| OBV | 能量潮，价格方向累计成交量 | `close,volume` | 单序列 | 0 |
| MFI | 资金流量指数 | `high,low,close,volume` | 单序列 | `period` |
| AD | 累积/派发线 | `high,low,close,volume` | 单序列 | 0 |
| ADOSC | AD 的快慢振荡 | `high,low,close,volume` | 单序列 | `max(fast,slow)-1` |
| BOP | 买卖力量平衡 | `open,high,low,close` | 单序列 | 0 |

## 常见问题

### 1. 什么时候先用 `backend="python"`？

- 当你在迁移旧策略、需要与原 TA-Lib 结果逐步对齐时。
- 对齐通过后，再切换高性能后端获取性能收益。

### 2. 为什么看到前几根是 `NaN`？

- 窗口类指标需要积累足够历史数据，这是指标定义的一部分，不是错误。

### 3. 多输出如何解包？

- `MACD -> (macd, signal, hist)`
- `BollingerBands -> (upper, middle, lower)`
- `STOCH -> (slowk, slowd)`
- `AROON -> (aroondown, aroonup)`
- `HeikinAshi.update(o, h, l, c) -> (ha_open, ha_high, ha_low, ha_close)`

## 相关资料

- [指标组合实战手册](./talib_indicator_playbook.md)
- [按策略场景选指标速查表（四象限）](./indicator_scenario_quickref.md)
- [第 16 章：AKQuant 指标全景与工程化使用](../textbook/16_rust_indicators.md)
