# 指标组合实战手册

本手册提供 `akquant.talib` 在策略实战中的组合方式、调参顺序与排错要点，面向“从 TA-Lib 迁移到 AKQuant”以及“新策略快速起步”两类场景。

## 0. 运行示例

```bash
python examples/45_talib_indicator_playbook_demo.py
python examples/45_talib_indicator_playbook_demo.py --data-source akshare --symbol sh600000 --start-date 20240101 --end-date 20260301
```

## 1. 快速起步

```python
from akquant import talib as ta

close = df["close"].to_numpy()
high = df["high"].to_numpy()
low = df["low"].to_numpy()
volume = df["volume"].to_numpy()

ema_fast = ta.EMA(close, timeperiod=20, backend="rust")
ema_slow = ta.EMA(close, timeperiod=60, backend="rust")
adx = ta.ADX(high, low, close, timeperiod=14, backend="rust")
natr = ta.NATR(high, low, close, timeperiod=14, backend="rust")
```

## 2. 组合模板

| 目标 | 推荐组合 | 触发逻辑骨架 |
| :--- | :--- | :--- |
| 趋势跟随 | `EMA + ADX + NATR` | 趋势方向 + 趋势强度 + 波动过滤 |
| 均值回归 | `BBANDS + RSI + MOM` | 偏离带宽 + 超买超卖 + 短动量确认 |
| 量价确认 | `OBV + MFI + ROC` | 价格方向先行，成交量信号放行 |
| 跟踪止损 | `SAR + ATR` | 趋势持有 + 动态止损宽度 |

```python
upper, middle, lower = ta.BBANDS(close, timeperiod=20, backend="rust")
rsi = ta.RSI(close, timeperiod=14, backend="rust")
mom = ta.MOM(close, period=10, backend="rust")

long_signal = close[-1] < lower[-1] and rsi[-1] < 30 and mom[-1] > 0
exit_signal = close[-1] > middle[-1]
```

## 3. 调参顺序

建议固定以下顺序，避免参数空间爆炸：

1. 先定交易风格：趋势或均值回归。
2. 再定主信号：`EMA/BBANDS/SAR` 之一。
3. 再定过滤器：`ADX/RSI/MOM` 之一。
4. 最后定风险参数：`ATR/NATR` 阈值与止损系数。

## 4. warmup 与输出形态

- 大多数指标在 warmup 区段会返回 `NaN`，应先做空值过滤再下单。
- 多输出指标保持 tuple 结构：
  - `MACD -> (macd, signal, hist)`
  - `BBANDS -> (upper, middle, lower)`
  - `STOCH -> (slowk, slowd)`

```python
macd, signal, hist = ta.MACD(close, backend="rust")
if macd.size == 0:
    return
if not (macd[-1] == macd[-1] and signal[-1] == signal[-1]):
    return
```

## 5. 常见误区

- 只看单指标，不做趋势强度过滤，导致震荡期反复交易。
- 忽略 warmup 区段，直接使用尾值触发信号。
- 将参数一次性全局搜索，导致样本内过拟合。
- 迁移时直接切换 `rust`，未先与 `python` 基线对齐。

## 6. 迁移清单

1. 保持 TA-Lib 函数签名不变，先跑 `backend="python"`。
2. 验证输出结构与 warmup 一致后，再切到 `backend="rust"`。
3. 对支持 `period` 别名的指标优先沿用旧参数名。
4. 用固定样本建立最小回归测试。

## 7. 数值口径

- **基准**：以 TA-Lib 0.8.2（内置 C 库 0.8.1）为准。预热长度（首个有效值之前的 `NaN` 个数）、种子取法、递推公式、除零时的取值与默认参数都与 TA-Lib 一致；TA-Lib 里有的函数，`backend="python"` 与 `backend="rust"` 逐值一致（相对误差在 1e-9 以内，`MAMA`、`HT_TRENDLINE` 为 1e-8）。
- **非有限值**：任一输入为 NaN 或 ±inf 时，这一行视为不存在。流式类（如 `aq.EMA(14).update(...)`）返回 `None` 且不改变内部状态；`akquant.talib` 函数该行输出 `NaN`，计算时跳过该行。开头的 NaN 与 TA-Lib 完全一致，因此 `ta.SMA(ta.RSI(close))` 这类串联可以直接使用。
- **流式类的预热**：预热期 `update` 返回 `None`、`is_ready` 为 `False`（例如 `aq.MACD(12, 26, 9)` 前 33 根），不再返回编造值。`Strategy.I()` 的预热推断同样按 TA-Lib lookback 计算（`MACD(12, 26, 9)` 为 34 根）。
- **有意偏离 TA-Lib**：下表各项是刻意与 TA-Lib 不同，不是缺陷。

| 项 | TA-Lib | akquant | 理由 |
|---|---|---|---|
| 序列中段的 NaN | 之后全部 NaN | 跳过该行继续算 | 停牌、缺数据后指标要能恢复 |
| MAXINDEX、MININDEX、MINMAXINDEX 的预热期 | 填 0 | 填 NaN | 0 是合法下标，填 0 会被当成真实结果 |
| NATR(timeperiod=1) | 直接返回 TRANGE，没有归一化 | 返回 TR / close × 100 | TA-Lib 的 bug |
| 滑动累加的舍入残差 | 0.8.1 用“平移 + 定期重算”的滑动累加，在接近退化（但不完全退化）的窗口上按各自的相对阈值判零 | 每个窗口精确重算（先减去窗口首值），判零规则照抄 0.8.1 | 完全退化的窗口（停牌、一字板）结果与 TA-Lib 相同；只在“几乎平盘”的人造数据上可能差在 1e-9 以内 |
| MACD、APO、PPO 的 slowperiod ≤ fastperiod | 自动互换 | `akquant.talib` 抛 ValueError（保持现状）；Rust 类构造时互换 | 不改现有的参数校验 |
| MA、BBANDS、STOCH 的 matype | 支持 0～8 | 只支持 0（SMA）；APO、PPO 固定为 EMA（等同 TA-Lib 默认的 matype=1） | 不在本轮范围 |
| LN、LOG10 的输入为 0 | -inf | NaN | 与现有数学函数对非法输入返回 NaN 的约定一致 |
| akquant 独有函数（COVAR、LINEARREG_R2、CLIP、CLAMP01、SQ 等数学扩展） | 无 | 只遵守非有限值规则；两个后端不保证逐值一致：COVAR 的 python 后端比 rust 后端晚 4 根开始输出，MOD 在操作数为负时两个后端的结果不同 | 没有 TA-Lib 基准 |

相关文档：
- [第 5 章：策略开发实战](../textbook/05_strategy.md)
- [可运行示例：45_talib_indicator_playbook_demo.py](https://github.com/akfamily/akquant/blob/main/examples/45_talib_indicator_playbook_demo.py)
