# 按策略场景选指标速查表（四象限）

本页给教学与实战一个“先选场景、再选指标”的快速入口。
核心目标：减少选指标时间，优先保证策略结构完整（主信号 + 过滤器 + 风控）。

## 一图理解：四象限

| 象限 | 核心任务 | 优先指标组 | 典型输出 |
| :--- | :--- | :--- | :--- |
| 趋势 | 找方向、确认趋势持续性 | `EMA/SMA/TEMA/KAMA/SAR` + `ADX/ADXR` | 单序列为主 |
| 震荡 | 找超买超卖与回归机会 | `BBANDS/RSI/CMO/WILLR/STOCH` | 单序列 + 多输出 |
| 风控 | 控制波动风险与止损宽度 | `ATR/NATR/TRANGE/STDDEV/VAR` | 单序列 |
| 量价确认 | 减少假突破与信号噪声 | `OBV/MFI/AD/ADOSC/BOP` | 单序列 |

## 1) 趋势象限

### 适用市场

- 单边上涨或单边下跌阶段
- 中高时间框架趋势延续行情

### 推荐组合

- **基础**：`EMA(20/60) + ADX(14)`
- **增强**：`EMA + SAR + NATR`
- **平滑替代**：`KAMA/T3` 替代固定参数均线

### 最小决策骨架

```python
trend_up = ema_fast[-1] > ema_slow[-1]
trend_strong = adx[-1] >= 20
enter_long = trend_up and trend_strong
```

## 2) 震荡象限

### 适用市场

- 横盘箱体或波段往复行情
- 强趋势衰减后的回归段

### 推荐组合

- **基础**：`BBANDS + RSI`
- **增强**：`BBANDS + RSI + MOM`
- **K 线位置型**：`STOCH/WILLR` 辅助入场确认

### 最小决策骨架

```python
oversold = close[-1] < lower[-1] and rsi[-1] < 30
momentum_ok = mom[-1] > 0
enter_long = oversold and momentum_ok
```

## 3) 风控象限

### 适用市场

- 所有场景都应启用
- 尤其适用于高波动、事件驱动阶段

### 推荐组合

- **波动宽度**：`ATR/NATR`
- **动态止损**：`SAR + ATR`
- **波动过滤**：`NATR` 阈值过滤开仓

### 最小决策骨架

```python
volatility_ok = natr[-1] < 4.5
trailing_stop = sar[-1]
risk_width = 2.0 * atr[-1]
```

## 4) 量价确认象限

### 适用市场

- 突破策略、追涨策略、趋势确认策略
- 需要判断“价格动作是否有成交量支持”

### 推荐组合

- **趋势确认**：`OBV + AD`
- **资金动量**：`MFI + ROC`
- **短线过滤**：`ADOSC/BOP` 作为次级过滤器

### 最小决策骨架

```python
price_breakout = close[-1] > resistance
volume_confirm = obv[-1] > obv[-2] and mfi[-1] > 50
enter_long = price_breakout and volume_confirm
```

## 场景到指标的快速映射

| 你要解决的问题 | 先看哪些指标 |
| :--- | :--- |
| 不知道当前是否有趋势 | `EMA + ADX` |
| 横盘中反复止损 | `BBANDS + RSI`，并增加 `ADX` 过滤 |
| 信号太多、胜率低 | 增加 `NATR` 与 `OBV/MFI` 过滤 |
| 止损不稳定、回撤大 | `SAR + ATR` 做动态止损 |
| 迁移后结果与老策略差异大 | 先 `backend="python"` 对齐，再切高性能后端 |

## 教学建议（45 分钟课堂）

1. 10 分钟：讲四象限框架与“主信号/过滤器/风控”三层结构。
2. 15 分钟：现场跑 `EMA+ADX+NATR` 与 `BBANDS+RSI+MOM` 两套骨架。
3. 10 分钟：讲 warmup 与多输出解包（`MACD/BBANDS/STOCH`）。
4. 10 分钟：做一次“加量价确认后胜率变化”的对比实验。

## 相关资料

- [AKQuant 指标全量说明（104 个）](./rust_indicator_reference.md)
- [指标组合实战手册](./talib_indicator_playbook.md)
- [第 16 章：AKQuant 指标全景与工程化使用](../textbook/16_rust_indicators.md)
