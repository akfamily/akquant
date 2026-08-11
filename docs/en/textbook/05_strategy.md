# Chapter 5: Strategy Development in Practice

This chapter is currently maintained in Chinese first.

- Chinese chapter: [第 5 章：策略开发实战 (Strategy Implementation)](../../zh/textbook/05_strategy.md)
- Textbook home: [Chinese textbook index](../../zh/textbook/index.md)
- Practice links:
  - Primary example: [examples/textbook/ch05_strategy.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch05_strategy.py)
  - Extended example: [examples/23_functional_callbacks_demo.py](https://github.com/akfamily/akquant/blob/main/examples/23_functional_callbacks_demo.py)
  - Framework hooks example: [examples/50_framework_hooks_demo.py](https://github.com/akfamily/akquant/blob/main/examples/50_framework_hooks_demo.py)
- Day-boundary rebalance example: [examples/strategies/05_stock_momentum_rotation_timer.py](https://github.com/akfamily/akquant/blob/main/examples/strategies/05_stock_momentum_rotation_timer.py)
- After-bar rebalance example: [examples/strategies/09_stock_momentum_rotation_after_bar.py](https://github.com/akfamily/akquant/blob/main/examples/strategies/09_stock_momentum_rotation_after_bar.py)
  - Class-style tick callback example: [examples/51_class_tick_callbacks_demo.py](https://github.com/akfamily/akquant/blob/main/examples/51_class_tick_callbacks_demo.py)
  - Pre-open execution example: [examples/52_pre_open_demo.py](https://github.com/akfamily/akquant/blob/main/examples/52_pre_open_demo.py)
  - Staged next-day pre-open example: [examples/53_timer_to_pre_open_demo.py](https://github.com/akfamily/akquant/blob/main/examples/53_timer_to_pre_open_demo.py)
  - Expiry callback example: [examples/49_on_expiry_demo.py](https://github.com/akfamily/akquant/blob/main/examples/49_on_expiry_demo.py)
  - Indicator playbook example: [examples/45_talib_indicator_playbook_demo.py](https://github.com/akfamily/akquant/blob/main/examples/45_talib_indicator_playbook_demo.py)
  - Real-data mode: `python examples/45_talib_indicator_playbook_demo.py --data-source akshare --symbol sh600000 --start-date 20240101 --end-date 20260301`
  - Functional twin example: [examples/textbook/ch05_strategy_functional.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch05_strategy_functional.py)
  - Guide: [Strategy Guide](../guide/strategy.md)
  - Guide: [Custom Indicator Guide](../guide/custom_indicator.md)

## TA-Lib Backend Usage

- `backend="auto"` defaults to `rust`.
- For baseline alignment with legacy strategies, explicitly use `backend="python"`.
- To override `auto` globally, set `AKQUANT_TALIB_AUTO_BACKEND=python|rust`.

```python
from akquant import talib as ta

close = df["close"].to_numpy()
high = df["high"].to_numpy()
low = df["low"].to_numpy()
volume = df["volume"].to_numpy()

rsi_py = ta.RSI(close, timeperiod=14, backend="python")
rsi_rs = ta.RSI(close, timeperiod=14, backend="rust")
mfi_rs = ta.MFI(high, low, close, volume, timeperiod=14, backend="rust")
tema_rs = ta.TEMA(close, timeperiod=20, backend="rust")
```

## Indicator Selection Template

- Trend-following: `EMA + ADX + NATR`
- Mean-reversion: `BBANDS + RSI`
- Volume confirmation: `OBV + MFI + ROC`
- Trailing risk control: `SAR + ATR`

```python
import numpy as np
from akquant import talib as ta

ema_fast = ta.EMA(close, timeperiod=20, backend="rust")
ema_slow = ta.EMA(close, timeperiod=60, backend="rust")
adx = ta.ADX(high, low, close, timeperiod=14, backend="rust")

if np.isnan(ema_fast[-1]) or np.isnan(adx[-1]):
    return
if ema_fast[-1] > ema_slow[-1] and adx[-1] >= 20:
    self.buy(symbol, 100)
```

For complete Chinese content including the full `on_xxx` callback map, warmup handling, pre-open execution semantics, staged next-day execution patterns, and migration tips:
- [第 5 章：策略开发实战 (Strategy Implementation)](../../zh/textbook/05_strategy.md)
- [指标组合实战手册](../../zh/guide/talib_indicator_playbook.md)
