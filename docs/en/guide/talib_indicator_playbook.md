# TA-Lib Indicator Playbook

This page is currently maintained in Chinese first.

- Chinese page: [指标组合实战手册](../../zh/guide/talib_indicator_playbook.md)
- Related docs:
  - [Chapter 5: Strategy Development](../textbook/05_strategy.md)
  - [Runnable example: 45_talib_indicator_playbook_demo.py](https://github.com/akfamily/akquant/blob/main/examples/45_talib_indicator_playbook_demo.py)

## Quick Run

```bash
python examples/45_talib_indicator_playbook_demo.py
python examples/45_talib_indicator_playbook_demo.py --data-source akshare --symbol sh600000 --start-date 20240101 --end-date 20260301
```

## Numerical Conventions

- **Baseline**: TA-Lib 0.8.2 (bundled C library 0.8.1). Lookback (number of leading `NaN`), seeding, recurrences, division-by-zero values and default parameters all match TA-Lib. For functions that exist in TA-Lib, the `python` and `rust` backends agree value by value (relative error within 1e-9; 1e-8 for `MAMA` and `HT_TRENDLINE`).
- **Non-finite inputs**: a row where any input is NaN or +/-inf is treated as absent. Streaming classes (for example `aq.EMA(14).update(...)`) return `None` and leave their state untouched; `akquant.talib` functions output `NaN` for that row and skip it in the computation. Leading NaN behaves exactly like TA-Lib, so chains such as `ta.SMA(ta.RSI(close))` work.
- **Streaming warmup**: during warmup `update` returns `None` and `is_ready` is `False` (for example the first 33 bars of `aq.MACD(12, 26, 9)`); no fabricated values are returned. `Strategy.I()` warmup inference also follows the TA-Lib lookback (34 bars for `MACD(12, 26, 9)`).
- **Intentional deviations from TA-Lib**:

| Item | TA-Lib | akquant | Reason |
|---|---|---|---|
| NaN in the middle of a series | all later outputs are NaN | the row is skipped and computation continues | indicators must recover after suspensions or missing data |
| Warmup of MAXINDEX, MININDEX, MINMAXINDEX | filled with 0 | filled with NaN | 0 is a valid index and would be read as a real result |
| NATR(timeperiod=1) | returns TRANGE without normalization | returns TR / close x 100 | a TA-Lib bug |
| Rounding residue of rolling sums | 0.8.1 uses shifted rolling sums with periodic recomputation and decides zero by its own relative thresholds on nearly (but not fully) degenerate windows | every window is recomputed exactly (first value subtracted), zero rules copied from 0.8.1 | fully degenerate windows (suspension, limit-locked bars) match TA-Lib; only on artificial near-flat data may differ within 1e-9 |
| MACD, APO, PPO with slowperiod <= fastperiod | swapped automatically | `akquant.talib` raises ValueError (unchanged); the Rust classes swap at construction | existing argument validation is kept |
| matype of MA, BBANDS, STOCH | 0 to 8 | only 0 (SMA); APO and PPO are fixed to EMA (TA-Lib default matype=1) | out of scope |
| LN, LOG10 of 0 | -inf | NaN | consistent with the other math functions returning NaN on invalid input |
| akquant-only functions (COVAR, LINEARREG_R2, CLIP, CLAMP01, SQ and other math extensions) | none | only the non-finite rule applies; the two backends are not guaranteed to agree: the python backend of COVAR starts 4 bars later than rust, and MOD differs between backends for negative operands | no TA-Lib baseline |
