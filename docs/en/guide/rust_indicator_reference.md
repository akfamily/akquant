# AKQuant Indicator Reference (All Supported Indicators)

This page is currently maintained in Chinese first.

- Chinese page: [AKQuant 指标全量说明（104 个）](../../zh/guide/rust_indicator_reference.md)
- Related docs:
  - [Indicator Playbook](talib_indicator_playbook.md)
  - [Indicator Scenario Quickref](indicator_scenario_quickref.md)
  - [Chapter 16: AKQuant Indicators in Practice](../textbook/16_rust_indicators.md)

## Warmup and Bad Rows

- Streaming indicator classes (such as `aq.EMA`) return `None` from `update` and report `is_ready == False` during warmup; they no longer return fabricated values. A row with NaN/inf input also returns `None` and does not advance the internal state.
- Numerical conventions follow TA-Lib 0.8.2 and, for functions that exist in TA-Lib, both backends agree value by value; see "Numerical Conventions" in the [Indicator Playbook](talib_indicator_playbook.md) for intentional deviations.
