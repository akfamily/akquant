# Textbook: Quantitative Investment from Theory to Practice

*This textbook is currently available in Chinese only. English translation is planned for future releases.*

Please refer to the [Chinese Version](../../zh/textbook/index.md) for the full content.

## Before You Read

To avoid confusion when you see local log timestamps alongside UTC fields such as `timestamp_iso`, read these two pages first:

- Intro explanation: [AKQuant Time and Timezones](../guide/quant_basics.md)
- Advanced FAQ: [Timezone Handling Guide](../advanced/timezone.md)

Core rule:

> **AKQuant stores factual event time in UTC and renders local time for humans.**

## English Chapter Skeleton

You can still browse the full chapter structure in English and jump to the Chinese source chapter from each page:

- [Chapter 1: Quantitative Investment Overview and Environment Setup](01_foundations.md)
- [Chapter 2: Programming Survival Guide](02_programming.md)
- [Chapter 3: Financial Data Acquisition and Processing](03_data.md)
- [Chapter 4: Event-Driven Backtesting Principles](04_backtest_engine.md)
- [Chapter 5: Strategy Development in Practice](05_strategy.md)
- [Chapter 6: A-Share Market Microstructure and Strategy Practice](06_stock_a.md)
- [Chapter 7: Futures Market and Derivatives Strategies](07_futures.md)
- [Chapter 8: Options Pricing and Volatility Strategies](08_options.md)
- [Chapter 9: Fund Investment and Asset Allocation Theory](09_funds.md)
- [Chapter 10: Strategy Evaluation Framework and Risk Metrics](10_analysis.md)
- [Chapter 11: Parameter Optimization and Robustness Validation](11_optimization.md)
- [Chapter 12: Machine Learning in Quantitative Investing](12_ml.md)
- [Chapter 13: Strategy Visualization and Report Analysis](13_visualization.md)
- [Chapter 14: High-Performance Factor Mining and Expression Engine](14_factor.md)
- [Chapter 15: Live Trading Systems and Operations](15_live_trading.md)
- [Chapter 16: AKQuant Indicator System and Practical Usage](16_rust_indicators.md)

Recent updates in the Chinese textbook:

* Chapter 11 includes the inline parameter field (`IntParam`/`FloatParam`/...) + `param_grid` workflow for UI-driven parameter optimization.
* Chapter 15 has been aligned with current live-trading APIs, including warm-start and strategy-loader paths.
* Chapter 5 now includes the full `on_xxx` callback map, framework hooks, class-style Tick callback study path, `on_pre_open` guidance, and a staged next-day execution example.
* Textbook chapter structure and example chapter labels are now synchronized.
