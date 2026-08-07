# 量化投资：从理论到实战——基于 AKQuant 框架

## 教材简介

本教材专为中国高校本科生及研究生（金融工程、计算机、统计学背景）设计，旨在填补理论与工程实践之间的鸿沟。

### 核心特色

*   **现代技术栈**：深入解析 Rust + Python 混合架构，掌握高性能量化系统的设计原理。
*   **中国本土化**：专注于 A 股（T+1、涨跌停场景建模）、国内期货（CTP接口）与期权市场。
*   **实战导向**：从数据清洗、策略回测到实盘交易的全链路覆盖，配套完整代码示例。

## 贯穿式课程项目：从第一个信号到准实盘闭环

为了避免“每章都学了，但很难串成一条主线”，建议把全书当作一个逐章升级的课程项目来学习。你可以围绕一套最小多均线/趋势过滤策略，按下面路径逐步扩展：

1. **第 1 到 3 章**：完成环境搭建、数据获取与最小清洗，跑通第一个可复现回测。
2. **第 4 到 5 章**：把策略改造成事件驱动版本，补上生命周期、下单接口、日志与风控。
3. **第 6 到 9 章**：把同一主线分别放到 A 股、期货、期权与基金配置场景中，理解市场制度差异。
4. **第 10 到 14 章**：为主线策略补上评价、优化、机器学习、可视化与因子研究能力。
5. **第 15 到 16 章**：继续完善准实盘、监控与指标工程化使用，形成从研究到运行的完整闭环。

如果你是第一次系统学习量化，强烈建议每读完一章，就把当前主线策略和实验记录保存为一个新版本，而不是只运行孤立示例。

## 学前提醒：先建立时间与时区心智模型

为了避免在阅读日志、回测结果与 `timestamp_iso` 时产生困惑，建议先阅读以下两篇说明：

*   入门解释：[`AKQuant` 的时间与时区](../guide/quant_basics.md)
*   进阶 FAQ： [时区处理指南](../advanced/timezone.md)

核心原则只有一句话：

> **AKQuant 用 UTC 保存事实时间，用本地时区展示给人看。**

## 目录大纲

> 📖 随查工具：[术语表与符号约定](00_glossary.md) ｜ [常见误区对照表](appendix_pitfalls.md) ｜ [附录 A：环境与复现](appendix_setup.md) ｜ [附录 B：引用、许可与勘误](appendix_cite.md)
>
> 🎓 学完全书后，请完成 [课程项目：端到端量化实战 (Capstone)](capstone.md)。

### 第一部分：量化基础与数据准备 (Foundations)

*   **[第 1 章：量化投资概述与环境搭建](01_foundations.md)**
    *   量化投资发展史与 Alpha/Beta 理论
    *   AKQuant 架构简介 (Rust Core + Python Wrapper)
    *   环境配置与 Hello World ([examples/textbook/ch01_quickstart.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch01_quickstart.py))
*   **[第 2 章：量化编程基础](02_programming.md)**
    *   Python for Quant: Pandas, NumPy, Matplotlib
    *   Rust 概念入门：类型系统与内存安全
    *   案例：数据处理实战 ([examples/textbook/ch02_programming.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch02_programming.py))
*   **[第 3 章：金融数据获取与处理](03_data.md)**
    *   时间序列分析基础
    *   AKShare 数据接口详解
    *   数据清洗与本地存储 ([examples/textbook/ch03_data.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch03_data.py))

### 第二部分：回测引擎架构 (The Engine)

*   **[第 4 章：事件驱动回测原理](04_backtest_engine.md)**
    *   向量化 vs 事件驱动 ([examples/textbook/ch04_comparison.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch04_comparison.py))
    *   核心组件解析：Engine, Strategy, DataFeed
*   **[第 5 章：策略开发实战](05_strategy.md)**
    *   策略生命周期、完整 `on_xxx` 回调地图与下单接口
    *   历史数据获取与防未来函数
    *   案例：双均线策略实现 ([examples/textbook/ch05_strategy.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch05_strategy.py))
    *   进阶：框架级钩子、Tick 回调、盘前开盘语义与“双阶段次日执行”模式（[examples/50_framework_hooks_demo.py](https://github.com/akfamily/akquant/blob/main/examples/50_framework_hooks_demo.py), [examples/51_class_tick_callbacks_demo.py](https://github.com/akfamily/akquant/blob/main/examples/51_class_tick_callbacks_demo.py), [examples/52_pre_open_demo.py](https://github.com/akfamily/akquant/blob/main/examples/52_pre_open_demo.py), [examples/53_timer_to_pre_open_demo.py](https://github.com/akfamily/akquant/blob/main/examples/53_timer_to_pre_open_demo.py)）

### 第三部分：多资产策略开发 (Strategies)

*   **[第 6 章：A 股市场微观结构与策略实战](06_stock_a.md)**
    *   T+1 交易制度的工程实现
    *   涨跌停场景建模与滑点模拟
    *   案例：A 股交易规则演示 ([examples/textbook/ch06_stock_a.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch06_stock_a.py))
*   **[第 7 章：期货市场与衍生品策略](07_futures.md)**
    *   保证金与杠杆
    *   期权基础与 Greeks 风控
    *   案例：期货动量策略 ([examples/textbook/ch07_futures.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch07_futures.py))
*   **[第 8 章：期权定价与波动率策略](08_options.md)**
    *   核心要素与 Greeks
    *   案例：备兑看涨 (Covered Call) ([examples/textbook/ch08_options.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch08_options.py))
*   **[第 9 章：基金投资与资产配置理论](09_funds.md)**
    *   ETF/LOF 交易规则与免税优势
    *   可转债 T+0 与双低轮动
    *   现代投资组合理论 (MPT) 与 60/40 策略
    *   案例：ETF 网格交易 ([examples/textbook/ch09_funds.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch09_funds.py))
    *   案例：股债平衡策略 ([examples/textbook/ch09_portfolio.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch09_portfolio.py))

### 第四部分：评价、优化与高阶话题 (Advanced)

*   **[第 10 章：策略评价体系与风险指标](10_analysis.md)**
    *   夏普比率、最大回撤与归因分析
    *   案例：回测结果深入分析 ([examples/textbook/ch10_analysis.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch10_analysis.py))
*   **[第 11 章：参数优化与稳健性检验](11_optimization.md)**
    *   网格搜索与滚动回测 (WFO)
    *   案例：多进程参数优化 ([examples/textbook/ch11_optimization.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch11_optimization.py))
*   **[第 12 章：机器学习在量化中的应用](12_ml.md)**
    *   特征工程与模型预测
    *   基于 Scikit-learn 的择时策略 ([examples/textbook/ch12_ml.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch12_ml.py))
*   **[第 13 章：策略可视化与报表分析](13_visualization.md)**
    *   权益曲线与回撤图绘制
    *   案例：生成回测图表 ([examples/textbook/ch13_visualization.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch13_visualization.py))
*   **[第 14 章：高性能因子挖掘与表达式引擎](14_factor.md)**
    *   因子表达式的原理与优势
    *   Polars 高性能计算架构
    *   案例：Alpha101 因子实战 ([examples/textbook/ch14_factor.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch14_factor.py))
### 第五部分：从回测到实盘 (Live Trading)

*   **[第 15 章：实盘交易系统与运维](15_live_trading.md)**
    *   实盘与回测的差异处理
    *   主示例：实盘启动与网关接入 ([examples/textbook/ch15_live_trading.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch15_live_trading.py))
    *   进阶示例：动态策略加载与运行时注入 ([examples/textbook/ch15_strategy_loader.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch15_strategy_loader.py))
    *   外部信号接入：HTTP webhook 驱动下单 ([examples/61_signal_platform_webhook.py](https://github.com/akfamily/akquant/blob/main/examples/61_signal_platform_webhook.py))
    *   日志/审计示例：订单审计落盘、敏感脱敏、日志语言 ([examples/66_logging_audit_demo.py](https://github.com/akfamily/akquant/blob/main/examples/66_logging_audit_demo.py))
    *   风控与熔断机制

### 第六部分：指标工程与工具链 (Indicator Engineering)

*   **[第 16 章：AKQuant 技术指标体系与应用](16_rust_indicators.md)**
    *   使用输入/输出/warmup 模板理解 AKQuant 的 103 个指标
    *   通过 `python -> rust` 迁移实验掌握指标工程化使用路径
    *   案例：指标三课演示（读/用/迁移）([examples/textbook/ch16_indicators.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch16_indicators.py)）
    *   指标词典与实验脚手架联动（见 [AKQuant 指标全量说明](../guide/rust_indicator_reference.md)）

### 课程项目与附录

*   **[课程项目：端到端量化实战 (Capstone)](capstone.md)**：把全书主线策略串成数据→策略→回测→评价→优化→可视化→实盘准备的完整闭环，附交付物清单。
*   **[术语表与符号约定](00_glossary.md)**：全书数学符号与核心术语随查表，标注首次讲解章节。
*   **[附录 A：环境与复现](appendix_setup.md)**：推荐环境、依赖版本基线与"如何复现书中回测数字"。
*   **[附录 B：引用、许可与勘误](appendix_cite.md)**：如何引用本教材、MIT 许可说明、Issue 勘误入口与免责声明。
*   **[附录 C：常见误区对照表](appendix_pitfalls.md)**：全书反复出现的陷阱汇成 ❌→✅ 速查表，动手前对照自查。

## 章节示例映射（主示例 / 进阶示例 / 对应指南）

| 章节 | 主示例 | 进阶示例 | 对应指南 |
| :--- | :--- | :--- | :--- |
| 第 1 章 | [ch01_quickstart.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch01_quickstart.py) | [01_quickstart.py](https://github.com/akfamily/akquant/blob/main/examples/01_quickstart.py) | [快速开始](../start/quickstart.md) |
| 第 2 章 | [ch02_programming.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch02_programming.py) | [17_readme_demo.py](https://github.com/akfamily/akquant/blob/main/examples/17_readme_demo.py) | [Python 基础](../guide/python_basics.md) |
| 第 3 章 | [ch03_data.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch03_data.py) | [37_feed_replay_alignment_demo.py](https://github.com/akfamily/akquant/blob/main/examples/37_feed_replay_alignment_demo.py) | [数据指南](../guide/data.md) |
| 第 4 章 | [ch04_comparison.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch04_comparison.py) | [25_streaming_backtest_demo.py](https://github.com/akfamily/akquant/blob/main/examples/25_streaming_backtest_demo.py), [68_backtest_tick_demo.py](https://github.com/akfamily/akquant/blob/main/examples/68_backtest_tick_demo.py) | [数据指南](../guide/data.md) |
| 第 5 章 | [ch05_strategy.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch05_strategy.py) | [23_functional_callbacks_demo.py](https://github.com/akfamily/akquant/blob/main/examples/23_functional_callbacks_demo.py), [50_framework_hooks_demo.py](https://github.com/akfamily/akquant/blob/main/examples/50_framework_hooks_demo.py), [51_class_tick_callbacks_demo.py](https://github.com/akfamily/akquant/blob/main/examples/51_class_tick_callbacks_demo.py), [52_pre_open_demo.py](https://github.com/akfamily/akquant/blob/main/examples/52_pre_open_demo.py), [53_timer_to_pre_open_demo.py](https://github.com/akfamily/akquant/blob/main/examples/53_timer_to_pre_open_demo.py) | [策略指南](../guide/strategy.md) |
| 第 6 章 | [ch06_stock_a.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch06_stock_a.py) | [20_risk_management_demo.py](https://github.com/akfamily/akquant/blob/main/examples/20_risk_management_demo.py) | [量化基础](../guide/quant_basics.md) |
| 第 7 章 | [ch07_futures.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch07_futures.py) | [04_mixed_assets.py](https://github.com/akfamily/akquant/blob/main/examples/04_mixed_assets.py) | [策略指南](../guide/strategy.md) |
| 第 8 章 | [ch08_options.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch08_options.py) | [07_option_test.py](https://github.com/akfamily/akquant/blob/main/examples/07_option_test.py) | [量化基础](../guide/quant_basics.md) |
| 第 9 章 | [ch09_funds.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch09_funds.py) | [ch09_portfolio.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch09_portfolio.py) | [策略指南](../guide/strategy.md) |
| 第 10 章 | [ch10_analysis.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch10_analysis.py) | [33_report_and_analysis_outputs.py](https://github.com/akfamily/akquant/blob/main/examples/33_report_and_analysis_outputs.py) | [分析指南](../guide/analysis.md) |
| 第 11 章 | [ch11_optimization.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch11_optimization.py) | [02_parameter_optimization.py](https://github.com/akfamily/akquant/blob/main/examples/02_parameter_optimization.py) | [优化指南](../guide/optimization.md) |
| 第 12 章 | [ch12_ml.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch12_ml.py) | [10_ml_walk_forward.py](https://github.com/akfamily/akquant/blob/main/examples/10_ml_walk_forward.py), [55_functional_ml_walk_forward.py](https://github.com/akfamily/akquant/blob/main/examples/55_functional_ml_walk_forward.py) | [机器学习指南](../advanced/ml.md) |
| 第 13 章 | [ch13_visualization.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch13_visualization.py) | [11_plot_visualization.py](https://github.com/akfamily/akquant/blob/main/examples/11_plot_visualization.py) | [可视化指南](../guide/visualization.md) |
| 第 14 章 | [ch14_factor.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch14_factor.py) | [16_adj_returns_signal.py](https://github.com/akfamily/akquant/blob/main/examples/16_adj_returns_signal.py) | [因子指南](../guide/factor.md) |
| 第 15 章 | [ch15_live_trading.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch15_live_trading.py) | [ch15_strategy_loader.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch15_strategy_loader.py), [38_live_functional_strategy_demo.py](https://github.com/akfamily/akquant/blob/main/examples/38_live_functional_strategy_demo.py), [39_live_mixed_broker_demo.py](https://github.com/akfamily/akquant/blob/main/examples/39_live_mixed_broker_demo.py), [66_logging_audit_demo.py](https://github.com/akfamily/akquant/blob/main/examples/66_logging_audit_demo.py), [61_signal_platform_webhook.py](https://github.com/akfamily/akquant/blob/main/examples/61_signal_platform_webhook.py) | [实盘函数式指南](../advanced/live_functional_quickstart.md), [外部信号平台接入](../advanced/signal_ingestion.md) |
| 第 16 章 | [ch16_indicators.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch16_indicators.py) | [45_talib_indicator_playbook_demo.py](https://github.com/akfamily/akquant/blob/main/examples/45_talib_indicator_playbook_demo.py), [60_custom_indicator_demo.py](https://github.com/akfamily/akquant/blob/main/examples/60_custom_indicator_demo.py), [62_indicator_streaming_demo.py](https://github.com/akfamily/akquant/blob/main/examples/62_indicator_streaming_demo.py) | [AKQuant 指标全量说明](../guide/rust_indicator_reference.md) |

---

**配套代码**：请参考项目根目录下的 `examples/textbook/` 文件夹。
