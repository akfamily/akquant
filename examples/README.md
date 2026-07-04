# Examples 索引

本文档用于快速浏览 `examples/` 目录中的可执行示例脚本。

## 最短执行路径

- 5 分钟入门: [01_quickstart.py](./01_quickstart.py) -> [17_readme_demo.py](./17_readme_demo.py)
- 页面化参数配置: [02_parameter_optimization.py](./02_parameter_optimization.py) -> `PARAM_MODEL` + `get_strategy_param_schema` + `validate_strategy_params`
- 报告与分析: [11_plot_visualization.py](./11_plot_visualization.py) -> [33_report_and_analysis_outputs.py](./33_report_and_analysis_outputs.py)
- 流式监控: [26_streaming_quickstart.py](./26_streaming_quickstart.py) -> [27_streaming_monitoring_console.py](./27_streaming_monitoring_console.py)
- 实时可视化: [31_streaming_live_console.py](./31_streaming_live_console.py) -> [32_streaming_live_web.py](./32_streaming_live_web.py)
- 高级目标仓位: [strategies/08_target_positions_long_short.py](./strategies/08_target_positions_long_short.py) -> `rebalance_positions` + `get_last_target_positions_plan`

## 基础与能力示例

- [01_quickstart.py](./01_quickstart.py): 多标的快速开始回测。
- [02_parameter_optimization.py](./02_parameter_optimization.py): 参数优化基础示例（含 `PARAM_MODEL`、schema 导出、参数校验演示）。
- [03_parameter_optimization_advanced.py](./03_parameter_optimization_advanced.py): 参数优化进阶示例。
- [04_mixed_assets.py](./04_mixed_assets.py): 混合资产回测示例。
- [05_live_trading_ctp.py](./05_live_trading_ctp.py): CTP 实盘接口示例。
- [06_complex_orders.py](./06_complex_orders.py): 复杂订单助手示例（`place_bracket` + 自动 OCO 联动）。
- [07_option_test.py](./07_option_test.py): 期权回测示例。
- [08_event_callbacks.py](./08_event_callbacks.py): 统一事件回调示例（`on_start/on_bar/on_order/on_trade/on_reject/on_timer/on_portfolio_update/on_stop`）。
- [09_ml_framework.py](./09_ml_framework.py): 机器学习框架基础示例。
- [10_ml_walk_forward.py](./10_ml_walk_forward.py): Walk-Forward 训练评估示例。
- [11_plot_visualization.py](./11_plot_visualization.py): 可视化报告生成示例。
- [12_wfo_integrated.py](./12_wfo_integrated.py): WFO 一体化示例。
- [13_quantstats_report.py](./13_quantstats_report.py): QuantStats 报告示例。
- [14_multi_frequency.py](./14_multi_frequency.py): 基于 DataFeedAdapter replay(session_windows) 的多频率回测示例。
- [15_plot_intraday.py](./15_plot_intraday.py): 日内绘图与回测示例。
- [16_adj_returns_signal.py](./16_adj_returns_signal.py): 复权收益信号示例。
- [17_readme_demo.py](./17_readme_demo.py): README 演示脚本。
- [18_benchmark_multisymbol.py](./18_benchmark_multisymbol.py): 多标的基准对比示例。
- [19_factor_expression.py](./19_factor_expression.py): 因子表达式示例。
- [20_risk_management_demo.py](./20_risk_management_demo.py): 风险管理示例。
- [21_warm_start_demo.py](./21_warm_start_demo.py): 热启动示例。
- [22_strategy_runtime_config_demo.py](./22_strategy_runtime_config_demo.py): 策略运行时配置示例。
- [23_functional_callbacks_demo.py](./23_functional_callbacks_demo.py): 函数式回调示例。
- [24_functional_tick_simulation_demo.py](./24_functional_tick_simulation_demo.py): 函数式 Tick 回调示例。
- [34_multi_strategy_demo.py](./34_multi_strategy_demo.py): 多策略 slot 与策略级风控对照示例。
- [35_custom_broker_registry_demo.py](./35_custom_broker_registry_demo.py): 自定义 Broker 注册与工厂创建示例。
- [36_trailing_orders.py](./36_trailing_orders.py): Trailing Stop/StopLimit 助手示例。
- [37_feed_replay_alignment_demo.py](./37_feed_replay_alignment_demo.py): replay 的 session/day/global 与 day_mode 对齐差异示例。
- [38_live_functional_strategy_demo.py](./38_live_functional_strategy_demo.py): LiveRunner 函数式策略入口示例。
- [39_live_broker_submit_order_demo.py](./39_live_broker_submit_order_demo.py): broker_live 下函数式 submit_order 最小闭环示例。
- [40_functional_multi_slot_risk_demo.py](./40_functional_multi_slot_risk_demo.py): 函数式 + 多策略 slot + 风控限制端到端示例。
- [41_live_multi_slot_orchestration_demo.py](./41_live_multi_slot_orchestration_demo.py): LiveRunner 多策略 slot 编排示例（paper）。
- [42_live_broker_event_audit_demo.py](./42_live_broker_event_audit_demo.py): broker 事件审计与 owner_strategy_id 追踪示例。
- [43_target_weights_rebalance.py](./43_target_weights_rebalance.py): TopN 动态权重调仓示例（横截面动量 + `rebalance_weights`）。
- [44_strategy_source_loader_demo.py](./44_strategy_source_loader_demo.py): strategy_source + strategy_loader 动态加载示例（明文 + 外部解密）。
- [45_talib_indicator_playbook_demo.py](./45_talib_indicator_playbook_demo.py): TA-Lib 指标组合模板示例（趋势跟随 + 均值回归 + 风险过滤，支持 `--data-source synthetic|akshare`）。
- [46_broker_profile_demo.py](./46_broker_profile_demo.py): `broker_profile` 模板注入示例（默认费率/滑点/手数一键生效）。
- [47_margin_liquidation_audit_demo.py](./47_margin_liquidation_audit_demo.py): 信用账户强平审计示例（`margin` 模式 + `liquidation_audit_df` + 报告输出）。
- [48_margin_liquidation_priority_compare.py](./48_margin_liquidation_priority_compare.py): 强平顺序对比示例（`short_first` vs `long_first`）。
- [49_on_expiry_demo.py](./49_on_expiry_demo.py): `on_expiry` 回调与流式 `expiry` 事件最小示例（期货到期结算后通知）。
- [50_framework_hooks_demo.py](./50_framework_hooks_demo.py): `on_session_start/on_session_end/on_before_trading/on_after_trading/on_portfolio_update/on_reject` 框架级钩子最小示例。
- [51_class_tick_callbacks_demo.py](./51_class_tick_callbacks_demo.py): 类风格 `on_tick` 最小示例（含 `on_order/on_trade/on_timer` 相邻回调）。
- [52_pre_open_demo.py](./52_pre_open_demo.py): `on_pre_open` 最小示例，演示“盘前决策，本次 open 成交”的推荐写法。
- [53_timer_to_pre_open_demo.py](./53_timer_to_pre_open_demo.py): “前一交易日更晚的 `on_timer` 准备，下一交易日 `on_pre_open` 执行”的双阶段示例。
- [55_functional_ml_walk_forward.py](./55_functional_ml_walk_forward.py): 函数式 `on_train_signal(ctx)` + Walk-Forward 训练评估示例。
- [56_functional_warm_start_demo.py](./56_functional_warm_start_demo.py): 函数式 `on_resume(ctx)` 热启动续跑最小示例。
- [57_functional_multi_slot_warm_start_demo.py](./57_functional_multi_slot_warm_start_demo.py): 函数式多 slot `on_resume(ctx)` 热启动续跑示例。
- [58_incremental_bootstrap_demo.py](./58_incremental_bootstrap_demo.py): 增量指标“历史预热 + 实时更新”最小示例，演示 `indicator_factory` 与 `warmup_bars`。
- [59_akshare_etf_rotation.py](./59_akshare_etf_rotation.py): AKShare + ETF 轮动最小示例，演示单个拼接后 `DataFrame` 的推荐多标输入方式。
- [60_custom_indicator_demo.py](./60_custom_indicator_demo.py): 自定义指标最小示例，同时演示 `Indicator(name, fn)` 的预计算写法和 `indicator_factory` 的增量写法。

## 流式回测与实时报告

- [25_streaming_backtest_demo.py](./25_streaming_backtest_demo.py): 流式回测回调错误模式示例。
- [26_streaming_quickstart.py](./26_streaming_quickstart.py): 流式快速开始示例。
- [27_streaming_monitoring_console.py](./27_streaming_monitoring_console.py): 终端监控示例。
- [28_streaming_alerts_and_persist.py](./28_streaming_alerts_and_persist.py): 告警与事件落盘示例。
- [29_streaming_event_report.py](./29_streaming_event_report.py): 事件 CSV 报告生成示例。
- [30_streaming_report_oneclick.py](./30_streaming_report_oneclick.py): 一键生成流式报告示例。
- [31_streaming_live_console.py](./31_streaming_live_console.py): 终端实时曲线示例。
- [32_streaming_live_web.py](./32_streaming_live_web.py): 浏览器实时曲线示例。
- [33_report_and_analysis_outputs.py](./33_report_and_analysis_outputs.py): 回测报告与结构化分析输出示例。

## 其他文件

- [benchmark_utils.py](./benchmark_utils.py): 基准辅助函数。
- [pb_mock.py](./pb_mock.py): 数据结构模拟辅助文件。

## 相关子目录

- [strategies/README.md](./strategies/README.md): 策略示例集合。
- [strategies/08_target_positions_long_short.py](./strategies/08_target_positions_long_short.py): 高级目标仓位与多空切换最小示例，演示 `rebalance_positions()`、`allow_short=True` 与调仓计划解释输出。
- [textbook/ch15_strategy_loader.py](./textbook/ch15_strategy_loader.py): 教程章节动态策略加载示例。
