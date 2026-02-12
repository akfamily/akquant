# 回测结果与指标详解 (Backtest Results & Metrics)

本文档详细介绍了 AKQuant 回测结果 (`metrics_df`) 中各项绩效指标的含义、单位及计算方法，帮助用户深入理解策略表现。

## 指标总览

| 指标名称 (Name) | 含义 (Description) | 单位/类型 | 计算说明 |
| :--- | :--- | :--- | :--- |
| `start_time` | 回测开始时间 | Datetime | 对应回测数据的第一个 Bar 的时间。 |
| `end_time` | 回测结束时间 | Datetime | 对应回测数据的最后一个 Bar 的时间。 |
| `duration` | 回测总时长 | Timedelta | `end_time - start_time`。注意：在 Python 中返回 `datetime.timedelta` 对象。 |
| `total_bars` | 总 Bar 数量 | Int | 回测经历的 K 线总数。 |
| `trade_count` | 交易笔数 | Int | 完成平仓的交易总数 (Round-trip)。 |
| `initial_market_value` | 初始市值 | Float | 初始资金 (通常为 Cash)。 |
| `end_market_value` | 结束市值 | Float | 回测结束时的总资产 (Cash + 持仓市值)。 |
| `total_pnl` | 总盈亏 | Float | `end_market_value - initial_market_value`。 |
| `unrealized_pnl` | 未实现盈亏 | Float | 结束时持仓的浮动盈亏。 |
| `total_return_pct` | 总收益率 | **% (百分比)** | `(结束市值 - 初始市值) / 初始市值 * 100`。 |
| `annualized_return` | 年化收益率 | Ratio (小数) | `(1 + 总收益率)^(1/年数) - 1`。注意这里是小数，如 0.2 表示 20%。 |
| `volatility` | 年化波动率 | Ratio (小数) | 收益率标准差 * sqrt(252)。 |
| `total_profit` | 总盈利 | Float | 所有盈利交易的盈利之和。 |
| `total_loss` | 总亏损 | Float | 所有亏损交易的亏损之和。 |
| `total_commission` | 总手续费 | Float | 交易产生的总佣金。 |
| `max_drawdown` | 最大回撤比率 | Ratio (小数) | 历史最大回撤幅度（如 0.1 表示 10%）。 |
| `max_drawdown_value` | 最大回撤金额 | Float | 历史最大回撤的绝对金额。 |
| `max_drawdown_pct` | 最大回撤百分比 | **% (百分比)** | `max_drawdown * 100`。 |
| `win_rate` | 胜率 | **% (百分比)** | `盈利次数 / 总交易次数 * 100`。 |
| `loss_rate` | 败率 | **% (百分比)** | `亏损次数 / 总交易次数 * 100`。 |
| `winning_trades` | 盈利次数 | Int | 盈利大于 0 的交易次数。 |
| `losing_trades` | 亏损次数 | Int | 盈利小于 0 的交易次数。 |
| `avg_pnl` | 平均盈亏 | Float | 每笔交易的平均净盈亏。 |
| `avg_return_pct` | 平均收益率 | **% (百分比)** | 每笔交易收益率的平均值。 |
| `avg_trade_bars` | 平均持仓 K 线数 | Float | 平均每笔交易持有的 Bar 数量。 |
| `avg_profit` | 平均盈利 | Float | 盈利交易的平均金额。 |
| `avg_profit_pct` | 平均盈利比例 | **% (百分比)** | 盈利交易的平均收益率。 |
| `avg_winning_trade_bars`| 平均盈利持仓时长 | Float | 盈利交易的平均持仓 Bar 数。 |
| `avg_loss` | 平均亏损 | Float | 亏损交易的平均金额。 |
| `avg_loss_pct` | 平均亏损比例 | **% (百分比)** | 亏损交易的平均收益率。 |
| `avg_losing_trade_bars` | 平均亏损持仓时长 | Float | 亏损交易的平均持仓 Bar 数。 |
| `largest_win` | 最大单笔盈利 | Float | 单笔交易的最大盈利金额。 |
| `largest_win_pct` | 最大单笔盈利比例 | **% (百分比)** | 单笔交易的最大收益率。 |
| `largest_win_bars` | 最大盈利持仓时长 | Float | 产生最大盈利那笔交易的持仓 Bar 数。 |
| `largest_loss` | 最大单笔亏损 | Float | 单笔交易的最大亏损金额。 |
| `largest_loss_pct` | 最大单笔亏损比例 | **% (百分比)** | 单笔交易的最大亏损率。 |
| `largest_loss_bars` | 最大亏损持仓时长 | Float | 产生最大亏损那笔交易的持仓 Bar 数。 |
| `max_wins` | 最大连续盈利次数 | Int | 连续盈利的最多次数。 |
| `max_losses` | 最大连续亏损次数 | Int | 连续亏损的最多次数。 |
| `sharpe_ratio` | 夏普比率 | Ratio | `(年化收益率 - 无风险利率) / 年化波动率`。无风险利率默认 0。 |
| `sortino_ratio` | 索提诺比率 | Ratio | `(年化收益率 - 无风险利率) / 下行波动率`。只考虑下行风险。 |
| `profit_factor` | 盈亏比 | Ratio | `总盈利 / abs(总亏损)`。 |
| `ulcer_index` | 溃疡指数 | Ratio | 衡量回撤深度和持续时间的指标。 |
| `upi` | 溃疡绩效指数 | Ratio | `(年化收益率 - 无风险利率) / 溃疡指数`。 |
| `equity_r2` | 权益曲线 R² | Ratio | 权益曲线相对于时间的线性回归拟合度 (0-1)，越接近 1 表示收益越稳定。 |
| `std_error` | 标准误差 | Float | 权益曲线回归的标准误差。 |
| `calmar_ratio` | 卡尔玛比率 | Ratio | `年化收益率 / 最大回撤比率`。 |
| `exposure_time_pct` | 市场暴露时间百分比 | **% (百分比)** | 持有仓位（非空仓）的时间占比。 |
| `var_95` | 95% 风险价值 | Ratio (小数) | 95% 置信度下的每日最大亏损比例。 |
| `var_99` | 99% 风险价值 | Ratio (小数) | 99% 置信度下的每日最大亏损比例。 |
| `cvar_95` | 95% 条件风险价值 | Ratio (小数) | 超过 95% VaR 阈值后的平均亏损比例（预期亏损）。 |
| `cvar_99` | 99% 条件风险价值 | Ratio (小数) | 超过 99% VaR 阈值后的平均亏损比例（预期亏损）。 |
| `sqn` | 系统质量指数 (SQN) | Float | `(平均盈亏 / 盈亏标准差) * sqrt(交易次数)`。 |
| `kelly_criterion` | 凯利公式比例 | Ratio (小数) | `胜率 - (败率 / 盈亏比)`。 |

## 重点指标解释

### 风险类

*   **最大回撤 (Max Drawdown)**: 衡量策略在最坏情况下的表现。例如 30% 的最大回撤意味着如果你在最高点买入，最多可能会亏损 30%。通常认为最大回撤越小越好。
*   **波动率 (Volatility)**: 衡量收益的不确定性。波动率越高，收益起伏越大。
*   **风险价值 (VaR) 与 条件风险价值 (CVaR)**:
    *   **VaR (Value at Risk)**: 在特定置信度下（如95%），策略在一天内可能发生的最大损失。
    *   **CVaR (Conditional VaR)**: 又称 Expected Shortfall，衡量当损失超过 VaR 时的平均损失。它比 VaR 更能反映极端尾部风险。
*   **市场暴露时间 (Exposure Time)**: 策略持有头寸（非空仓）的时间占比。低暴露时间意味着资金大部分时间是安全的（现金），但也可能错失机会。

### 收益风险比类

*   **夏普比率 (Sharpe Ratio)**: 投资界的"性价比"指标。它衡量的是每承担一单位总风险，产生了多少超额收益。通常大于 1 是可接受的，大于 2 是非常优秀的。
*   **索提诺比率 (Sortino Ratio)**: 类似于夏普比率，但只考虑"有害的波动"（下行波动）。如果一个策略经常暴涨（上行波动大），夏普比率可能会低估它，而索提诺比率则更公允。
*   **卡尔玛比率 (Calmar Ratio)**: 年化收益率与最大回撤之比。衡量"收益回撤比"，是很多基金经理非常看重的指标。
*   **系统质量指数 (SQN)**: 衡量交易系统的稳定性。SQN 越高，越容易通过加大仓位来获利。一般认为 >2.0 为合格，>3.0 为优秀，>7.0 为圣杯。
*   **凯利公式 (Kelly Criterion)**: 基于胜率和盈亏比计算的理论最佳仓位比例。注意凯利公式通常过于激进，实盘中常使用 "半凯利" (Half-Kelly) 甚至更低比例。

## 权益与现金曲线 (Curves)

`result` 对象提供了权益和现金随时间变化的曲线数据，方便进行绘图和分析。

| 属性名称 (Property) | 含义 (Description) | 类型 (Type) | 说明 |
| :--- | :--- | :--- | :--- |
| `equity_curve` | 权益曲线 | `pandas.Series` | 索引为时间 (`Datetime`)，值为账户总权益 (`Equity`)。反映账户资产净值的变化趋势。 |
| `cash_curve` | 现金曲线 | `pandas.Series` | 索引为时间 (`Datetime`)，值为账户可用现金 (`Cash`)。反映账户流动资金的变化情况，有助于资金管理分析。 |

## 交易明细 (Trades)

`result.trades_df` 包含了每笔平仓交易的详细信息。

| 指标名称 (Name) | 含义 (Description) | 单位/类型 | 计算说明 |
| :--- | :--- | :--- | :--- |
| `symbol` | 标的代码 | String | 交易标的。 |
| `entry_time` | 开仓时间 | Datetime | 开仓成交时间。 |
| `exit_time` | 平仓时间 | Datetime | 平仓成交时间。 |
| `entry_price` | 开仓均价 | Float | 开仓成交均价。 |
| `exit_price` | 平仓均价 | Float | 平仓成交均价。 |
| `quantity` | 交易数量 | Float | 成交数量。 |
| `side` | 交易方向 | String | `long` (做多) 或 `short` (做空)。 |
| `pnl` | 毛盈亏 | Float | 不包含手续费的盈亏。 |
| `net_pnl` | 净盈亏 | Float | `pnl - commission`。 |
| `return_pct` | 收益率 | Float | 交易收益率 (小数)。 |
| `commission` | 手续费 | Float | 交易产生的佣金。 |
| `duration_bars` | 持仓 K 线数 | Int | 持仓期间经历的 Bar 数量。 |
| `duration` | 持仓时长 | Timedelta | `exit_time - entry_time`。注意：在 Python 中返回 `datetime.timedelta` 对象。 |
| `mae` | 最大不利偏移 | **% (百分比)** | 持仓期间最大亏损幅度 (相对于开仓价)。 |
| `mfe` | 最大有利偏移 | **% (百分比)** | 持仓期间最大盈利幅度 (相对于开仓价)。 |
| `entry_tag` | 开仓标签 | String | 开仓订单的标签。 |
| `exit_tag` | 平仓标签 | String | 平仓订单的标签。 |
| `entry_portfolio_value` | 开仓总资产 | Float | 开仓时刻的账户总权益 (Equity)。用于计算仓位占比。 |
| `max_drawdown_pct` | 单笔最大回撤 | **% (百分比)** | 持仓期间从最高点回落的最大幅度。 |

## 订单记录 (Orders)

`result.orders_df` 包含了所有历史订单记录（包括已成交、取消、拒绝等）。

| 指标名称 (Name) | 含义 (Description) | 单位/类型 | 计算说明 |
| :--- | :--- | :--- | :--- |
| `id` | 订单 ID | String | 唯一标识符。 |
| `symbol` | 标的代码 | String | 交易标的。 |
| `side` | 订单方向 | String | `buy` (买入) 或 `sell` (卖出)。 |
| `order_type` | 订单类型 | String | `market` (市价), `limit` (限价), `stop` (止损)。 |
| `quantity` | 订单数量 | Float | 下单数量。 |
| `filled_quantity` | 成交数量 | Float | 实际成交数量。 |
| `limit_price` | 限价 | Float | 限价单的价格 (Market 单为 NaN)。 |
| `stop_price` | 触发价 | Float | 止损单的触发价格。 |
| `avg_price` | 成交均价 | Float | 实际成交的平均价格。 |
| `commission` | 手续费 | Float | 订单产生的手续费。 |
| `status` | 订单状态 | String | `filled`, `cancelled`, `rejected` 等。 |
| `time_in_force` | 有效期 | String | `gtc`, `day`, `ioc` 等。 |
| `created_at` | 创建时间 | Datetime | 订单创建时间。 |
| `updated_at` | 更新时间 | Datetime | 订单最后更新时间。 |
| `duration` | 存续时长 | Timedelta | `updated_at - created_at`。 |
| `filled_value` | 成交金额 | Float | `filled_quantity * avg_price`。 |
| `tag` | 订单标签 | String | 用户自定义标签。 |
| `reject_reason` | 拒绝原因 | String | 订单被拒绝的原因 (若有)。 |

## 持仓记录 (Positions)

`result.positions_df` 包含了每日（或每 Bar）持仓快照。

| 指标名称 (Name) | 含义 (Description) | 单位/类型 | 计算说明 |
| :--- | :--- | :--- | :--- |
| `date` | 日期 | Datetime | 持仓快照时间。 |
| `symbol` | 标的代码 | String | 交易标的。 |
| `long_shares` | 多头持仓 | Float | 多头方向的持仓数量。 |
| `short_shares` | 空头持仓 | Float | 空头方向的持仓数量。 |
| `close` | 收盘价 | Float | 当时的收盘价。 |
| `equity` | 账户权益 | Float | 包含持仓盈亏的账户总权益。 |
| `market_value` | 持仓市值 | Float | 当前持仓的市场价值。 |
| `margin` | 占用保证金 | Float | 当前持仓占用的保证金。 |
| `unrealized_pnl` | 未实现盈亏 | Float | 持仓浮动盈亏。 |
| `entry_price` | 持仓均价 | Float | 当前持仓的平均成本价格。 |
