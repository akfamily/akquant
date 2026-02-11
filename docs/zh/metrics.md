# 绩效指标详解 (Performance Metrics)

本文档详细介绍了 AKQuant 回测结果 (`metrics_df`) 中各项绩效指标的含义、单位及计算方法，帮助用户深入理解策略表现。

## 指标总览

| 指标名称 (Name) | 含义 (Description) | 单位/类型 | 计算说明 |
| :--- | :--- | :--- | :--- |
| `start_time` | 回测开始时间 | Datetime | 对应回测数据的第一个 Bar 的时间。 |
| `end_time` | 回测结束时间 | Datetime | 对应回测数据的最后一个 Bar 的时间。 |
| `duration` | 回测总时长 | Timedelta | `end_time - start_time`。 |
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

## 重点指标解释

### 风险类

*   **最大回撤 (Max Drawdown)**: 衡量策略在最坏情况下的表现。例如 30% 的最大回撤意味着如果你在最高点买入，最多可能会亏损 30%。通常认为最大回撤越小越好。
*   **波动率 (Volatility)**: 衡量收益的不确定性。波动率越高，收益起伏越大。

### 收益风险比类

*   **夏普比率 (Sharpe Ratio)**: 投资界的"性价比"指标。它衡量的是每承担一单位总风险，产生了多少超额收益。通常大于 1 是可接受的，大于 2 是非常优秀的。
*   **索提诺比率 (Sortino Ratio)**: 类似于夏普比率，但只考虑"有害的波动"（下行波动）。如果一个策略经常暴涨（上行波动大），夏普比率可能会低估它，而索提诺比率则更公允。
*   **卡尔玛比率 (Calmar Ratio)**: 年化收益率与最大回撤之比。衡量"收益回撤比"，是很多基金经理非常看重的指标。
