# Golden Baseline Regression Report

- Baseline directory: `tests\golden\baselines`
- Current directory: `tests\golden\current`

| Scenario | Metrics Ready | Δtotal_return_pct | Δsharpe_ratio | Δmax_drawdown_pct | Orders (base/curr) | Trades (base/curr) | Equity points (base/curr) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| futures_margin | yes | 0.0000000000 | 0.0000000000 | 0.0000000000 | 2/2 | 0/0 | 5/5 |
| futures_night_weekend | yes | 0.0000000000 | 0.0000000000 | 0.0000000000 | 2/2 | 1/1 | 15/15 |
| option_basic | yes | 0.0000000000 | 0.0000000000 | 0.0000000000 | 2/2 | 0/0 | 5/5 |
| order_cancel | yes | 0.0000000000 | 0.0000000000 | 0.0000000000 | 6/6 | 2/2 | 6/6 |
| stock_minute_auction | yes | 0.0000000000 | 0.0000000000 | 0.0000000000 | 4/4 | 1/1 | 482/482 |
| stock_t1 | yes | 0.0000000000 | 0.0000000000 | 0.0000000000 | 5/5 | 2/2 | 5/5 |

## 2026-08-12: `__engine_rule_version__` 1.3.7 → 1.4.0（tick 对齐特性，Task 8）

本次规则版本升级对应股票/基金 tick 校验特性（Task 1-7）：股票/基金委托价 tick 校验从"仅期货"扩展到覆盖，且缺省最小变动价位按资产类型分流（`AssetType::Fund` → 0.001，其余 → 0.01）。

golden 套件复跑结果为 `2 passed`，四个基线场景（`futures_margin`/`option_basic`/`order_cancel`/`stock_t1`）与基线**零漂移**（上表 Δ 全为 0，orders/trades/equity 计数不变），**未重生成基线**。原因：

1. `tests/golden/strategies/` 中不存在任何 Fund/ETF 场景（仅 `futures_margin.py`、`option_basic.py`、`order_cancel.py`、`stock_t1.py`），基金缺省 tick 变化（0.01 → 0.001）无场景可触达。
2. `futures_margin.py` 中的委托全部为不带价格的市价单，tick 校验无价格可校验，因此该场景在新旧规则下行为一致。

结论：本次是"规则版本号 bump 但基线内容不变"的干净升级，未执行 `runner.py --generate-baseline`，`tests/golden/baselines/**` 保持不动。

## 2026-09-28: `__engine_rule_version__` 1.6.0 → 1.7.0（期权与可转债回测正确性）

本次规则版本升级对应期权费用拆分并默认走 ChinaMarket、到期结算改在到期日之后进行、期权缺标的价延后结算、未配置 `expiry_date` 的期权视为不到期、行权结算费、按品种费用覆盖生效。已执行 `runner.py --generate-baseline` 重新生成基线，逐文件核对如下：

| 场景 | 变化 | 解释 |
| :--- | :--- | :--- |
| option_basic | `total_commission` 0 → 6.6；`net_pnl` 0 → -6.6；`total_return_pct` -40 → 29.934；`max_drawdown_pct` 40 → 10.066；`sharpe_ratio` 随之变化；`end_market_value` 6000 → 12993.4；权益曲线 10000/10000/6000/6000/… → 10000/9993.4/8993.4/11993.4/12993.4；买单 `commission` 0 → 6.6；期末卖单 `rejected`（保证金不足）→ `new`、`position_effect` open → close | ① 期权改走 ChinaMarket 按张计费：买开 = 佣金 5 + 经手费 1.3 + 结算费 0.3 = 6.6。② 该合约没有 `expiry_date`：旧逻辑按到期日 0 处理，首次换日即以 0 内在价值作废（权益掉到 6000），之后的卖单变成卖出开仓因保证金不足被拒；新逻辑视为不到期、按市价估值，末根 bar 的卖单是平仓单，因无后续 bar 撮合而保持 `new`。 |
| futures_margin / order_cancel / stock_t1 | `metrics.json` 仅 `engine_rule_version`、`akquant_version` 变化；重新生成的 `orders.parquet` 仅订单 UUID 与行序变化 | 去掉 `id` 并按 `created_at/side/order_type/quantity/limit_price` 排序后逐列比对完全一致，故这三个 `orders.parquet` 保留旧文件不提交；`equity_curve` 与 `trades` 未变。期货场景无到期日，不受到期时点修正影响。 |

复跑 `pytest tests/golden/test_golden.py` 结果 `2 passed`。

## 2026-09-29: `__engine_rule_version__` 1.7.0 → 1.7.1（期权/可转债一期遗留项）

规则版本 1.7.0 → 1.7.1，golden 数值不变，未重新生成基线（`metrics.json` 中 `engine_rule_version` 仍为 1.7.0）。四个场景均不涉及按品种滑点、checkpoint 续跑、到期平仓或保证金账户期货到期，复跑 `pytest tests/golden/test_golden.py` 仍 `2 passed`。

## 2026-10-23: 交易时段/交易日重构阶段 3（按真实时段撮合）

`order_cancel` 的数据原为 2024-01-01～01-06 逐日（含周六 01-06）。阶段 3 起非交易日行情落在闭市时段不撮合，第 5 根 bar 上下的平仓卖单原本靠周六那根 bar 成交，现在保持 `new`（filled 3→2、new 0→1）。这是夹具日期落在非交易日（类别 (e)），不是引擎回归，因此改夹具而非接受弱化的基线：`gen_data.py::gen_cancel_data` 改为 `pd.bdate_range("2024-01-02", periods=6)`（01-02～01-09 六个交易日），只重新生成 `order_cancel.parquet` 及其 `.sha256`，只重新生成 `baselines/order_cancel/`。

逐文件核对：把新旧日期按 bar 序号对应后，订单（方向、类型、数量、成交量、限价、均价、状态、TIF、下单/更新所在 bar）、权益曲线数值、两笔交易的开平仓 bar 全部一致；差异只有时间戳（旧基线由 0.3.64 生成，只有日期的时间戳尚未改写到 15:00）、订单 UUID 与行序、`metrics.json` 的版本字段（1.7.0/0.3.64 → 1.7.2/0.3.71）。阶段 2 末（`4910acc1`）的 release 构建在新夹具上与阶段 3 逐笔一致。其余三个场景未变，基线不动。

## 2026-10-25: `__engine_rule_version__` 1.7.2 → 2.0.0（交易时段/交易日重构，0.3.72）

### runner 比较项增强

原比较只看净值余弦相似度、5 项汇总指标、成交笔数与订单状态分布，对“同一笔单换了撮合时点”不敏感：例如把 Day 单从 10:00 挪到 10:30 下（只改下单时刻，成交、状态、权益全都不变）时上述各项全部通过。新增两项，对全部场景生效：

- 逐笔比对订单的 `created_at`、`updated_at`、方向、类型、数量、成交量、限价、成交均价、手续费、状态、TIF（订单 UUID 随机，按下单时刻等确定性字段排序后比对；浮点容差 1e-9）。
- 新增基线文件 `trading_day_ends.json`（引擎实际开过的交易日及其收盘/日终结算时刻，来自 `BacktestResult.trading_day_ends`）并逐项比对；基线没有该文件时跳过。

另外 `run_test`/`run_stream_consistency_test` 支持场景配置里的 `run_kwargs`（透传 `freq`、`sessions` 等给 `run_backtest`），两个新场景也加进了流式一致性检查。

### 三个旧场景（`stock_t1`、`futures_margin`、`option_basic`）

数据原为 2024-01-01～01-05 逐日（01-01 是元旦），改为 `pd.bdate_range("2024-01-02", periods=5)`，即 01-02～01-05 与 01-08（跨一个周末）；价格序列、策略不变，重新生成数据与基线。先在旧数据上跑新引擎、再换新数据，分两步核对，差异全部可归类：

| 类别 | 涉及 | 说明 |
| :--- | :--- | :--- |
| 快照时间戳 00:00 → 15:00 | 三个场景的权益曲线、订单 `created_at`/`updated_at`、交易 `entry_time`/`exit_time` | 只有日期的日线时间戳从阶段 1 起改写为当日收盘 15:00；旧基线由 0.3.64 生成，仍是 00:00。旧数据在新引擎上跑，只有这一类差异。 |
| 日期顺移 | 同上 | 第 1～5 根 bar 由 01-01～01-05 变为 01-02～01-05、01-08，按 bar 序号逐一对应。 |
| 跨周末的持续时间 | `stock_t1` 第 2 笔交易与第 5 笔订单 `duration` 1 天 → 3 天 | 第 4 根 bar（周五）下单、第 5 根 bar（周一）成交/平仓，自然日跨度为 3 天。 |
| 利息天数 | 无 | 三个场景都没有配置利息/融资费用，跨周末不产生计息差异；权益曲线数值逐点不变。 |
| 结算时刻 | 新增 `trading_day_ends.json` | 每个交易日在 15:00 收盘结算；周末两天没有交易日。期货逐日盯市的结算价不变（日线收盘价），保证金不足拒单的金额与原基线一致。 |
| 版本字段 | `metrics.json` | `engine_rule_version` 1.7.0 → 2.0.0、`akquant_version` 0.3.64 → 0.3.72。 |

汇总指标（收益率、夏普、回撤、手续费、净盈亏、期末市值）、权益数值、订单方向/类型/数量/成交量/成交价/手续费/状态、交易盈亏全部不变。夏普不变是因为日收益按交易日（`trading_day_ends`）取点，周末不补零收益点。

### `order_cancel`

runner 不支持只刷新元数据，因此整体重生成后核对：`equity_curve.parquet`、`trades.parquet` 字节不变；`orders.parquet` 只有订单 UUID 与行序变化，保留旧文件；`metrics.json` 只有版本字段变化（1.7.2/0.3.71 → 2.0.0/0.3.72）；新增 `trading_day_ends.json`。

### 新场景 `stock_minute_auction`（A 股分钟线含集合竞价）

数据：600000.SH，2024-01-02（周二）、01-03（周三）两天，每天 241 根 1 分钟 bar：09:25 一根（区间 (09:24, 09:25] 属开盘集合竞价）、09:31～11:30、13:01～15:00（14:58～15:00 三根区间落在收盘集合竞价 14:57～15:00 内，14:58、14:59 零量）。`run_backtest(freq="1min")`，T+1，手续费万三（最低 5 元）、印花税千一、过户费十万分之二，`NextOpen`。价格分档，从成交价可直接认出撮合来源：09:25 竞价价 = base（首日 10.00、次日 10.50），连续竞价 bar 开盘价在 base+0.05～base+0.11，15:00 收盘竞价价 = base+0.30。

策略与基线核对（已逐笔打印确认）：

| 订单 | 下单 | 成交/终态 | 锁住的语义 |
| :--- | :--- | :--- | :--- |
| 市价买 100 | 01-02 10:00 bar | 01-02 10:01 成交，10.07（10:01 开盘价） | 连续竞价按下一根开盘价成交 |
| Day 限价买 100 @ 9.00 | 01-02 10:00 bar | 01-02 15:00 `expired` | Day 单在当日收盘过期 |
| 市价买 100 | 01-03 09:20 定时器（`schedule_daily`） | 01-03 09:31 成交，10.50 | 开盘竞价前下的单在首根连续竞价 bar 上按 09:25 竞价价统一撮合（不是 09:31 开盘价 10.55） |
| 市价卖 100 | 01-03 14:58 bar（收盘集合竞价内） | 01-03 15:00 成交，10.80 | 收盘竞价期内的单在 15:00 按收盘竞价价统一撮合；卖出的是首日买入的 100 股（T+1 允许） |

首日 09:20 定时器不触发：首根 bar 在 09:25，晚于 09:20（`schedule_daily` 首日规则），故只有 4 笔订单。策略查询到的时段：09:25 为 `OpeningAuction`，09:31～14:57 为 `Continuous`，14:58、15:00 为 `ClosingAuction`，09:20 定时器为 `OpeningAuction`。`trading_day_ends`：01-02 15:00、01-03 15:00。期末市值 100086.85726，手续费 16.14274（买 5.02014 + 5.02100，卖 6.10160）。

### 新场景 `futures_night_weekend`（期货夜盘跨周末）

数据：rb2405，`sessions={"rb": "CN_FUTURES_NIGHT_23"}`，`freq="1min"`；2024-01-05 周五日盘 09:01/09:02/10:00/14:59/15:00、周五夜盘 21:01/21:02/21:30/22:59/23:00、01-08 周一日盘 09:01/09:02/10:00/14:59/15:00，共 15 根。乘数 10、保证金 10%、手续费万一、`NextOpen`。

策略：第一根“交易日晚于自然日”的 bar（即周五夜盘）买开 1 手；第一根周一日盘 bar 卖平。两处都按 `self.trading_day` 分支，夜盘若被归到周五，就不会开仓，基线随即报差异。

基线核对（已逐笔打印确认）：

- 夜盘归属：周五 21:01～23:00 五根 bar 的 `trading_day` 均为 2024-01-08（周一），日盘 bar 为自然日当天。
- 买开：周五 21:01 下单，21:02 按开盘价 3522 成交，手续费 3.522。
- 卖平：周一 09:01 下单，09:02 按开盘价 3542 成交，手续费 3.542；盈亏 200（20 点 × 10），净盈亏 192.936。
- 结算时刻：`trading_day_ends` 只有两项——2024-01-05 15:00 与 2024-01-08 15:00；周五夜盘与周末之间不发生结算，持仓从周五夜盘带到周一日盘。
- 保证金按持仓逐 bar 计：开仓后占用 3523～3541（价格 × 10 × 10%），周一 09:02 平仓后归零。本场景未配置利息，不涉及计息天数。

### 回退验证

- 把 `stock_minute_auction` 的收盘竞价卖单改到 01-03 11:45（午休）的定时器下：卖单改为 13:01 按 10.56 成交，runner 报 `total_return_pct`、`max_drawdown_pct`、`net_pnl` 与第 4 笔订单的 `created_at`/`updated_at`/`avg_price`/`commission` 差异。恢复后通过。
- 只把 Day 单的下单时刻从 10:00 挪到 10:30：汇总指标、权益、成交笔数、状态分布都不变，原有比较项全部通过；只有新增的逐笔订单比对报差异。恢复后通过。
- 去掉 `futures_night_weekend` 的 `sessions`（缺省期货时段表无夜盘）：夜盘下的买单排队到周一 09:01 按 3540 成交，runner 报 `total_return_pct`、`net_pnl` 与买单的 `updated_at`/`avg_price`/`commission` 差异。恢复后通过。

复跑 `runner.py` 六个场景全部 PASSED，`pytest tests/golden/test_golden.py` `2 passed`。
