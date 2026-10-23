# Golden Baseline Regression Report

- Baseline directory: `tests\golden\baselines`
- Current directory: `tests\golden\current`

| Scenario | Metrics Ready | Δtotal_return_pct | Δsharpe_ratio | Δmax_drawdown_pct | Orders (base/curr) | Trades (base/curr) | Equity points (base/curr) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| futures_margin | yes | 0.0000000000 | 0.0000000000 | 0.0000000000 | 2/2 | 0/0 | 5/5 |
| option_basic | yes | 0.0000000000 | 0.0000000000 | 0.0000000000 | 2/2 | 0/0 | 5/5 |
| order_cancel | yes | 0.0000000000 | 0.0000000000 | 0.0000000000 | 6/6 | 2/2 | 6/6 |
| stock_t1 | yes | 0.0000000000 | 0.0000000000 | 0.0000000000 | 5/5 | 1/1 | 5/5 |

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
