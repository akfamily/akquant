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
