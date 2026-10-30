# 市场规则与默认值

`run_backtest` 不传任何参数时是“无摩擦”回测：佣金、印花税都是 0，不开 T+1，最小交易单位是 1；做 A 股回测时，应该传 `broker_profile` 或显式设置费率。

## 默认值

| 参数 | 默认 |
|---|---|
| `initial_cash` | 100000 |
| `commission_rate` / `stamp_tax_rate` / `min_commission` | 0 |
| `t_plus_one` | 不开（没有 `broker_profile` 时） |
| `lot_size` | 1 |
| `timezone` | `Asia/Shanghai` |
| 成交量限制 | 不限制 |

## A 股：用券商模板

`broker_profile="cn_stock_miniqmt"` 的设置是：佣金 0.0003、最低 5 元，印花税 0.0005（只在卖出时收），过户费 0.00001，滑点 0.02%，`lot_size` 100，开 T+1，单笔成交不超过当根 bar 成交量的 20%（超出部分转为部分成交）。另外两个模板是 `cn_stock_t1_low_fee` 和 `cn_stock_sim_high_slippage`。显式传入的参数优先于模板。

```json
{"broker_profile": "cn_stock_miniqmt", "initial_cash": 1000000}
```

- **印花税**：模板用的是 2023-08-28 起的现行税率 0.0005。回测在此之前的行情，要显式传 `"stamp_tax_rate": 0.001`。
- **T+1**：当日买入的股票当日不能卖出，卖单会被拒，原因写在 `orders` 明细的 `reject_reason` 里。
- **手数**：`lot_size` 为 100 时，买单数量必须是 100 的整数倍，否则会被拒；`order_target_*` 和 sizer 会自动向下取整到整手。
- 拒单不会抛异常，回测会照常跑完。用 `get_run_details(run_id, "orders")` 查看 `status` 和 `reject_reason`。

## 期货

一期 MCP 的 `config` 不开放期货合约参数（合约乘数、保证金率等）。期货回测请直接写脚本调用 `run_backtest`，相关参数用 `search_api("futures")` 或 `search_api("margin")` 查询。
