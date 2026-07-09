# QMF 期权组合策略（Phase 3d）设计

日期：2026-07-09 · 分支：dev · 状态：设计已批准（薄便捷方法）

## 目标

为 akquant QMF broker 补全**期权组合策略**（多腿）下单/确认/查询，对接 `chibi_quant`
`/api/v1/option/combo-*` 与 `history-combo-orders`。沿用 3a/3c/A 风格：非协议**便捷方法**，
返回柜台原始行（写=`dict`，读=`list[dict]`）；需期权会话（`enable_options`）否则 `RuntimeError`；
client 注入 `fund_account`（期权账户字段服务端按会话注入）；零 core/mapper 改动。

## 关键设计结论

柜台把组合建模为**固定两腿**：`first_option_code`/`first_opthold_type` +
`second_option_code`/`second_opthold_type` + 策略码 `optcomb_code` + 方向 `comb_bs`
（`1`=组合 / `2`=拆分），**不是可变长腿列表**。故无需新请求形态，仍用扁平薄方法透传。

## 端点与方法（5 个）

| 方法 | 端点 | 返回 | 必填参数（除注入项） |
|---|---|---|---|
| `place_option_combo_order(exchange_type, optcomb_code, first_option_code, first_opthold_type, second_option_code, second_opthold_type, entrust_amount, comb_bs, optcomb_id=None)` | `/option/combo-order` | dict（写） | 全部（除 optcomb_id） |
| `confirm_option_combo(exchange_type, optcomb_code, comb_bs, first_option_code=None, first_opthold_type=None, second_option_code=None, second_opthold_type=None, optcomb_id=None)` | `/option/combo-confirm` | dict（写） | exchange_type/optcomb_code/comb_bs |
| `query_option_combo_orders(optcomb_code=None, optcomb_id=None)` | `/option/combo-orders` | list | — |
| `query_option_combo_positions(optcomb_code=None, query_mode=None)` | `/option/combo-positions` | list | — |
| `query_option_history_combo_orders(start_date, end_date)` | `/option/history-combo-orders` | list | start_date/end_date |

约定：可选参数仅在非 `None` 时进 payload；`comb_bs` `1`组合/`2`拆分；`query_mode` `0`明细/`1`汇总。

## 分层

- `client.py`：5 薄 HTTP 方法（`_post`，写 `dict`，读 `list(...)`）。
- `adapter.py`：5 同名便捷方法，`_require_option_client()` 守卫后委托。标注"非协议；原始行透传"。
- `mapper.py`：不改。

## 测试

- 单元（client）：httpx `MockTransport` 断言 path + `fund_account` 注入 + 必填/可选参数透传 + 返回值（dict/list）。
- 单元（adapter）：fake option client 断言委托 + 未启用期权 `RuntimeError`。
- ruff（E,F,I,D）+ 全量 pytest。

## 范围外

**confirm-write 尾巴**：`option/contract-confirm`（Phase A 排除的合约确认，唯一未覆盖端点）。
组合的 Unified 建模（本期原始透传）。
