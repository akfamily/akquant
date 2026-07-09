# QMF 期权合约/额度只读（Phase A）设计

日期：2026-07-09 · 分支：dev · 状态：设计已批准（沿用 3a/3c 薄便捷方法）

## 目标

为 akquant QMF broker 补全期权**合约元数据 / 额度 / 提示类只读查询**，对接 `chibi_quant`
`/api/v1/option/*`。沿用 3a/3c 风格：非协议**便捷方法**，返回柜台原始行（`list[dict]` 或
`dict`）；需期权会话（`enable_options`）否则 `RuntimeError`；注入 `fund_account`（期权账户
字段服务端按会话注入）；零 core/mapper 改动，仅改 `brokers/qmf/{client,adapter}.py` + tests + docs。

## 端点与方法（7 个只读/计算）

| 方法 | 端点 | 返回 | 必填参数 |
|---|---|---|---|
| `query_option_contracts(stock_code=None, option_code=None)` | `/option/contracts` | list | — |
| `query_option_underlyings(stock_code=None)` | `/option/underlyings` | list | — |
| `query_option_strategies(optcomb_code=None)` | `/option/strategies` | list | — |
| `query_option_position_limits(stock_code=None)` | `/option/position-limits` | list | — |
| `query_option_contract_tips(money_type="0")` | `/option/contract-tips` | list | money_type |
| `query_option_enable_amount(exchange_type, option_code, opt_entrust_price, entrust_prop, entrust_bs, entrust_oc, covered_flag=None)` | `/option/enable-amount` | dict | exchange_type/option_code/opt_entrust_price/entrust_prop/entrust_bs/entrust_oc |
| `query_option_underlying_amount_tip(exchange_type, option_code, entrust_amount, entrust_bs, entrust_oc)` | `/option/underlying-amount-tip` | dict | 全部 |

约定：可选参数仅在非 `None` 时进 payload（同 3c `covered_transferable` 的 stock_code 处理）；
`entrust_bs` `1`买/`2`卖，`entrust_oc` `O`开/`C`平/`X`行权（传原始字符串）。

## 分层

- `client.py`：7 薄 HTTP 方法（经 `_post`，读 `list(...)`，计算类返回 `dict`）。
- `adapter.py`：7 同名便捷方法，`_require_option_client()` 守卫后委托。标注"非协议；原始行透传"。
- `mapper.py`：不改。

## 测试

- 单元（client）：httpx `MockTransport` 断言 path + `fund_account` 注入 + 必填/可选参数透传 + 返回值（list/dict）。
- 单元（adapter）：fake option client 断言委托 + 未启用期权 `RuntimeError`。
- ruff（E,F,I,D）+ 全量 pytest。

## 范围外 / 决策记录

- **`option/contract-confirm` 不在本期**：`合约确认` 疑似带写副作用（T2 confirm），归入后续"confirm 写"批（与 3d `combo-confirm` 一起处理），避免把写当读建模。
- 3d 组合、F 可转债、covered/exercise（已在 3c）均不在本期。
