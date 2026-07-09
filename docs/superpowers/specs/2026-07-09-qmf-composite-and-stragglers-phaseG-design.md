# QMF 综合业务(composite) + 期权尾巴接口（Phase G）设计

日期：2026-07-09 · 分支：dev · 状态：设计已批准（薄便捷方法）

## 目标

补齐服务端最新更新后 akquant 尚未对接的接口，沿用既有薄便捷方法风格（非协议、原始行透传、
client 注入 `fund_account`、零 core/mapper 改动，仅改 `brokers/qmf/{client,adapter}.py` + tests + docs）：

1. **综合业务 composite（证券侧，本次新增）**：`invoke_t2`，始终可用（无期权守卫）。
2. **期权尾巴（期权侧，需 `enable_options`，`_require_option_client` 守卫）**：合约确认 + 历史账单/对账单。

## 端点与方法

### 综合业务（证券侧）

| 方法 | 端点 | 返回 | 必填（除 fund_account） |
|---|---|---|---|
| `place_composite_order(exchange_type, stock_account, stock_code, entrust_price, entrust_amount, entrust_prop, entrust_bs, extra=None)` | `/trading/composite-order` | dict（写） | 7 项 |
| `cancel_composite_order(entrust_no, entrust_reference=None)` | `/trading/composite-cancel` | dict（写） | entrust_no |
| `query_composite_orders(**filters)` | `/account/composite-orders` | list | — |
| `query_composite_trades(**filters)` | `/account/composite-trades` | list | — |

- `place_composite_order`：7 个必填做显式参数；其余 ~16 个可选字段（relation_name/cbpconfer_id/
  agreement_id/subscribe_balance/reduction_type/registe_sure_flag/… ）经 `extra: dict[str, str] | None`
  合并进 payload（仅在提供时）。
- 查询方法可选过滤（exchange_type/stock_account/stock_code/entrust_no/query_type/…）经
  `**filters` 透传，仅非空进 payload（保持薄封装、不硬编码全部字段）。

### 期权尾巴（期权侧）

| 方法 | 端点 | 返回 | 必填 |
|---|---|---|---|
| `option_contract_confirm(exchange_type, option_code)` | `/option/contract-confirm` | list | exchange_type/option_code |
| `query_option_history_bill(begin_date, end_date, money_type="0")` | `/option/history-bill` | list | begin_date/end_date/money_type |
| `query_option_history_statements(begin_date, end_date, query_mode)` | `/option/history-statements` | list | begin_date/end_date/query_mode |

注：history-bill/statements 服务端 schema 用 `begin_date`/`end_date`（非 `start_date`），payload key 须一致；
`query_mode` `1`每日汇总/`2`时间段汇总；`option_contract_confirm` 疑似带 confirm 写副作用（返回 list），
调用方按需使用。

## 分层

- `client.py`：7 薄 HTTP 方法（`_post`，写 `dict`，读 `list(...)`；注入 `fund_account`，可选/extra 仅非空进 payload）。
- `adapter.py`：7 便捷方法。composite 4 个直委托 `self._client`（无守卫）；期权 3 个经 `_require_option_client()`。
- `mapper.py`：不改。

## 测试

- 单元（client）：httpx `MockTransport` 断言 path + `fund_account` 注入 + 必填/extra/filters 透传 + 返回值。
- 单元（adapter）：fake client 断言委托；期权 3 个断言未启用期权 `RuntimeError`。
- ruff（E,F,I,D）+ 全量 pytest。

## 范围外

IPO 打新、账户资料类（shareholders/investment-preference/repo-impawn-codes/sub-asset-history）、
`audit/*`、`auth/logout`、`auth/password` —— 非实盘交易主链，暂不接。
