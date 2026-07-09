# QMF 可转债交易（Phase F）设计

日期：2026-07-09 · 分支：dev · 状态：设计已批准（薄便捷方法，证券侧）

## 目标

为 akquant QMF broker 补全**可转债交易**（下单/撤单）与相关查询，对接 `chibi_quant`
`/api/v1/trading/convertible-bond-*` 与 `/api/v1/account/convertible-bond-orders`、
`/api/v1/account/bond-putback-info`。这些走**证券会话**（服务端 `invoke_t2`），**始终可用**
（无需 `enable_options`）。沿用 3a/3c/A 风格：非协议**便捷方法**，返回柜台原始行（写=`dict`，
读=`list[dict]`）；client 注入 `fund_account`；零 core/mapper 改动，仅改
`brokers/qmf/{client,adapter}.py` + tests + docs。

## 端点与方法（4 个，证券侧）

| 方法 | 端点 | 返回 | 必填参数（除 fund_account） |
|---|---|---|---|
| `place_convertible_bond_order(stock_code, exchange_type, entrust_prop, entrust_amount, stock_account=None, stb_stock_property=None)` | `/trading/convertible-bond-order` | dict（写） | stock_code/exchange_type/entrust_prop/entrust_amount |
| `cancel_convertible_bond_order(entrust_no)` | `/trading/convertible-bond-cancel` | dict（写） | entrust_no |
| `query_convertible_bond_orders(stock_code=None, entrust_no=None, query_flag=None, en_entrust_prop=None)` | `/account/convertible-bond-orders` | list | — |
| `query_bond_putback_info(stock_code=None)` | `/account/bond-putback-info` | list | — |

约定：client 一律注入 `fund_account`；可选参数仅在非 `None` 时进 payload。

## 分层

- `client.py`：4 薄 HTTP 方法（`_post`，写返回 `dict`，读 `list(...)`）。
- `adapter.py`：4 同名便捷方法，**直接委托 `self._client`**（证券会话恒在，无 `_require_option_client` 守卫）。标注"非协议；原始行透传"。
- `mapper.py`：不改。

## 测试

- 单元（client）：httpx `MockTransport` 断言 path + `fund_account` 注入 + 必填/可选参数透传 + 返回值（dict/list）。
- 单元（adapter）：fake 证券 client 断言委托（无期权会话也可用）。
- ruff（E,F,I,D）+ 全量 pytest。

## 范围外

3d 组合、confirm-write 批（contract-confirm + combo-confirm）；可转债转股/回售的语义建模（本期原始透传，entrust_prop 由调用方按柜台语义传）。
