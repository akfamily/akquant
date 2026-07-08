# QMF 期权行权查询 + 备兑划转（Phase 3c）设计

日期：2026-07-09 · 分支：dev · 状态：设计已批准（全量 3c + 薄便捷方法）

## 目标

为 akquant QMF broker 补全期权**行权相关查询**与**备兑证券划转**，对接 `chibi_quant`
的 `/api/v1/option/*` 端点。沿用 Phase 3a 既定风格：非 `TraderGateway` 协议的**便捷方法**，
返回柜台**原始行**（读=`list[dict]`，写=`dict`）；均需期权会话（`enable_options`），否则
`RuntimeError`。零 akquant-core 改动，仅改 `brokers/qmf/{client,adapter}.py` + tests + docs。

## 端点与方法映射（全 8 个）

**行权查询（只读）**
- `query_option_exercise_assignments()` → POST `/option/exercise-assignments`（338024 行权指派）
- `query_option_exercise_settlements()` → POST `/option/exercise-settlements`（行权交割）
- `query_option_exercise_debts()` → POST `/option/exercise-debts`（行权负债）
- `query_option_history_exercise_assignments(start_date, end_date)` → POST `/option/history-exercise-assignments`
- `query_option_history_exercise_settlements(start_date, end_date)` → POST `/option/history-exercise-settlements`

**备兑**
- `query_option_covered_shortages()` → POST `/option/covered-shortages`（备兑不足，读）
- `query_option_covered_transferable(exchange_type, lock_direction, stock_code=None)` → POST `/option/covered-transferable`（可划转，读）
- `covered_transfer(exchange_type, stock_code, entrust_amount, lock_direction)` → POST `/option/covered-transfer`（338031 备兑证券划转，**写**）

## 请求约定

- 所有方法经 `QMFHttpClient._post`（统一信封 `result/msg/data`，`result!="0"` 抛 `QMFApiError`；带 Bearer）。
- 一律注入 `{"fund_account": self.fund_account, ...}`；期权账户字段（`fund_account_opt`/`option_account`/`stock_account`/`seat_no`）由服务端按会话三层注入，client 不传。
- 必填参数按服务端 schema：`covered-transferable` 需 `exchange_type`+`lock_direction`；`covered-transfer` 需 `exchange_type`+`stock_code`+`entrust_amount`+`lock_direction`；两个 history 需 `start_date`+`end_date`。其余可选过滤（position_str/request_num/option_code/opthold_type）本期**不暴露**（YAGNI，后续可加）。
- `lock_direction` 传原始字符串 `"1"`（锁定）/`"2"`（解锁）。

## 分层

- `client.py`：8 个薄 HTTP 方法（读返回 `list(...)`，写返回 `dict`）。
- `adapter.py`：8 个同名便捷方法，经既有 `self._require_option_client()` 守卫后委托 client（未启用期权抛 `RuntimeError`）。均标注"非协议；原始行透传"。
- `mapper.py`：**不改**（原始行透传，不做 Unified 建模，与 Phase 3a 一致）。

## 测试

- 单元（client）：仿 `test_gateway_qmf_option_client.py` 的 httpx `MockTransport`，断言各方法命中正确 path、注入 `fund_account`、透传必填参数、返回 `data`。
- 单元（adapter）：仿 `test_gateway_qmf_option_adapter.py`，用 fake option client 断言便捷方法委托正确、未启用期权时 `RuntimeError`。
- ruff（E,F,I,D）、全量 pytest 通过。
- e2e vs `chibi_quant --mock`：mock 有对应 opt_exercise_*/opt_covered_* fixtures，可选手动联调（HTTP 读/写往返）。

## 范围外

3d（组合多腿）、A（合约/额度只读）、F（可转债）；covered-transfer 的 Unified 建模（本期用薄封装）；可选过滤参数。
