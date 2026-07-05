# QMF 查询扩充设计（Phase 3a）

日期：2026-07-03
状态：待评审
前置：Phase 1（证券）+ 1.5（插件化）+ 2（期权）已合入 dev。本阶段延续，**零核心改动**，全部落在 `brokers/qmf/`。

## 1. 目标

补齐 QMF 的只读查询能力：
1. **期权资产并入 `query_account`**（M1 汇总口径）。
2. 新增只读查询：证券**交割单**、**资金流水**；期权**历史委托/成交/交割单**。

均为读操作、低风险；组合/行权/备兑相关查询留待 3c/3d。

## 2. 已确认决策

- **账户合并 = M1 汇总**：`query_account` 在启用期权时返回**合并**的单一 `UnifiedAccount`：
  `equity = 证券 asset_balance + 期权 total_asset`、`cash = 证券 current_balance + 期权 current_balance`、
  `available_cash = 证券 enable_balance + 期权 enable_balance`，`account_id` 用证券资金账户。
  未启用期权时行为不变（仅证券）。
- **历史/流水查询返回原始 `list[dict]`**（YAGNI，不为只读历史造 Unified 模型）。这些**不属于 `TraderGateway`
  协议**，作为 QMFTraderGateway 的附加便捷方法暴露，直接委托到对应会话。

## 3. chibi_quant 接口（已核实）

- 期权资产 `POST /api/v1/option/assets`（`OptAssetsRequest.money_type` 必填，默认 "0"）；返回
  `total_asset/current_balance/enable_balance/...`。**client 已有 `query_option_assets(money_type="0")`**。
- 证券交割单 `POST /api/v1/account/settlements`（`fund_account`、`start_date` 必填、`end_date` 必填、
  `stock_type?` + 分页 `position_str?/request_num?`）→ 数组。
- 证券资金流水 `POST /api/v1/account/fund-flow`（`fund_account` 必填 + 分页；日期可选）→ 数组。
- 期权历史 `POST /api/v1/option/history-orders`、`/history-trades`、`/history-settlements`
  （`start_date/end_date` 必填 + 分页）→ 数组。
- 会话字段由网关注入；client 只发 `fund_account` + 业务字段。

## 4. 组件设计（qmf 包内）

### 4.1 `mapper.py`（追加一个纯函数）
- `merge_option_assets(account: UnifiedAccount, opt_assets: dict) -> UnifiedAccount`：
  把期权资产（`total_asset/current_balance/enable_balance`）累加到已解析的证券 `UnifiedAccount`，
  返回新的合并 `UnifiedAccount`（`account_id` 保持证券侧）。

### 4.2 `client.py`（追加只读查询方法）
- `query_settlements(start_date, end_date, stock_type=None) -> list[dict]` → `/api/v1/account/settlements`
- `query_fund_flow(start_date=None, end_date=None) -> list[dict]` → `/api/v1/account/fund-flow`
  （日期非空才带上）
- `query_option_history_orders(start_date, end_date) -> list[dict]` → `/api/v1/option/history-orders`
- `query_option_history_trades(start_date, end_date) -> list[dict]` → `/api/v1/option/history-trades`
- `query_option_history_settlements(start_date, end_date) -> list[dict]` → `/api/v1/option/history-settlements`
- 均自动注入 `fund_account`，走现有 `_post` 信封处理。

### 4.3 `adapter.py`（`QMFTraderGateway`）
- `query_account()` 改为：解析证券资金 → 若有期权会话则 `mapper.merge_option_assets(account,
  self._option_client.query_option_assets())`。
- 附加便捷方法（原始行透传）：`query_settlements(...)`、`query_fund_flow(...)` 委托证券 client；
  `query_option_history_orders/trades/settlements(...)` 委托期权 client——**无期权会话时抛清晰
  RuntimeError**（"期权历史查询需 enable_options"）。

## 5. 数据流

- 组合账户：`query_account()` → 证券 `query_funds` + 期权 `query_option_assets` → `merge_option_assets`
  → 单一 `UnifiedAccount`（合并 equity/cash/available_cash）。
- 历史/流水：`trader.query_settlements(start, end)` → 证券 client `/account/settlements` → 原始行列表。
  `trader.query_option_history_orders(start, end)` → 期权 client `/option/history-orders` → 原始行列表。

## 6. 错误处理

- 期权历史/资产在无期权会话（`enable_options=False`）时：合并类（`query_account`）自然跳过期权；
  期权历史便捷方法**抛清晰 RuntimeError**。
- 其余沿用现有信封处理（`result!="0"` → `QMFApiError`）。

## 7. 测试策略（TDD）

- mapper：`merge_option_assets` 累加正确（证券+期权 → 合并值）。
- client：`httpx.MockTransport` 打桩各端点，验证路径、`fund_account` 注入、日期参数、数组解析。
- adapter：`query_account` 启用期权时返回合并值、未启用时仅证券；历史便捷方法委托正确 client、
  无期权会话时抛错。
- 端到端：`chibi_quant --mock`（有 `opt_assets/settlements/fund_flow/opt_history_*` mock），`enable_options=True`
  跑 `query_account`（合并）+ 历史查询 exit 0。
- 现有 gateway/qmf 全绿；ruff `E,F,I,D`；提交 `--no-verify`。

## 8. 不做（YAGNI / 冻结）

- 期权组合/行权/交割/备兑相关的历史查询（`history-exercise-*`、`history-combo-orders`）→ 归 3c/3d。
- 为历史/流水造 Unified 模型（保持 `list[dict]` 原始行）。
- 把历史查询塞进 `TraderGateway` 协议（保持为附加便捷方法）。
- 不改 akquant 核心。

## 9. 待确认

1. 账户合并 M1（默认已选）。
2. `query_fund_flow` 是否需要日期范围（schema 仅 `fund_account` 必填 → 本阶段日期设为可选参数）。
3. 历史查询是否需要分页透传（本阶段先不透传分页，取默认首页 `request_num`；如需全量翻页后续再加）。
