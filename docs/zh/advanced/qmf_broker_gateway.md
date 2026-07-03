# QMF（期魔方）Broker 网关

QMF broker 通过 HTTP/WS 对接部署在服务器侧的 chibi_quant 前置机网关（恒生 T2 柜台），
使 akquant 策略能对期魔方证券账户实盘下单/撤单/查询并接收委托与成交回报。

- 仅实现 `TraderGateway`（交易）；**行情继续走 akquant 现有 feed**，本 broker 不提供行情。
- 依赖为可选组：`pip install 'akquant[qmf]'`（httpx / websocket-client / cryptography）。
- 能力矩阵：Phase 1 证券（`position_effect=False`，无开平/融券概念）。

## 快速运行

前置：chibi_quant 网关已启动（联调可用 `--mock`），客户端与网关约定同一 `CHIBI_PASSWORD_KEY`。

```python
from akquant import DataFeed
from akquant.gateway import create_gateway_bundle
from akquant.gateway.broker_models import UnifiedOrderRequest

bundle = create_gateway_bundle(
    broker="qmf",
    feed=DataFeed(),
    symbols=["600000.SH"],
    base_url="http://127.0.0.1:18080",
    ws_url="ws://127.0.0.1:18080/api/v1/stream",
    qmf_user_id="u",
    account_content="8888000001",
    password="明文交易密码",   # 客户端内部以 AES-256-GCM 加密后再上送
    input_content="1",
    content_type="1",
    password_key="<base64(32B) 共享密钥>",
)
trader = bundle.trader_gateway
trader.connect()               # 登录，获取 gateway token
trader.place_order(
    UnifiedOrderRequest(
        client_order_id="demo-1", symbol="600000.SH", side="Buy",
        quantity=100, price=10.5, order_type="Limit",
    )
)
print(trader.query_account())
print(trader.query_positions())
trader.disconnect()
```

完整示例：`examples/40_qmf_broker_live_demo.py`。

## 说明

- 下单返回柜台 `entrust_no` 作为 `broker_order_id`；适配器内部维护
  `entrust_no -> client_order_id` 反查表，用于把推送/查询结果映射回策略订单。
- 委托与成交回报经 `WebSocket /api/v1/stream` 推送；网关不缓存推送历史，断线重连后由
  `sync_open_orders` / `sync_today_trades`（HTTP 查询）补齐。
- 注册方式与其它内置 broker 一致（`create_gateway_bundle(broker="qmf", ...)`），
  另见 [自定义 Broker 注册](custom_broker_registry.md) 与
  [Broker 能力矩阵](broker_capability_matrix.md)。

## 期权（Phase 2）

同一个 `broker="qmf"` 通过**双会话**支持期权：证券会话（`asset_prop="0"`）与期权会话
（`asset_prop="B"`）。装配时传 `enable_options=True` 即额外登录期权会话并声明期权能力
（`features` 含 `"options"`、`broker_extra_fields` 含 `entrust_oc`/`covered_flag`/`entrust_prop`）。
证券路径不受影响（默认 `enable_options=False`）。

期权下单用 `asset_type="option"`，期权专属语义经 `extra` 传入：

```python
bundle = create_gateway_bundle(
    broker="qmf", feed=DataFeed(), symbols=["10003456.SH"],
    base_url="http://127.0.0.1:18080", ws_url="ws://127.0.0.1:18080/api/v1/stream",
    qmf_user_id="u", account_content="8888000001", password="明文交易密码",
    input_content="1", content_type="1", password_key="<base64(32B)>",
    enable_options=True,
)
trader = bundle.trader_gateway
trader.connect()
trader.place_order(
    UnifiedOrderRequest(
        client_order_id="opt-1", symbol="10003456.SH", side="Buy",
        quantity=1, price=0.05, order_type="Limit", asset_type="option",
        extra={"entrust_oc": "O", "covered_flag": "0", "entrust_prop": "F0"},
    )
)
```

- `entrust_oc`：`O`=开仓 / `C`=平仓 / `X`=行权（必填）；`covered_flag`：`1`=备兑 / `0`=非（默认 `0`）。
- 期权路由到 `/api/v1/option/*`；`query_positions`/`sync_*` 合并证券与期权。
- 启用期权后 `query_account` 合并证券与期权资产（`/option/assets`），详见下节。
- 完整示例：`examples/41_qmf_option_live_demo.py`。

## 只读查询（Phase 3a）

- 启用期权后（`enable_options=True`）`query_account()` 返回**合并**账户
  （证券资金 + 期权资产汇总为 `equity`/`cash`/`available_cash`）；未启用期权时仍只返回证券资金。
- `trader.query_settlements(start_date, end_date, stock_type=None)` /
  `trader.query_fund_flow(start_date=None, end_date=None)` 查询证券交割单 / 资金流水。
- `trader.query_option_history_orders/trades/settlements(start_date, end_date)`
  查询期权历史委托/成交/交割单，需先 `enable_options=True` 建立期权会话，
  否则抛出 `RuntimeError`。
- 以上方法均非 `TraderGateway` 协议方法，返回柜台**原始行** `list[dict]`（不做 Unified 建模）。

## 实盘就绪（broker_ready）

`trading_mode="broker_live"` 下 `LiveRunner` 会先 `connect()`（登录）再 `start()`
（起 WebSocket 推送），随后轮询 `trader_gateway.heartbeat()` 直至就绪或超时
（`broker_ready_timeout`，默认 10s）；就绪状态写回策略上下文的 `broker_ready` 属性。

- 策略应以 `if ctx.broker_ready:` 门首单，而不是 `hasattr(ctx, "submit_order")`——
  `submit_order` 在 `broker_live` 模式下总会被注入，但 broker 未就绪前调用它会直接
  抛出清晰的 `RuntimeError`（`broker 尚未就绪，请在 broker_ready=True
  (on_broker_connected 之后)再下单`）。
- 就绪达成时，`LiveRunner` 会对策略与各 slot 触发 `on_broker_connected(ctx)`
  （策略方法与 `LiveRunner(on_broker_connected=...)` 函数式回调均支持）。
- `paper`/其它非 `broker_live` 模式下 `broker_ready` 默认即为 `True`，不受该守卫影响。
- 拒单与错误分别通过 `on_reject(ctx, order)` / `on_error(ctx, error)` 回调上报，
  不要依赖 `on_order` 里再判断状态字符串。
- 就绪判定基于登录（`heartbeat`）。QMF 登录完成即可下单/查询（HTTP），但推送 WS 可能
  略晚建立；就绪到 WS 建立之间的委托/成交回报由断线补齐（`sync_open_orders`/
  `sync_today_trades` 的 HTTP 补齐）兜底，不会丢。

完整示例：`examples/39_live_broker_submit_order_demo.py`。

## 范围与后续

组合策略（338013/14）、行权指派/交割管理、备兑划转、可交易数量(338010)、
组合/行权/交割相关历史查询、历史查询分页透传、期权独立实时 WS 订阅、Market 委托属性、
完整柜台状态集与密钥下发方案属于后续 / 待确认项。
