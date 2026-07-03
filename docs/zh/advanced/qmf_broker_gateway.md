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
- 期权资产（`/option/assets`）本阶段**不并入** `query_account`（`query_account` 返回证券资金）。
- 完整示例：`examples/41_qmf_option_live_demo.py`。

## 范围与后续

组合策略（338013/14）、行权指派/交割管理、备兑划转、历史查询、可交易数量(338010)、
期权资产并入统一账户、期权独立实时 WS 订阅、Market 委托属性、完整柜台状态集与密钥下发方案
属于后续 / 待确认项。
