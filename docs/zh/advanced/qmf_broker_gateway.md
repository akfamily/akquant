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

## 范围与后续

Phase 1 覆盖证券（股票）交易。期权（`/api/v1/option/*`、开平/备兑/行权、`asset_prop="B"`
独立会话）、Market 委托属性、完整柜台状态集与密钥下发方案属于 Phase 2 / 待确认项。
