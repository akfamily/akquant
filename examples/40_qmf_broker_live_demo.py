# -*- coding: utf-8 -*-
"""QMF broker 实盘对接示例（Phase 1 证券）.

前置：chibi_quant 网关已启动（联调可用 `--mock`），并已约定 CHIBI_PASSWORD_KEY。
本示例仅演示 gateway 装配与一次下单/查询链路，不含策略主循环。
"""

import base64
import os

from akquant import DataFeed
from akquant.gateway import create_gateway_bundle
from akquant.gateway.broker_models import UnifiedOrderRequest


def main() -> None:
    """装配 QMF gateway 并演示登录/下单/查询链路."""
    password_key = os.getenv(
        "CHIBI_PASSWORD_KEY", base64.b64encode(b"0" * 32).decode("ascii")
    )
    bundle = create_gateway_bundle(
        broker="qmf",
        feed=DataFeed(),
        symbols=["600000.SH"],
        base_url=os.getenv("QMF_BASE_URL", "http://127.0.0.1:18080"),
        ws_url=os.getenv("QMF_WS_URL", "ws://127.0.0.1:18080/api/v1/stream"),
        qmf_user_id=os.getenv("QMF_USER_ID", "u"),
        account_content=os.getenv("QMF_ACCOUNT", "8888000001"),
        password=os.getenv("QMF_PASSWORD", "pw"),
        input_content="1",
        content_type="1",
        password_key=password_key,
    )
    trader = bundle.trader_gateway
    assert trader is not None
    trader.connect()
    print("capabilities:", trader.get_capabilities().broker_name)
    print("account:", trader.query_account())
    order_id = trader.place_order(
        UnifiedOrderRequest(
            client_order_id="demo-1",
            symbol="600000.SH",
            side="Buy",
            quantity=100,
            price=10.5,
            order_type="Limit",
        )
    )
    print("broker_order_id:", order_id)
    print("positions:", trader.query_positions())
    trader.disconnect()


if __name__ == "__main__":
    main()
