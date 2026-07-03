# -*- coding: utf-8 -*-
"""QMF 期权实盘对接示例（Phase 2）.

前置：chibi_quant 网关已启动（联调 `--mock`），约定 CHIBI_PASSWORD_KEY。
演示 enable_options 装配 + 一次期权下单/查询链路。
"""

import base64
import os

from akquant import DataFeed
from akquant.gateway import create_gateway_bundle
from akquant.gateway.broker_models import UnifiedOrderRequest


def main() -> None:
    """装配启用期权的 QMF gateway 并演示期权下单/查询链路."""
    password_key = os.getenv(
        "CHIBI_PASSWORD_KEY", base64.b64encode(b"0" * 32).decode("ascii")
    )
    bundle = create_gateway_bundle(
        broker="qmf",
        feed=DataFeed(),
        symbols=["10003456.SH"],
        base_url=os.getenv("QMF_BASE_URL", "http://127.0.0.1:18080"),
        ws_url=os.getenv("QMF_WS_URL", "ws://127.0.0.1:18080/api/v1/stream"),
        qmf_user_id=os.getenv("QMF_USER_ID", "u"),
        account_content=os.getenv("QMF_ACCOUNT", "8888000001"),
        password=os.getenv("QMF_PASSWORD", "pw"),
        input_content="1",
        content_type="1",
        password_key=password_key,
        enable_options=True,
    )
    trader = bundle.trader_gateway
    assert trader is not None
    trader.connect()
    print("features:", sorted(trader.get_capabilities().features))
    order_id = trader.place_order(
        UnifiedOrderRequest(
            client_order_id="opt-demo-1",
            symbol="10003456.SH",
            side="Buy",
            quantity=1,
            price=0.05,
            order_type="Limit",
            asset_type="option",
            extra={"entrust_oc": "O", "covered_flag": "0", "entrust_prop": "F0"},
        )
    )
    print("option broker_order_id:", order_id)
    print("positions:", trader.query_positions())
    trader.disconnect()


if __name__ == "__main__":
    main()
