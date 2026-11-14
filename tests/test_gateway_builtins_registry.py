from typing import Any

import pytest
from akquant import DataFeed
from akquant.gateway import create_gateway_bundle, list_registered_brokers


def test_builtins_registered_on_import() -> None:
    """Importing akquant.gateway registers all built-in brokers."""
    names = set(list_registered_brokers())
    assert {"ctp", "replay"} <= names
    assert not {"miniqmt", "ptrade"} & names


def test_unknown_broker_lists_builtins() -> None:
    """Unknown broker error names the registered brokers."""
    with pytest.raises(ValueError) as exc:
        create_gateway_bundle(broker="nope", feed=DataFeed(), symbols=[])
    msg = str(exc.value)
    assert "ctp" in msg


def test_ctp_requires_md_front_unchanged() -> None:
    """Built-in ctp keeps its md_front requirement (behavior parity)."""
    with pytest.raises(ValueError):
        create_gateway_bundle(broker="ctp", feed=DataFeed(), symbols=["x"])


_CTP_TRADER = {
    "td_front": "tcp://td",
    "broker_id": "9999",
    "user_id": "u",
    "password": "p",
    "app_id": "app",
    "auth_code": "code",
}


def test_ctp_md_front_error_points_to_gateway_options() -> None:
    """缺 md_front 的报错要指明它该放在 gateway_options 里."""
    with pytest.raises(ValueError, match="gateway_options 缺少 md_front"):
        create_gateway_bundle(broker="ctp", feed=DataFeed(), symbols=["x"])


@pytest.mark.parametrize("dropped", sorted(_CTP_TRADER))
def test_ctp_partial_trader_options_are_rejected(dropped: str) -> None:
    """交易通道六项缺任何一项都报错, 不补仿真默认值、不悄悄跳过交易通道."""
    options: Any = {k: v for k, v in _CTP_TRADER.items() if k != dropped}
    with pytest.raises(ValueError, match="交易通道参数不完整") as exc:
        create_gateway_bundle(
            broker="ctp",
            feed=DataFeed(),
            symbols=["x"],
            md_front="tcp://md",
            **options,
        )
    assert dropped in str(exc.value)
