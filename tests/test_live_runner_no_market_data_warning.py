"""trader-only broker 无行情时的启动告警.

``broker='middleware'``/``'qmf'`` 这类只有交易通道的 broker 会让
``bundle.market_gateway is None``: ``on_bar``/``on_tick`` 永不触发,
``current_tick`` 恒为 ``None``。这是**预期行为**(``run_live`` docstring 已写明),
但此前启动阶段完全静默, 用户只看到"策略没反应"而无从判断是配置问题还是没行情。

不走 ``run_live`` 端到端: 无行情时 ``duration`` 不生效, 会话会挂死。
"""

import logging
from typing import Any, cast

from akquant.gateway.protocols import GatewayBundle
from akquant.live._runner import LiveRunner


class _TraderOnlyGateway:
    """只有交易通道的网关替身."""

    def connect(self) -> None:
        """no-op."""

    def disconnect(self) -> None:
        """no-op."""

    def start(self) -> None:
        """no-op."""


def _runner(broker: str = "middleware", **attrs: Any) -> LiveRunner:
    """构造一个绕过 __init__ 的 runner, 只装 warn 判断需要的字段."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = broker
    runner.market_broker = None
    runner.trader_broker = None
    for key, value in attrs.items():
        setattr(runner, key, value)
    return runner


def _bundle(market: Any, trader: Any) -> GatewayBundle:
    return GatewayBundle(market_gateway=market, trader_gateway=trader)


def test_warns_when_trader_only_broker_has_no_market_gateway(
    caplog: Any,
) -> None:
    """无行情网关时告警, 并点明 on_bar/on_tick 不会触发."""
    runner = _runner()
    bundle = _bundle(None, _TraderOnlyGateway())
    with caplog.at_level(logging.WARNING):
        runner._warn_if_no_market_gateway(bundle)
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "middleware" in text
    assert "on_bar" in text or "on_tick" in text
    assert "market_broker" in text  # 给出补救办法


def test_no_warning_when_market_gateway_present(caplog: Any) -> None:
    """有行情网关时不告警(别对正常配置刷警告)."""
    runner = _runner()
    bundle = _bundle(cast(Any, _TraderOnlyGateway()), _TraderOnlyGateway())
    with caplog.at_level(logging.WARNING):
        runner._warn_if_no_market_gateway(bundle)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == []


def test_no_warning_when_market_broker_explicitly_configured(caplog: Any) -> None:
    """已显式配 market_broker 却仍无行情网关是另一类问题, 不由本告警负责.

    这种情况说明所配的 market_broker 自己没给出行情网关, 属于配置错误,
    应由 factory/该 broker 报错, 而不是这里含糊地劝用户"去配 market_broker"。
    """
    runner = _runner(market_broker="replay", trader_broker="middleware")
    bundle = _bundle(None, _TraderOnlyGateway())
    with caplog.at_level(logging.WARNING):
        runner._warn_if_no_market_gateway(bundle)
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "market_broker" not in text
