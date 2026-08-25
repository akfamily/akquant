"""broker_recovery.handle_error 的默认模式可见性: compatible 也必须留痕.

背景: 默认 recovery_mode 是 ``compatible``, 而 ``handle_error`` 此前第一行就是
``if recovery_mode != "strict": return`` —— 心跳失败/重连失败/sync 报错在默认
配置下完全不打日志、不通知策略、不发事件, 柜台已经掉线而用户毫无感知。

修法把 strict 判断下移到 error_key 去重之后, 中间插入一条两种模式都打的
WARNING; strict 专属的 ``_notify_strategy_error`` + observer 事件保持原位不动
—— 这次只补可见性, 不改 compatible 模式"不中断交易"的容错语义。
"""

import logging
from typing import Any

import pytest
from akquant.gateway.broker_recovery import BrokerRecovery

LOGGER_NAME = "akquant.gateway.live"


class _Recorder:
    """记录 notify_strategy_error / on_broker_event 是否被调用及调用内容."""

    def __init__(self) -> None:
        """初始化空的调用记录."""
        self.notified: list[tuple[Any, Exception, str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def notify_strategy_error(
        self, strategy: Any, error: Exception, source: str, payload: Any
    ) -> None:
        """记录一次策略错误通知."""
        self.notified.append((strategy, error, source, payload))

    def on_broker_event(self, event: dict[str, Any]) -> None:
        """记录一次 observer 事件."""
        self.events.append(event)


def _recovery(mode: str, recorder: _Recorder) -> BrokerRecovery:
    last_error_key = {"value": ""}
    return BrokerRecovery(
        get_trader_gateway=lambda: None,
        queue_broker_event=lambda name, payload: None,
        notify_strategy_error=recorder.notify_strategy_error,
        get_on_broker_event=lambda: recorder.on_broker_event,
        get_recovery_mode=lambda: mode,
        get_last_error_key=lambda: last_error_key["value"],
        set_last_error_key=lambda key: last_error_key.__setitem__("value", key),
    )


def test_compatible_mode_logs_warning_with_source_and_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """默认 compatible 模式下也要打 WARNING, 文本含 source 与异常信息."""
    recorder = _Recorder()
    recovery = _recovery("compatible", recorder)
    error = ConnectionError("heartbeat timed out")

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        recovery.handle_error(None, "broker_recovery.heartbeat", error, {})

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "broker_recovery.heartbeat" in message
    assert "heartbeat timed out" in message


def test_compatible_mode_does_not_notify_strategy_or_emit_event() -> None:
    """Compatible 模式的容错语义不变: 不通知策略, 不发 observer 事件."""
    recorder = _Recorder()
    recovery = _recovery("compatible", recorder)

    recovery.handle_error(
        object(), "broker_recovery.sync_open_orders", RuntimeError("boom"), {}
    )

    assert recorder.notified == []
    assert recorder.events == []


def test_same_error_key_only_logs_once(caplog: pytest.LogCaptureFixture) -> None:
    """同一 error_key 重复上报只打一次日志(复用既有去重, 不新建一套)."""
    recorder = _Recorder()
    recovery = _recovery("compatible", recorder)
    error = RuntimeError("still down")

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        recovery.handle_error(None, "broker_recovery.heartbeat", error, {})
        recovery.handle_error(None, "broker_recovery.heartbeat", error, {})

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_strict_mode_still_notifies_and_emits_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Strict 模式原有行为不回归: 通知策略 + observer 事件仍然发生, 且照样打日志."""
    recorder = _Recorder()
    recovery = _recovery("strict", recorder)
    strategy = object()
    error = RuntimeError("query_account failed")

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        recovery.handle_error(strategy, "broker_recovery.query_account", error, {})

    assert len(recorder.notified) == 1
    assert recorder.notified[0][:3] == (
        strategy,
        error,
        "broker_recovery.query_account",
    )
    assert len(recorder.events) == 1
    assert recorder.events[0]["event_type"] == "recovery_error"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
