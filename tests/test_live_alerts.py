"""实盘消息提醒: 按日志记录分类、合并发送、长度切分、发送失败记日志、会话启停."""

import logging
import sys
from types import SimpleNamespace
from typing import Any, List

import pytest
from akquant.gateway import ctp_native
from akquant.gateway.ctp_native import CTPTraderGateway
from akquant.gateway.order_audit import record_reject
from akquant.live import _runner, alerts
from akquant.live._runner import LiveRunner
from akquant.live.alerts import (
    ALERT_LOGGER_NAME,
    WECOM_TEXT_MAX_BYTES,
    AlertConfig,
    WeComAlertHandler,
    install_alerts,
    uninstall_alerts,
)
from akquant.log import build_order_audit_extra

WEBHOOK = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test"


def _handler(sent: List[str], **kwargs: Any) -> WeComAlertHandler:
    config = AlertConfig(wecom_webhook=WEBHOOK, batch_interval_sec=0.01, **kwargs)
    return WeComAlertHandler(config, post=lambda url, text: sent.append(text))


def _record(
    level: int = logging.INFO, msg: str = "m", name: str = "akquant.x", **extra: Any
) -> logging.LogRecord:
    record = logging.LogRecord(name, level, __file__, 1, msg, None, None)
    record.__dict__.update(extra)
    return record


def _audit(event: str, **fields: Any) -> logging.LogRecord:
    return _record(
        logging.INFO,
        event,
        "akquant.audit.order",
        **build_order_audit_extra(event=event, **fields),
    )


def test_category_selects_trades_rejects_errors_and_marked_records() -> None:
    """成交、拒单、报单异常、ERROR 以上与带 alert 标记的记录推送, 其余不推."""
    handler = _handler([])
    try:
        assert handler.category(_audit("order_fill")) == "成交"
        assert handler.category(_audit("order_reject")) == "拒单"
        assert handler.category(_audit("order_submit_unknown")) == "报单异常"
        assert handler.category(_audit("order_submit")) is None
        assert handler.category(_record(logging.ERROR)) == "错误"
        assert handler.category(_record(logging.CRITICAL)) == "严重错误"
        assert handler.category(_record(logging.WARNING, alert="连接")) == "连接"
        assert handler.category(_record(logging.WARNING)) is None
        # 提醒自身的日志(发送失败)不再推送
        assert handler.category(_record(logging.ERROR, name=ALERT_LOGGER_NAME)) is None
    finally:
        handler.close()

    quiet = _handler([], trades=False, rejects=False)
    try:
        assert quiet.category(_audit("order_fill")) is None
        assert quiet.category(_audit("order_reject")) is None
    finally:
        quiet.close()


def test_alerts_are_merged_and_prefixed_with_name() -> None:
    """同一批提醒合并为一条, 带进程名前缀; 审计记录按中文模板渲染, 异常附类型与信息."""
    sent: List[str] = []
    handler = _handler(sent, name="期权实盘")
    handler.emit(
        _audit(
            "order_reject",
            symbol="m2701-C-3000",
            client_order_id="c1",
            reason="超过单日开仓次数",
        )
    )
    try:
        raise ValueError("boom")
    except ValueError:
        error = _record(logging.WARNING, "回调失败", alert="策略异常")
        error.exc_info = sys.exc_info()
    handler.emit(error)
    handler.close()

    assert len(sent) == 1
    lines = sent[0].split("\n")
    assert lines[0] == "【期权实盘】"
    assert lines[1].startswith("[拒单] ")
    assert lines[1].endswith("拒单 m2701-C-3000 [c1] 原因: 超过单日开仓次数")
    assert lines[2].startswith("[策略异常] ") and lines[2].endswith("回调失败")
    assert lines[3] == "ValueError: boom"


def test_long_batches_split_under_wecom_limit() -> None:
    """超过企业微信 2048 字节上限时拆成多条, 单条过长的提醒截断."""
    sent: List[str] = []
    handler = WeComAlertHandler(
        AlertConfig(wecom_webhook=WEBHOOK), post=lambda u, t: sent.append(t)
    )
    lines = [f"提醒{i} " + "x" * 300 for i in range(10)] + ["长" * 2000]
    messages = handler._split(lines)
    handler.close()

    assert all(len(m.encode("utf-8")) <= WECOM_TEXT_MAX_BYTES for m in messages)
    assert "\n".join(messages).count("提醒") == 10
    assert messages[-1].endswith("…(截断)")


def test_send_failure_is_logged_and_not_pushed_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """发送失败写入 akquant.alerts 日志, 写明原因; 该日志不再进入提醒队列."""
    calls: List[str] = []

    def post(url: str, text: str) -> None:
        calls.append(text)
        raise RuntimeError("企业微信返回 errcode=93000 invalid webhook url")

    handler = WeComAlertHandler(
        AlertConfig(wecom_webhook=WEBHOOK, batch_interval_sec=0.01), post=post
    )
    logging.getLogger("akquant").addHandler(handler)
    try:
        with caplog.at_level(logging.WARNING, logger=ALERT_LOGGER_NAME):
            handler.push("[错误] 12:00:00 x")
            handler.close()
    finally:
        logging.getLogger("akquant").removeHandler(handler)
    assert calls == ["[错误] 12:00:00 x"]
    assert any("errcode=93000" in r.getMessage() for r in caplog.records)


def test_install_pushes_session_start_rejects_and_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """挂上后推送会话启动、经日志的拒单, 摘掉时推送结束原因并发完积攒的提醒."""
    sent: List[str] = []
    monkeypatch.setattr(alerts, "_post_wecom", lambda url, text: sent.append(text))
    handler = install_alerts(
        AlertConfig(wecom_webhook=WEBHOOK, batch_interval_sec=0.01), "Demo"
    )
    record_reject(
        strategy_id="s", symbol="rb2701", client_order_id="c9", reason="风控熔断"
    )
    uninstall_alerts(handler, "Manual Stop")

    text = "\n".join(sent)
    assert "[会话]" in text and "启动: Demo" in text
    assert "拒单 rb2701 [c9] 原因: 风控熔断" in text
    assert text.rstrip().endswith("结束: Manual Stop")
    assert handler not in logging.getLogger("akquant").handlers


def test_runner_reports_abort_reason_to_alerts(monkeypatch: pytest.MonkeyPatch) -> None:
    """会话抛出异常时, 结束提醒写明异常, 异常照常抛出."""
    ended: List[str] = []
    monkeypatch.setattr(_runner, "install_alerts", lambda config, title: title)
    monkeypatch.setattr(
        _runner, "uninstall_alerts", lambda handler, reason: ended.append(reason)
    )
    runner: Any = LiveRunner.__new__(LiveRunner)
    runner.alerts = AlertConfig(wecom_webhook=WEBHOOK)
    runner.strategy_cls = SimpleNamespace
    runner.broker, runner.trading_mode = "ctp", "broker_live"

    def boom(*args: Any) -> None:
        raise RuntimeError("网关配置缺少 front_url")

    # 会话首个步骤即失败(配置引擎时区)
    monkeypatch.setattr(runner, "_session_timezone", boom, raising=False)
    runner.engine = SimpleNamespace(set_timezone_name=lambda name: None)
    with pytest.raises(RuntimeError):
        runner.run()
    assert ended == ["异常中止: RuntimeError: 网关配置缺少 front_url"]


def test_alert_config_rejects_bad_webhook() -> None:
    """企业微信地址不是 https 时报出原值."""
    with pytest.raises(ValueError, match="须为企业微信机器人的 https 地址"):
        AlertConfig(wecom_webhook="qyapi.weixin.qq.com")


def test_ctp_reconnect_after_disconnect_is_marked(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """交易前置断线带提醒标记; 断线后重连成功也带标记, 首次连上不带."""
    monkeypatch.setattr(
        ctp_native.tdapi,
        "CThostFtdcReqAuthenticateField",
        lambda: SimpleNamespace(),
        raising=False,
    )
    gateway = CTPTraderGateway.__new__(CTPTraderGateway)
    gateway.front_url = "tcp://x"
    gateway.api = SimpleNamespace(ReqAuthenticate=lambda request, req_id: 0)
    gateway.req_id, gateway.broker_id, gateway.user_id = 0, "9999", "u"
    gateway.app_id, gateway.auth_code = "a", "0"
    with caplog.at_level(logging.INFO, logger="akquant"):
        gateway.OnFrontConnected()
        gateway.OnFrontDisconnected(4097)
        gateway.OnFrontConnected()
    marks = [
        (r.getMessage().split(":")[0], getattr(r, "alert", None))
        for r in caplog.records
        if "front" in r.getMessage()
    ]
    assert marks == [
        ("CTP trader front connected", None),
        ("CTP trader front disconnected", "连接"),
        ("CTP trader front connected", "连接"),
    ]
