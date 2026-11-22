"""实盘消息提醒: 把关键日志推送到企业微信群机器人.

提醒挂在 ``akquant`` 日志上, 推送以下记录:

- 成交: 订单审计 ``order_fill``;
- 拒单: 订单审计 ``order_reject``(本地风控、策略限额、柜台拒单);
- 报单异常: 订单审计 ``order_submit_unknown``(报单是否到达柜台未知)、
  ``order_cancel_failed``;
- ERROR 及以上级别的日志(交易前置断线、登录失败、会话因异常中止等);
- 带 ``alert`` 字段的记录(``extra={"alert": "<类别>"}``), 如行情断线与重连、
  策略回调异常; 类别即消息标题。

发送在后台线程: 每 ``batch_interval_sec`` 秒把积攒的提醒合并成一条, 按企业微信
机器人每分钟 20 条的限制发送, 超出时继续积攒、不丢弃。发送失败的原因写入
``akquant.alerts`` 日志(该日志本身不推送)。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..log import mask_sensitive_text, render_audit_message

ALERT_LOGGER_NAME = "akquant.alerts"
#: 企业微信群机器人每分钟最多 20 条消息
WECOM_MESSAGES_PER_MINUTE = 20
#: 企业微信文本消息内容上限 2048 字节, 留出余量
WECOM_TEXT_MAX_BYTES = 2000
_AUDIT_CATEGORIES = {
    "order_fill": "成交",
    "order_reject": "拒单",
    "order_submit_unknown": "报单异常",
    "order_cancel_failed": "报单异常",
}

logger = logging.getLogger(ALERT_LOGGER_NAME)


@dataclass
class AlertConfig:
    """实盘消息提醒配置.

    :param wecom_webhook: 企业微信群机器人的 webhook 地址
    :param name: 消息前缀, 同一个群里有多个实盘进程时用来区分
    :param trades: 是否推送成交
    :param rejects: 是否推送拒单
    :param batch_interval_sec: 合并发送的间隔(秒)
    """

    wecom_webhook: str
    name: str = ""
    trades: bool = True
    rejects: bool = True
    batch_interval_sec: float = 2.0

    def __post_init__(self) -> None:
        """核对 webhook 地址与合并间隔."""
        if not str(self.wecom_webhook).startswith("https://"):
            raise ValueError(
                "AlertConfig.wecom_webhook 须为企业微信机器人的 https 地址, "
                f"实际: {self.wecom_webhook!r}"
            )
        if self.batch_interval_sec <= 0:
            raise ValueError(
                "AlertConfig.batch_interval_sec 须为正数, "
                f"实际: {self.batch_interval_sec}"
            )


def _post_wecom(url: str, text: str) -> None:
    """向企业微信机器人发一条文本消息; 失败时抛出带原因的 RuntimeError."""
    body = json.dumps({"msgtype": "text", "text": {"content": text}}).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        reply = json.loads(response.read().decode("utf-8"))
    if reply.get("errcode") != 0:
        raise RuntimeError(
            f"企业微信返回 errcode={reply.get('errcode')} {reply.get('errmsg')}"
        )


class WeComAlertHandler(logging.Handler):
    """把需要提醒的日志记录合并后推送到企业微信群机器人."""

    def __init__(
        self,
        config: AlertConfig,
        post: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """创建处理器并启动后台发送线程; ``post`` 默认发往企业微信."""
        super().__init__(level=logging.DEBUG)
        self._config = config
        self._post = post if post is not None else _post_wecom
        self._pending: list[str] = []
        self._cond = threading.Condition()
        self._closing = False
        self._sent_at: deque[float] = deque()
        self._thread = threading.Thread(
            target=self._run, name="akquant-alerts", daemon=True
        )
        self._thread.start()

    def category(self, record: logging.LogRecord) -> Optional[str]:
        """记录对应的提醒类别; 不需要提醒时为 None."""
        if record.name == ALERT_LOGGER_NAME:
            return None
        marked = getattr(record, "alert", None)
        if marked:
            return str(marked)
        audit = _AUDIT_CATEGORIES.get(str(getattr(record, "event", "") or ""))
        if audit is not None and getattr(record, "phase", None) == "gateway":
            if audit == "成交" and not self._config.trades:
                return None
            if audit == "拒单" and not self._config.rejects:
                return None
            return audit
        if record.levelno >= logging.CRITICAL:
            return "严重错误"
        if record.levelno >= logging.ERROR:
            return "错误"
        return None

    def emit(self, record: logging.LogRecord) -> None:
        """把需要提醒的记录放入待发队列."""
        try:
            category = self.category(record)
            if category is None:
                return
            event = getattr(record, "event", None)
            message = (
                render_audit_message(event, record.__dict__, "zh")
                if event in _AUDIT_CATEGORIES
                else record.getMessage()
            )
            if record.exc_info and record.exc_info[1] is not None:
                exc = record.exc_info[1]
                message += f"\n{type(exc).__name__}: {exc}"
            stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
            self.push(f"[{category}] {stamp} {mask_sensitive_text(message)}")
        except Exception:  # noqa: BLE001 提醒旁路: 失败不影响交易主流程
            self.handleError(record)

    def push(self, text: str) -> None:
        """直接放入一条提醒(会话启动、结束等不经日志的消息)."""
        with self._cond:
            self._pending.append(text)
            self._cond.notify()

    def close(self) -> None:
        """发完积攒的提醒后停止后台线程(最多等 15 秒)."""
        with self._cond:
            self._closing = True
            self._cond.notify()
        self._thread.join(timeout=15)
        super().close()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._closing:
                    self._cond.wait()
                if not self._closing:
                    self._cond.wait(timeout=self._config.batch_interval_sec)
                batch, self._pending = self._pending, []
                closing = self._closing
            for text in self._split(batch):
                self._send(text, closing)
            if closing:
                with self._cond:
                    if not self._pending:
                        return

    def _split(self, lines: list[str]) -> list[str]:
        """按企业微信文本长度上限把提醒拼成若干条消息."""
        prefix = f"【{self._config.name}】\n" if self._config.name else ""
        limit = WECOM_TEXT_MAX_BYTES - len(prefix.encode("utf-8"))
        messages: list[str] = []
        chunk: list[str] = []
        size = 0
        for line in lines:
            data = line.encode("utf-8")
            if len(data) > limit:
                line = data[: limit - 20].decode("utf-8", errors="ignore") + "…(截断)"
                data = line.encode("utf-8")
            if chunk and size + 1 + len(data) > limit:
                messages.append(prefix + "\n".join(chunk))
                chunk, size = [], 0
            size += len(data) + (1 if chunk else 0)
            chunk.append(line)
        if chunk:
            messages.append(prefix + "\n".join(chunk))
        return messages

    def _send(self, text: str, closing: bool) -> None:
        now = time.monotonic()
        while self._sent_at and now - self._sent_at[0] >= 60:
            self._sent_at.popleft()
        if len(self._sent_at) >= WECOM_MESSAGES_PER_MINUTE:
            wait = 60 - (now - self._sent_at[0])
            if closing and wait > 10:
                logger.warning(
                    "企业微信每分钟 %s 条已用完, 收尾时放弃发送: %s",
                    WECOM_MESSAGES_PER_MINUTE,
                    text,
                )
                return
            time.sleep(max(wait, 0.0))
            self._sent_at.popleft()
        self._sent_at.append(time.monotonic())
        try:
            self._post(self._config.wecom_webhook, text)
        except Exception as exc:  # noqa: BLE001 发送失败只记日志
            logger.warning("企业微信提醒发送失败(%s): %s", exc, text)


def install_alerts(config: AlertConfig, title: str) -> WeComAlertHandler:
    """在 ``akquant`` 日志上挂提醒处理器, 并推送会话启动消息."""
    handler = WeComAlertHandler(config)
    logging.getLogger("akquant").addHandler(handler)
    handler.push(f"[会话] {time.strftime('%H:%M:%S')} 启动: {title}")
    return handler


def uninstall_alerts(handler: Any, stop_reason: str) -> None:
    """推送会话结束消息, 发完积攒的提醒后摘掉处理器."""
    handler.push(f"[会话] {time.strftime('%H:%M:%S')} 结束: {stop_reason}")
    logging.getLogger("akquant").removeHandler(handler)
    handler.close()
