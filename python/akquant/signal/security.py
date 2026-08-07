"""网络信号入口的鉴权与防重放.

这是**能触发真实下单**的网络端点的守门逻辑, 故设计成硬约束而非可选项:
缺 token 直接启动失败, 不降级为警告。对标 Freqtrade 的取舍(其
``force_entry_enable`` 默认关闭、只听 localhost、官方明确警告勿暴露公网)。

签名方案: ``HMAC-SHA256(secret, f"{timestamp}.{body}")``, 十六进制小写。
时间戳(秒)必须落在 ``±window`` 内, 配合 ``signal_id`` 幂等去重构成三重防护。
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass

DEFAULT_WINDOW_SEC = 30


class AuthError(Exception):
    """鉴权失败(调用方应回 401, 且不得泄漏具体原因给外部)."""


def sign(secret: str, body: bytes, timestamp: int) -> str:
    """按约定算签名, 供客户端与测试复用."""
    payload = f"{timestamp}.".encode() + body
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class TokenAuth:
    """Bearer token + 可选 HMAC 签名的校验器.

    ``secret`` 为 None 时只校验 token(适合本机开发); 提供 secret 则额外要求
    签名与时间戳, 用于跨主机传输。
    """

    token: str
    secret: str | None = None
    window_sec: int = DEFAULT_WINDOW_SEC

    def __post_init__(self) -> None:
        """拒绝空/非字符串 token —— 硬约束, 不允许无鉴权端点.

        必须显式挡住 ``None``: 若走 ``str(None)`` 会得到字面量 ``"None"``, 那就成了
        一个"看起来有鉴权"的端点 —— 从缺失的环境变量读 token 正是这个场景。
        """
        if not isinstance(self.token, str) or not self.token.strip():
            raise ValueError(
                "token 必须是非空字符串: 该端点可触发真实下单, 必须鉴权"
                f"(收到 {type(self.token).__name__})。"
                "如需本机免鉴权调试, 请改用 QueueSignalSource"
            )

    def verify(
        self,
        *,
        authorization: str | None,
        body: bytes,
        timestamp: str | None = None,
        signature: str | None = None,
        now: int | None = None,
    ) -> None:
        """校验一次请求; 失败抛 :class:`AuthError`.

        用 ``compare_digest`` 做常量时间比较, 避免按字节泄漏 token/签名。
        """
        supplied = (authorization or "").strip()
        prefix = "bearer "
        if supplied.lower().startswith(prefix):
            supplied = supplied[len(prefix) :].strip()
        if not hmac.compare_digest(supplied, self.token):
            raise AuthError("token mismatch")

        if self.secret is None:
            return

        if not timestamp or not signature:
            raise AuthError("missing signature headers")
        try:
            sent_at = int(timestamp)
        except (TypeError, ValueError) as exc:
            raise AuthError("bad timestamp") from exc
        current = int(time.time()) if now is None else now
        if abs(current - sent_at) > self.window_sec:
            raise AuthError("timestamp outside window")
        if not hmac.compare_digest(signature, sign(self.secret, body, sent_at)):
            raise AuthError("signature mismatch")
