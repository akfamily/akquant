"""HTTP / Redis 信号源与安全基线.

设计依据: docs/zh/meta/signal-ingestion-rfc.md 第 4.3/4.4 节。

**HTTP 部分起真实服务、发真实 HTTP 请求**(httpx), 不是 mock —— 这个端点能触发真实
下单, 安全逻辑必须被真的测到。这也是它用标准库 ``http.server`` 而非 FastAPI 的原因:
可选依赖在 CI 缺失时只能靠 mock 覆盖。

Redis 部分注入 fake client 覆盖消费循环(XREADGROUP → dispatch → XACK)。真实 Redis
连接路径**未验证**(环境未装 redis 包), 已在 RFC 中标注。
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Tuple

import httpx
import pytest
from akquant.signal import (
    AuthError,
    HttpSignalSource,
    RedisSignalSource,
    Signal,
    SignalDispatcher,
    SignalStatus,
    TokenAuth,
    sign,
)

TOKEN = "test-token-abc"
SECRET = "test-secret-xyz"
SYMBOL = "SIGP3_A"


class _RecordingSink:
    """记录收到的信号, 返回递增订单 id."""

    mode = "paper"

    def __init__(self) -> None:
        """初始化记录容器."""
        self.submitted: List[Signal] = []

    def submit(self, signal: Signal) -> str:
        """记录并返回订单 id."""
        self.submitted.append(signal)
        return f"order-{len(self.submitted)}"


def _payload(signal_id: str = "s-1", **overrides: Any) -> Dict[str, Any]:
    """造一份合法的信号 JSON 负载."""
    body = {
        "signal_id": signal_id,
        "symbol": SYMBOL,
        "action": "buy",
        "quantity": 100.0,
        "price": 10.0,
    }
    body.update(overrides)
    return body


def _serve(**kwargs: Any) -> Tuple[HttpSignalSource, _RecordingSink, str]:
    """起一个真实的 HTTP 信号源, 返回 (source, sink, url)."""
    sink = _RecordingSink()
    source = HttpSignalSource(token=TOKEN, port=0, **kwargs)
    dispatcher = SignalDispatcher(sink, on_result=source.on_result)
    source.bind(dispatcher.dispatch)
    source.start()
    return source, sink, f"http://127.0.0.1:{source.bound_port}/signal"


# --------------------------------------------------------------------------
# 安全基线(纯单元, 不起服务)
# --------------------------------------------------------------------------


def test_empty_token_is_rejected_at_construction() -> None:
    """空 token 必须构造即失败 —— 不允许无鉴权的下单端点."""
    for bad in ("", "   ", None):
        with pytest.raises(ValueError, match="token"):
            TokenAuth(token=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="token"):
        HttpSignalSource(token="")


def test_remote_bind_requires_explicit_opt_in() -> None:
    """绑非本机地址必须显式 allow_remote, 默认拒绝."""
    with pytest.raises(ValueError, match="allow_remote"):
        HttpSignalSource(token=TOKEN, host="0.0.0.0", port=0)
    # 显式开启则允许(不真正 start, 避免在 CI 上监听外部地址)
    source = HttpSignalSource(token=TOKEN, host="0.0.0.0", port=0, allow_remote=True)
    assert source is not None


def test_token_verification_accepts_bearer_and_raw() -> None:
    """Authorization 支持 ``Bearer x`` 与裸 token 两种写法."""
    auth = TokenAuth(token=TOKEN)
    auth.verify(authorization=f"Bearer {TOKEN}", body=b"")
    auth.verify(authorization=TOKEN, body=b"")
    with pytest.raises(AuthError):
        auth.verify(authorization="Bearer wrong", body=b"")
    with pytest.raises(AuthError):
        auth.verify(authorization=None, body=b"")


def test_hmac_signature_and_replay_window() -> None:
    """开启 secret 后: 签名必须匹配, 且时间戳须落在窗口内."""
    auth = TokenAuth(token=TOKEN, secret=SECRET, window_sec=30)
    body = b'{"a":1}'
    now = 1_700_000_000

    auth.verify(
        authorization=TOKEN,
        body=body,
        timestamp=str(now),
        signature=sign(SECRET, body, now),
        now=now,
    )
    # 缺签名头
    with pytest.raises(AuthError, match="missing"):
        auth.verify(authorization=TOKEN, body=body, now=now)
    # 签名不匹配(body 被篡改)
    with pytest.raises(AuthError, match="signature"):
        auth.verify(
            authorization=TOKEN,
            body=b'{"a":2}',
            timestamp=str(now),
            signature=sign(SECRET, body, now),
            now=now,
        )
    # 超出重放窗口
    with pytest.raises(AuthError, match="window"):
        auth.verify(
            authorization=TOKEN,
            body=body,
            timestamp=str(now - 31),
            signature=sign(SECRET, body, now - 31),
            now=now,
        )


# --------------------------------------------------------------------------
# HTTP 端到端(真实服务 + 真实请求)
# --------------------------------------------------------------------------


def test_http_accepts_signed_signal_end_to_end() -> None:
    """带正确 token 的 POST 应被受理并派发到 sink."""
    source, sink, url = _serve()
    try:
        response = httpx.post(
            url, json=_payload(), headers={"Authorization": f"Bearer {TOKEN}"}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == SignalStatus.ACCEPTED.value
        assert body["order_id"] == "order-1"
        assert len(sink.submitted) == 1
        assert sink.submitted[0].symbol == SYMBOL
        assert sink.submitted[0].side == "Buy"
    finally:
        source.stop()


def test_http_rejects_bad_token_without_dispatching() -> None:
    """Token 错误应回 401, 且**不得**派发信号."""
    source, sink, url = _serve()
    try:
        response = httpx.post(
            url, json=_payload(), headers={"Authorization": "Bearer nope"}
        )
        assert response.status_code == 401
        # 不泄漏具体失败原因(避免成为区分预言机)
        assert response.json() == {"error": "unauthorized"}
        assert sink.submitted == [], "鉴权失败的请求绝不能下单"
    finally:
        source.stop()


def test_http_requires_signature_when_secret_configured() -> None:
    """配了 secret 后, 无签名的请求应 401; 正确签名放行."""
    source, sink, url = _serve(secret=SECRET)
    try:
        raw = json.dumps(_payload()).encode()
        unsigned = httpx.post(
            url, content=raw, headers={"Authorization": f"Bearer {TOKEN}"}
        )
        assert unsigned.status_code == 401
        assert sink.submitted == []

        now = int(time.time())
        signed = httpx.post(
            url,
            content=raw,
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
                "X-Signal-Timestamp": str(now),
                "X-Signal-Signature": sign(SECRET, raw, now),
            },
        )
        assert signed.status_code == 200, signed.text
        assert len(sink.submitted) == 1
    finally:
        source.stop()


def test_http_duplicate_returns_200_with_duplicate_status() -> None:
    """重复投递应回 200 + duplicate, 而非错误 —— 否则平台会一直重试."""
    source, sink, url = _serve()
    try:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        first = httpx.post(url, json=_payload("dup-1"), headers=headers)
        second = httpx.post(url, json=_payload("dup-1"), headers=headers)
        assert first.json()["status"] == SignalStatus.ACCEPTED.value
        assert second.status_code == 200
        assert second.json()["status"] == SignalStatus.DUPLICATE.value
        assert len(sink.submitted) == 1, "重复信号只应下一次单"
    finally:
        source.stop()


def test_http_rejects_malformed_payload() -> None:
    """坏 JSON / 不合契约的负载应回 400, 不派发."""
    source, sink, url = _serve()
    try:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        bad_json = httpx.post(url, content=b"{not json", headers=headers)
        assert bad_json.status_code == 400
        not_object = httpx.post(url, content=b"[1,2]", headers=headers)
        assert not_object.status_code == 400
        bad_qty = httpx.post(url, json=_payload(quantity=0), headers=headers)
        assert bad_qty.status_code == 400
        wrong_path = httpx.post(
            url.replace("/signal", "/nope"), json=_payload(), headers=headers
        )
        assert wrong_path.status_code == 404
        assert sink.submitted == []
    finally:
        source.stop()


class _FakeRedis:
    """最小 Redis Stream 桩: 覆盖 xgroup_create / xreadgroup / xack."""

    def __init__(self, entries: List[Tuple[str, Dict[str, str]]]) -> None:
        """预置待投递的 stream entries."""
        self._entries = list(entries)
        self.acked: List[str] = []
        self.groups: List[Tuple[str, str]] = []

    def xgroup_create(
        self, stream: str, group: str, id: str = "0", mkstream: bool = False
    ) -> None:
        """记录建组(第二次调用模拟 BUSYGROUP)."""
        if (stream, group) in self.groups:
            raise RuntimeError("BUSYGROUP Consumer Group name already exists")
        self.groups.append((stream, group))

    def xreadgroup(
        self,
        group: str,
        consumer: str,
        streams: Dict[str, str],
        count: int = 10,
        block: int = 0,
    ) -> Any:
        """一次性交出全部预置 entries, 之后返回空(模拟 block 超时)."""
        if not self._entries:
            time.sleep(0.02)
            return []
        batch = self._entries[:count]
        self._entries = self._entries[count:]
        return [(next(iter(streams)), batch)]

    def xack(self, stream: str, group: str, entry_id: str) -> int:
        """记录 ack."""
        self.acked.append(entry_id)
        return 1


def _drain_redis(entries: List[Tuple[str, Dict[str, str]]]) -> Any:
    """跑一轮 Redis 消费, 返回 (sink, fake, source)."""
    sink = _RecordingSink()
    fake = _FakeRedis(entries)
    source = RedisSignalSource(client=fake, block_ms=10)
    dispatcher = SignalDispatcher(sink, on_result=source.on_result)
    source.bind(dispatcher.dispatch)
    source.start()
    deadline = time.monotonic() + 5.0
    expected = len(entries)
    while len(fake.acked) < expected and time.monotonic() < deadline:
        time.sleep(0.02)
    source.stop()
    return sink, fake, source


def test_redis_consumes_dispatches_and_acks() -> None:
    """Stream entry → dispatch → XACK 全链路(fake client)."""
    entries = [
        ("1-1", {"signal": json.dumps(_payload("r-1"))}),
        ("1-2", {"signal": json.dumps(_payload("r-2", action="sell"))}),
    ]
    sink, fake, _ = _drain_redis(entries)

    assert [s.signal_id for s in sink.submitted] == ["r-1", "r-2"]
    assert sink.submitted[1].side == "Sell"
    assert fake.acked == ["1-1", "1-2"], "每条都必须 ack"


def test_redis_acks_unparsable_entry_to_avoid_blocking_stream() -> None:
    """坏消息也要 ack, 否则会永远堵在 pending 里反复重投."""
    entries = [
        ("2-1", {"signal": "{not json"}),
        ("2-2", {"signal": json.dumps(_payload("r-ok"))}),
    ]
    sink, fake, _ = _drain_redis(entries)

    assert [s.signal_id for s in sink.submitted] == ["r-ok"], "坏消息不应被派发"
    assert fake.acked == ["2-1", "2-2"], "坏消息同样要 ack"


def test_redis_tolerates_existing_consumer_group() -> None:
    """重复建组(BUSYGROUP)是正常路径, 不应抛出."""
    fake = _FakeRedis([])
    fake.xgroup_create("akquant:signals", "akquant")  # 预先建好
    source = RedisSignalSource(client=fake, block_ms=10)
    source.bind(SignalDispatcher(_RecordingSink()).dispatch)
    source.start()  # 不应抛
    source.stop()
    assert len(fake.groups) == 1


def test_http_concurrent_duplicates_submit_once() -> None:
    """并发投同一 signal_id, 只应下一次单(幂等在锁内)."""
    source, sink, url = _serve()
    try:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        barrier = threading.Barrier(8)
        statuses: List[str] = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            reply = httpx.post(url, json=_payload("race-1"), headers=headers)
            with lock:
                statuses.append(reply.json()["status"])

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)

        assert len(sink.submitted) == 1, f"并发重复下了 {len(sink.submitted)} 单"
        assert statuses.count(SignalStatus.ACCEPTED.value) == 1
        assert statuses.count(SignalStatus.DUPLICATE.value) == 7
    finally:
        source.stop()
