"""client_order_id 生成: 会话内递增 + 跨重启/跨进程唯一(旧格式重启撞号 → 409)."""

import threading
import uuid

from akquant.live._runner import LiveRunner


def _runner(broker: str = "middleware") -> LiveRunner:
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = broker
    runner._broker_submit_seq = 0
    runner._broker_session_tag = uuid.uuid4().hex[:8]
    runner._broker_submit_lock = threading.Lock()
    return runner


def test_ids_increment_within_session() -> None:
    """同一会话内序号递增且带 broker 前缀."""
    runner = _runner()
    ids = [runner._next_client_order_id() for _ in range(3)]
    assert [i.split("-")[-1] for i in ids] == ["1", "2", "3"]
    assert all(i.startswith("middleware-") for i in ids)
    assert len(set(ids)) == 3


def test_ids_differ_across_sessions() -> None:
    """两个 runner(等价于重启)的首个 id 不再相同——旧格式此处必然撞号."""
    first = _runner()._next_client_order_id()
    second = _runner()._next_client_order_id()
    assert first != second
    assert first.endswith("-1") and second.endswith("-1")


def test_session_tag_shared_by_all_ids_of_one_session() -> None:
    """同一会话所有 id 共享会话标记,便于按会话检索."""
    runner = _runner()
    ids = [runner._next_client_order_id() for _ in range(4)]
    tags = {i.rsplit("-", 1)[0] for i in ids}
    assert len(tags) == 1


def test_concurrent_generation_is_unique() -> None:
    """并发生成不重号(锁覆盖序号自增)."""
    runner = _runner()
    ids: list[str] = []
    lock = threading.Lock()

    def _worker() -> None:
        local = [runner._next_client_order_id() for _ in range(50)]
        with lock:
            ids.extend(local)

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ids) == 200
    assert len(set(ids)) == 200


def test_missing_session_tag_falls_back() -> None:
    """绕过 __init__ 未设标记时退化为固定串,不抛 AttributeError."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner._broker_submit_seq = 0
    runner._broker_submit_lock = threading.Lock()
    assert runner._next_client_order_id() == "ctp-nosess-1"
