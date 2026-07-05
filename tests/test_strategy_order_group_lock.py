"""Strategy OCO/bracket 锁: 存在、可 pickle(锁不序列化)、行为不变."""

import pickle
import threading

from akquant.strategy import Strategy

_RLOCK_TYPE = type(threading.RLock())


def test_order_group_lock_present() -> None:
    """构造即带 RLock."""
    s = Strategy.__new__(Strategy)
    assert isinstance(s._order_group_lock, _RLOCK_TYPE)


def test_pickle_roundtrip_recreates_lock() -> None:
    """RLock 不入 pickle, 反序列化重建."""
    s = Strategy.__new__(Strategy)
    s._oco_groups = {"g": {"a", "b"}}
    s._oco_order_to_group = {"a": "g", "b": "g"}
    data = pickle.dumps(s)  # 锁不可 pickle → __getstate__ 必须剔除
    s2 = pickle.loads(data)
    assert isinstance(s2._order_group_lock, _RLOCK_TYPE)
    assert s2._oco_groups == {"g": {"a", "b"}}


def test_oco_cancel_peer_still_works_under_lock() -> None:
    """一腿成交撤对手(锁不改变行为)."""

    class _Exec:
        def __init__(self) -> None:
            self.canceled: list = []

        def submit_order(self, **kw: object) -> str:
            return "X"

        def cancel_order(self, oid: str) -> None:
            self.canceled.append(oid)

    s = Strategy.__new__(Strategy)
    s.execution = _Exec()
    s._oco_groups = {"g": {"a", "b"}}
    s._oco_order_to_group = {"a": "g", "b": "g"}
    s._use_engine_oco = False
    s._use_engine_bracket = False
    s._pending_brackets = {}

    class _T:
        order_id = "a"
        symbol = "X"
        quantity = 1.0

    s._process_order_groups(_T())
    assert s.execution.canceled == ["b"]  # 对手 b 被撤
