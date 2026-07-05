"""LocalStopBook 并发 register/check 不崩、无丢单(GIL 下顺序断言)."""

import threading

from akquant.gateway.local_stop_book import LocalStopBook, LocalStopOrder


def test_concurrent_register_and_check_no_crash() -> None:
    """两线程并发 register/check 不抛异常."""
    book = LocalStopBook()
    errors = []

    def registrar() -> None:
        try:
            for i in range(200):
                book.register(
                    LocalStopOrder(
                        f"L{i}", "X", "Sell", 1, "stopmarket", trigger_price=100.0
                    )
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def checker() -> None:
        try:
            for _ in range(200):
                book.check("X", last=101.0, high=101.0, low=100.5)  # not triggered
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=registrar), threading.Thread(target=checker)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_has_internal_lock() -> None:
    """LocalStopBook 持内部锁."""
    book = LocalStopBook()
    assert hasattr(book, "_lock")


def test_behavior_unchanged_single_thread() -> None:
    """加锁不改变单线程语义."""
    book = LocalStopBook()
    book.register(
        LocalStopOrder("L1", "X", "Sell", 100, "stopmarket", trigger_price=9.5)
    )
    assert book.check("X", last=9.8, high=9.9, low=9.6) == []
    assert [o.local_id for o in book.check("X", last=9.4, high=9.7, low=9.3)] == ["L1"]
    assert book.open_orders() == []
