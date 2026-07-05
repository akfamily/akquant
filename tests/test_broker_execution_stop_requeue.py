"""check_stop_triggers: 失败重试(不崩)+ on_error + 成功 remap 记录."""

from akquant.gateway.broker_execution import MAX_STOP_SUBMIT_ATTEMPTS, BrokerExecution


class _Cache:
    def positions(self):
        return {}

    def available_positions(self):
        return {}

    def open_orders(self):
        return []

    def account(self):
        return None


class _Gw:
    def cancel_order(self, bid):
        pass

    def sync_open_orders(self):
        return []


class _OkSub:
    def __init__(self):
        self.n = 0

    def submit_order(self, **kw):
        self.n += 1
        return "BID-9"


class _FailSub:
    def __init__(self):
        self.n = 0

    def submit_order(self, **kw):
        self.n += 1
        raise RuntimeError("broker not ready")


class _S:
    current_bar = None
    current_tick = None

    def __init__(self):
        self.errors = []

    def on_error(self, exc, source, payload=None):
        self.errors.append((source, payload))


def test_success_records_remap() -> None:
    """止损触发提交成功后应调用 record_stop_remap(local_id, broker_order_id)."""
    remaps = []
    ex = BrokerExecution(
        _S(),
        _Gw(),
        _Cache(),
        _OkSub(),
        record_stop_remap=lambda lid, bid: remaps.append((lid, bid)),
    )
    oid = ex.submit_order(
        symbol="X",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=9.5,
    )
    ex.check_stop_triggers("X", last=9.4, high=9.6, low=9.3)
    assert remaps == [(oid, "BID-9")]


def test_failure_requeues_and_notifies_then_gives_up() -> None:
    """止损触发提交失败应重试(上限 MAX_STOP_SUBMIT_ATTEMPTS)+on_error, 不崩溃."""
    s = _S()
    ex = BrokerExecution(s, _Gw(), _Cache(), _FailSub())
    ex.submit_order(
        symbol="X",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=9.5,
    )
    # attempt 1: fails → requeued, on_error, still in book, no raise
    ex.check_stop_triggers("X", last=9.4, high=9.6, low=9.3)
    assert len(ex.get_open_orders("X")) == 1
    assert len(s.errors) == 1
    # attempts 2..MAX: keeps failing; after MAX total attempts, dropped
    for _ in range(MAX_STOP_SUBMIT_ATTEMPTS):
        ex.check_stop_triggers("X", last=9.4, high=9.6, low=9.3)
    assert ex.get_open_orders("X") == []  # given up
    assert len(s.errors) == MAX_STOP_SUBMIT_ATTEMPTS
