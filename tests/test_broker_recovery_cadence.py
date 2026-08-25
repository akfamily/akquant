"""broker recovery 三层节奏: heartbeat 每拍, account 中频, 全量 sync 低频兜底."""

from typing import Any

from akquant.gateway.broker_recovery import BrokerRecovery


class _Gateway:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive
        self.heartbeats = 0
        self.sync_orders_calls = 0
        self.sync_trades_calls = 0
        self.account_calls = 0
        self.connects = 0

    def heartbeat(self) -> bool:
        self.heartbeats += 1
        return self.alive

    def connect(self) -> None:
        self.connects += 1
        self.alive = True

    def sync_open_orders(self) -> list:
        self.sync_orders_calls += 1
        return []

    def sync_today_trades(self) -> list:
        self.sync_trades_calls += 1
        return []

    def query_account(self) -> None:
        self.account_calls += 1
        return None


def _recovery(gateway: _Gateway) -> BrokerRecovery:
    return BrokerRecovery(
        get_trader_gateway=lambda: gateway,
        queue_broker_event=lambda name, payload: None,
        notify_strategy_error=lambda *a: None,
        get_on_broker_event=lambda: None,
        get_recovery_mode=lambda: "compatible",
        get_last_error_key=lambda: "",
        set_last_error_key=lambda key: None,
    )


def test_heartbeat_only_cycle_skips_sync_and_account() -> None:
    """只跑心跳的那些拍不碰柜台的全量查询接口."""
    g = _Gateway()
    _recovery(g).run_cycle(sync_orders=False, sync_trades=False, refresh_account=False)

    assert g.heartbeats == 1
    assert g.sync_orders_calls == 0
    assert g.sync_trades_calls == 0
    assert g.account_calls == 0


def test_full_cycle_runs_everything() -> None:
    """到兜底点的那一拍跑全量."""
    g = _Gateway()
    _recovery(g).run_cycle(sync_orders=True, sync_trades=True, refresh_account=True)

    assert (g.sync_orders_calls, g.sync_trades_calls, g.account_calls) == (1, 1, 1)


def test_account_can_refresh_without_full_sync() -> None:
    """Account 是独立一档: 刷资金不等于拉全量委托."""
    g = _Gateway()
    _recovery(g).run_cycle(sync_orders=False, sync_trades=False, refresh_account=True)

    assert g.account_calls == 1
    assert g.sync_orders_calls == 0


def test_reconnect_forces_sync_even_when_disabled() -> None:
    """断线重连成功后本轮强制补齐, 不等兜底间隔——这才是 recovery 的本意."""
    g = _Gateway(alive=False)
    reconnected = _recovery(g).run_cycle(
        sync_orders=False, sync_trades=False, refresh_account=False
    )

    assert reconnected is True
    assert g.connects == 1
    assert g.sync_orders_calls == 1
    assert g.sync_trades_calls == 1
    assert g.account_calls == 0


def test_no_reconnect_returns_false() -> None:
    """连接正常时不报重连, runner 据此不重置兜底计时."""
    g = _Gateway()
    assert _recovery(g).run_cycle() is False


def test_sync_orders_and_sync_trades_are_independent_switches() -> None:
    """两个开关是 run_cycle 公开签名的独立契约, 不能互相串线."""
    g = _Gateway()
    _recovery(g).run_cycle(sync_orders=True, sync_trades=False, refresh_account=False)

    assert g.sync_orders_calls == 1
    assert g.sync_trades_calls == 0

    g2 = _Gateway()
    _recovery(g2).run_cycle(sync_orders=False, sync_trades=True, refresh_account=False)

    assert g2.sync_orders_calls == 0
    assert g2.sync_trades_calls == 1


def test_intervals_default_to_three_tier() -> None:
    """默认 1s / 5s / 30s 三层节奏."""
    from akquant.live._runner import _resolve_recovery_intervals

    tick, account, sync = _resolve_recovery_intervals({})
    assert (tick, account, sync) == (1.0, 5.0, 30.0)


def test_intervals_read_from_gateway_options() -> None:
    """柜台限流阈值是部署期约束, 三档都可覆盖."""
    from akquant.live._runner import _resolve_recovery_intervals

    tick, account, sync = _resolve_recovery_intervals(
        {
            "recovery_interval_sec": 2.0,
            "recovery_account_interval_sec": 10.0,
            "recovery_sync_interval_sec": 60.0,
        }
    )
    assert (tick, account, sync) == (2.0, 10.0, 60.0)


def test_out_of_order_intervals_are_clamped_not_raised() -> None:
    """越界只钳制, **不抛异常** —— 让运维在启动时吃 ValueError 是最差处理."""
    from akquant.live._runner import _resolve_recovery_intervals

    tick, account, sync = _resolve_recovery_intervals(
        {
            "recovery_interval_sec": 5.0,
            "recovery_account_interval_sec": 1.0,  # < tick
            "recovery_sync_interval_sec": 2.0,  # < account
        }
    )
    assert tick == 5.0
    assert account >= tick
    assert sync >= account


def test_invalid_interval_falls_back_to_default() -> None:
    """非法值(负数/非数字)回退默认值并告警, 不中断启动."""
    from akquant.live._runner import _resolve_recovery_intervals

    tick, _, _ = _resolve_recovery_intervals({"recovery_interval_sec": -1})
    assert tick == 1.0
    tick2, _, _ = _resolve_recovery_intervals({"recovery_interval_sec": "abc"})
    assert tick2 == 1.0


def test_nan_and_inf_interval_fall_back_to_default() -> None:
    """`nan`/`inf` 不抛异常也不满足 `<= 0`, 需显式挡掉, 否则该档节奏会静默永久失效."""
    from akquant.live._runner import _resolve_recovery_intervals

    tick_nan, _, _ = _resolve_recovery_intervals({"recovery_interval_sec": "nan"})
    assert tick_nan == 1.0
    tick_inf, _, _ = _resolve_recovery_intervals({"recovery_interval_sec": "inf"})
    assert tick_inf == 1.0


def test_jittered_sync_interval_stays_within_ten_percent(monkeypatch: Any) -> None:
    """抖动落在 [base*0.9, base*1.1] 闭区间内, 断言不依赖真实随机数取值."""
    from akquant.live._runner import LiveRunner

    monkeypatch.setattr("akquant.live._runner.random.uniform", lambda a, b: a)
    runner = LiveRunner.__new__(LiveRunner)
    runner._broker_sync_interval_sec = 30.0

    assert runner._jittered_sync_interval() == 27.0
