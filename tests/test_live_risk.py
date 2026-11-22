"""实盘前置风控: 委托合规规则与账户级盈亏熔断."""

import datetime as dt
import threading
from types import SimpleNamespace
from typing import Any, List

import pytest
from akquant import LiveRiskConfig
from akquant.gateway import ctp_native
from akquant.gateway.broker_models import BrokerCapability, UnifiedOrderRequest
from akquant.gateway.brokers.ctp.offset import OffsetBook
from akquant.gateway.ctp_native import CTPTraderGateway
from akquant.gateway.live_risk import LiveRiskGuard
from akquant.gateway.order_submitter import BrokerOrderSubmitter

DAY1, DAY2 = dt.date(2026, 11, 23), dt.date(2026, 11, 24)
OPEN = [("open", 1.0)]
CLOSE = [("close_today", 1.0)]


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _check(guard: LiveRiskGuard, legs: Any = OPEN, **kwargs: Any) -> Any:
    params: dict = dict(
        symbol="rb2701",
        side="Buy",
        price=3000.0,
        legs=legs,
        trading_day=DAY1,
        open_orders=lambda: [],
        account=lambda: None,
    )
    params.update(kwargs)
    return guard.check(**params)


def test_order_flow_limit_uses_rolling_window() -> None:
    """滚动窗口内报出的腿数超过上限时拒绝, 窗口过后恢复."""
    clock = _Clock()
    guard = LiveRiskGuard(LiveRiskConfig(order_flow_limit=2), clock=clock)
    guard.record_sent("rb2701", "open", 1, DAY1)
    assert _check(guard) is None
    assert _check(guard, legs=OPEN * 2) == (
        "委托流速超限: 1 秒内已报 1 笔, 本笔 2 笔, 上限 2 笔"
    )
    guard.record_sent("rb2701", "open", 1, DAY1)
    assert _check(guard) is not None
    clock.now = 1.0
    assert _check(guard) is None


def test_cancel_limit_blocks_new_orders_per_symbol_and_day() -> None:
    """单合约被撤委托达到上限后拒绝该合约新委托; 同一委托只计一次, 换日清零."""
    guard = LiveRiskGuard(LiveRiskConfig(cancel_limit=2))
    for order_id in ("a", "a", "b"):
        guard.on_order_event(
            {"status": "cancelled", "broker_order_id": order_id, "symbol": "rb2701"},
            DAY1,
        )
    guard.on_order_event({"status": "filled", "broker_order_id": "c"}, DAY1)
    assert _check(guard) == (
        "rb2701 本交易日已撤单 2 次, 达到撤单次数上限 2, 不再报新委托"
    )
    assert _check(guard, symbol="hc2701") is None
    assert _check(guard, trading_day=DAY2) is None


def test_open_count_and_volume_limits() -> None:
    """单合约本交易日开仓笔数、手数上限; 平仓不计."""
    guard = LiveRiskGuard(LiveRiskConfig(open_count_limit=2, open_volume_limit=5))
    guard.record_sent("rb2701", "open", 3, DAY1)
    guard.record_sent("rb2701", "close_today", 3, DAY1)
    assert _check(guard, legs=[("open", 3.0)]) == (
        "rb2701 本交易日已报开仓 3 手, 本笔 3 手, 超过开仓手数上限 5"
    )
    guard.record_sent("rb2701", "open", 1, DAY1)
    assert _check(guard) == (
        "rb2701 本交易日已报开仓委托 2 笔, 本笔 1 笔, 超过开仓次数上限 2"
    )
    assert _check(guard, legs=CLOSE) is None


def test_active_orders_and_self_trade() -> None:
    """未完成委托数上限; 同合约反向可成交的挂单视为自成交风险."""
    working = [
        SimpleNamespace(
            symbol="rb2701", side="Sell", price=3001.0, broker_order_id="1:7:5"
        )
    ]
    guard = LiveRiskGuard(LiveRiskConfig(active_order_limit=1, self_trade_check=True))
    assert _check(guard, open_orders=lambda: working) == (
        "未完成委托 1 笔, 本笔 1 笔, 超过上限 1 笔"
    )
    guard = LiveRiskGuard(LiveRiskConfig(self_trade_check=True))
    assert _check(guard, price=3000.0, open_orders=lambda: working) is None
    assert _check(guard, price=3001.0, open_orders=lambda: working) == (
        "rb2701 自成交风险: 本笔 buy 3001 可与本账户未完成的 sell 委托 1:7:5(3001) 成交"
    )
    assert "本笔 buy 市价" in _check(guard, price=None, open_orders=lambda: working)
    assert _check(guard, side="Sell", open_orders=lambda: working) is None


def test_daily_loss_locks_opening_for_the_day() -> None:
    """当日亏损超限后本交易日只允许平仓, 权益回升也不解除, 换日解除."""
    guard = LiveRiskGuard(LiveRiskConfig(max_daily_loss=0.04))
    account = SimpleNamespace(equity=95_000.0, day_start_equity=100_000.0)
    assert _check(guard, account=lambda: account) == (
        "账户当日亏损超限后只允许平仓: 当日初始权益 100000.00, 当前权益 95000.00, "
        "亏损 5000.00(5.00%), 上限 4.00%"
    )
    assert _check(guard, legs=CLOSE, account=lambda: account) is None
    account.equity = 99_000.0
    assert _check(guard, account=lambda: account) is not None
    assert _check(guard, trading_day=DAY2, account=lambda: account) is None
    missing = SimpleNamespace(equity=95_000.0, day_start_equity=None)
    assert _check(guard, account=lambda: missing) == (
        "柜台账户快照没有当日初始权益(day_start_equity), 无法校验 max_daily_loss"
    )
    zero = SimpleNamespace(equity=95_000.0, day_start_equity=0.0)
    assert _check(guard, trading_day=DAY2, account=lambda: zero) == (
        "柜台账户快照的当日初始权益 0.00 不为正, 无法按比例校验 max_daily_loss"
    )


def test_drawdown_locks_opening_for_the_run() -> None:
    """回撤按观察到的权益峰值计算, 超限后锁定到进程结束."""
    guard = LiveRiskGuard(LiveRiskConfig(max_account_drawdown=0.05))
    guard.observe_equity(110_000.0)
    account = SimpleNamespace(equity=104_000.0, day_start_equity=100_000.0)
    assert _check(guard, account=lambda: account) == (
        "账户回撤超限后只允许平仓: 权益峰值 110000.00, 当前权益 104000.00, "
        "回撤 6000.00(5.45%), 上限 5.00%"
    )
    account.equity = 109_000.0
    assert _check(guard, trading_day=DAY2, account=lambda: account) is not None
    assert _check(guard, legs=CLOSE, account=lambda: account) is None


def test_config_rejects_non_positive_limits() -> None:
    """限额须为正数."""
    with pytest.raises(ValueError) as exc:
        LiveRiskConfig(cancel_limit=0)
    assert str(exc.value) == "LiveRiskConfig.cancel_limit 须为正数, 实际: 0"
    with pytest.raises(ValueError) as exc:
        LiveRiskConfig(max_daily_loss=50_000)
    assert str(exc.value) == (
        "LiveRiskConfig.max_daily_loss 是比例(如 0.05 表示 5%), 须在 0 与 1 之间, "
        "实际: 50000"
    )


def test_submitter_applies_live_risk_after_leg_split() -> None:
    """下单器在拆腿后检查、报出后计数; 超限作为前置拒单回到 on_reject."""
    placed: List[UnifiedOrderRequest] = []
    rejects: List[str] = []

    class _Gw:
        def place_order(self, req: UnifiedOrderRequest) -> str:
            placed.append(req)
            return f"b{len(placed)}"

    class _Strategy:
        _owner_strategy_id = "_default"
        broker_ready = True
        trading_day = DAY1
        execution = SimpleNamespace(
            broker_open_orders=lambda: [],
            broker_account=lambda: SimpleNamespace(equity=1.0, day_start_equity=1.0),
        )

        def on_order(self, order: Any) -> None:
            pass

        def on_reject(self, order: Any) -> None:
            rejects.append(order.reject_reason)

    cap = BrokerCapability(
        broker_name="middleware",
        position_effect=True,
        supported_position_effects=("open", "close"),
    )
    submitter = BrokerOrderSubmitter(
        trader_gateway=_Gw(),
        strategy=_Strategy(),
        resolve_trader_capabilities=lambda _gw: cap,
        next_client_order_id=lambda: f"c{len(placed)}",
        can_submit_client_order=lambda _cid: True,
        sync_order_id_mapping=lambda _c, _b: None,
        bind_order_owner=lambda _c, _b, _o: None,
        notify_strategy_error=lambda *_a, **_k: None,
        payload_field=lambda obj, name: getattr(obj, name, None),
        get_execution_capabilities=lambda: cap.as_execution_capabilities(),
        record_order_request=lambda *_a: None,
        live_risk=LiveRiskConfig(open_count_limit=1, cancel_limit=1),
    )

    def submit() -> Any:
        return submitter.submit_order(
            symbol="rb2701",
            side="Buy",
            quantity=1,
            price=3000.0,
            order_type="Limit",
            position_effect="open",
        )

    assert submit().failure is None
    assert submit().failure == "rejected"
    assert len(placed) == 1
    assert rejects == [
        "rb2701 本交易日已报开仓委托 1 笔, 本笔 1 笔, 超过开仓次数上限 1"
    ]
    # 柜台回报经下单器转给风控
    submitter.on_broker_event(
        "order", {"status": "cancelled", "broker_order_id": "b1", "symbol": "hc2701"}
    )
    receipt = submitter.submit_order(
        symbol="hc2701",
        side="Buy",
        quantity=1,
        price=3000.0,
        order_type="Limit",
        position_effect="open",
    )
    assert receipt.failure == "rejected"
    assert rejects[-1].startswith("hc2701 本交易日已撤单 1 次")


def test_live_risk_requires_broker_live() -> None:
    """非 broker_live 模式配置 live_risk 时报错."""
    from akquant.live._runner import LiveRunner

    with pytest.raises(ValueError, match="live_risk 只在 trading_mode='broker_live'"):
        LiveRunner(
            strategy_cls=lambda ctx, bar: None,
            instruments=[],
            broker="replay",
            trading_mode="paper",
            live_risk=LiveRiskConfig(cancel_limit=1),
        )


def test_ctp_account_reports_day_start_equity(monkeypatch: pytest.MonkeyPatch) -> None:
    """CTP 账户快照的当日初始权益按静态权益公式计算(含出入金)."""
    monkeypatch.setattr(
        ctp_native.tdapi,
        "CThostFtdcQryTradingAccountField",
        SimpleNamespace,
        raising=False,
    )
    gateway = CTPTraderGateway.__new__(CTPTraderGateway)
    gateway.__dict__.update(
        broker_id="9999",
        user_id="tester",
        req_id=0,
        connected=True,
        login_status=True,
        ready_to_trade=True,
        _account_query_timeout_sec=0.5,
        _account_query_event=threading.Event(),
        _account_query_result=None,
        _account_query_error="",
        _query_lock=threading.RLock(),
        _query_retry_interval_sec=0.0,
        offset_book=OffsetBook(),
    )

    def query(request: Any, req_id: int) -> int:
        gateway.OnRspQryTradingAccount(
            SimpleNamespace(
                Balance=101_000.0,
                Available=90_000.0,
                PreBalance=100_000.0,
                PreCredit=0.0,
                PreMortgage=0.0,
                Mortgage=0.0,
                Withdraw=2_000.0,
                Deposit=5_000.0,
            ),
            SimpleNamespace(ErrorID=0, ErrorMsg=""),
            req_id,
            True,
        )
        return 0

    gateway.api = SimpleNamespace(ReqQryTradingAccount=query)
    account = gateway.query_account()
    assert account is not None and account["day_start_equity"] == 103_000.0


def test_circuit_trip_logs_once_with_alert_mark(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """熔断触发时记一条带提醒标记的日志, 之后的拒单不重复记."""
    account = SimpleNamespace(equity=95_000.0, day_start_equity=100_000.0)
    with caplog.at_level("WARNING", logger="akquant.gateway.live_risk"):
        for config in (
            LiveRiskConfig(max_daily_loss=0.04),
            LiveRiskConfig(max_account_drawdown=0.03),
        ):
            guard = LiveRiskGuard(config)
            _check(guard, account=lambda: account)
            _check(guard, account=lambda: account)
    marked = [
        (r.getMessage().split(":")[0], getattr(r, "alert", None))
        for r in caplog.records
    ]
    assert marked == [
        ("账户当日亏损超限, 当日只允许平仓", "风控熔断"),
        ("账户回撤超限, 此后只允许平仓", "风控熔断"),
    ]
