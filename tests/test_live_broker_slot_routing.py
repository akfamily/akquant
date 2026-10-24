"""broker_live 多槽位: 柜台回报按所属槽位派发.

回测/paper 侧(阶段 5)订单、成交、拒单、过期回调按 ``owner_strategy_id`` 只派
给所属槽位, owner 为空或未知归默认槽位; broker_live 的柜台回报此前在多槽位下
广播给全部槽位, 且 OCO/Bracket 协调(``_process_order_groups``)与账户回调
(``on_portfolio_update``)被静默丢弃。

这里驱动完整的 ``LiveRunner.run()``(假引擎 + 假柜台), 走真实的回调目标装配
与 ``BrokerEventBridge`` 派发链, 不直接构造回调目标。
"""

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, cast

import akquant.live._runner as live_module
import pytest
from akquant.live._runner import LiveRunner
from akquant.strategy import Strategy


class _FakeTraderGateway:
    """最小假柜台: 登记回调、心跳就绪、记录撤单."""

    def __init__(self) -> None:
        self.callbacks: Dict[str, Callable[[Any], None]] = {}
        self.canceled: List[str] = []

    def on_order(self, callback: Callable[[Any], None]) -> None:
        self.callbacks["order"] = callback

    def on_trade(self, callback: Callable[[Any], None]) -> None:
        self.callbacks["trade"] = callback

    def on_execution_report(self, callback: Callable[[Any], None]) -> None:
        self.callbacks["execution_report"] = callback

    def connect(self) -> None:
        pass

    def start(self) -> None:
        pass

    def heartbeat(self) -> bool:
        return True

    def query_positions(self) -> list:
        return []

    def sync_open_orders(self) -> list:
        return []

    def sync_today_trades(self) -> list:
        return []

    def cancel_order(self, broker_order_id: str) -> None:
        self.canceled.append(str(broker_order_id))

    def emit(self, event_name: str, payload: Any) -> None:
        self.callbacks[event_name](payload)


class _FakeEngine:
    """驱动 run() 的引擎替身: run() 内执行场景脚本, 再冲刷柜台事件队列."""

    def __init__(self, scenario: Callable[[LiveRunner], None]) -> None:
        self.scenario = scenario
        self.runner: Optional[LiveRunner] = None

    def add_data(self, *a: Any) -> None:
        pass

    def set_cash(self, *a: Any) -> None:
        pass

    def add_instrument(self, *a: Any) -> None:
        pass

    def use_simulated_execution(self) -> None:
        pass

    def use_realtime_execution(self) -> None:
        pass

    def use_china_market(self) -> None:
        pass

    def use_china_futures_market(self) -> None:
        pass

    def configure_sessions(self, *a: Any) -> None:
        pass

    def set_strategy_slots(self, slot_ids: list) -> None:
        pass

    def set_default_strategy_id(self, strategy_id: str) -> None:
        pass

    def set_strategy_for_slot(self, slot_index: int, strategy: Any) -> None:
        pass

    def run(self, strategy: Any, show_progress: bool = False) -> None:
        if hasattr(strategy, "_on_start_internal"):
            strategy._on_start_internal()
        assert self.runner is not None
        self.scenario(self.runner)
        # 停掉派发线程: 退出前对回调目标做最后一次 drain, 结果确定、无需等待。
        self.runner._stop_broker_dispatcher()

    def get_results(self) -> Any:
        raise RuntimeError("fake engine has no results")


class _Recorder(Strategy):
    """记录收到的柜台回调(按 client_order_id / 成交 id)."""

    def __init__(self) -> None:
        super().__init__()
        self.order_cids: List[str] = []
        self.trade_ids: List[str] = []
        self.report_ids: List[str] = []
        self.portfolio_updates: List[Any] = []

    def on_bar(self, bar: Any) -> None:
        pass

    def on_order(self, order: Any) -> None:
        self.order_cids.append(str(getattr(order, "client_order_id", "")))

    def on_trade(self, trade: Any) -> None:
        self.trade_ids.append(str(getattr(trade, "id", "")))

    def on_execution_report(self, report: Any) -> None:
        self.report_ids.append(str(report.get("id", "")))

    def on_portfolio_update(self, snapshot: Any) -> None:
        self.portfolio_updates.append(snapshot)


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> _FakeTraderGateway:
    """把 run() 的网关工厂替换成带假柜台的 bundle."""
    trader = _FakeTraderGateway()
    monkeypatch.setattr(
        live_module,
        "create_gateway_bundle",
        lambda **kw: SimpleNamespace(
            market_gateway=None,
            trader_gateway=trader,
            trader_capabilities=None,
            metadata=None,
        ),
    )
    return trader


def _run(
    main: Strategy,
    slots: Dict[str, Strategy],
    scenario: Callable[[LiveRunner], None],
) -> LiveRunner:
    runner = LiveRunner(
        strategy_cls=main,
        instruments=[],
        broker="ctp",
        trading_mode="broker_live",
        strategy_id="alpha",
        strategies_by_slot=cast(Any, slots) or None,
    )
    engine = _FakeEngine(scenario)
    engine.runner = runner
    runner.engine = cast(Any, engine)
    runner.run(cash=1000.0)
    return runner


def _order(cid: str, bid: str) -> Dict[str, Any]:
    return {
        "broker_order_id": bid,
        "client_order_id": cid,
        "symbol": "IF2406",
        "status": "Submitted",
        "filled_quantity": 0.0,
    }


def _trade(tid: str, cid: str, bid: str) -> Dict[str, Any]:
    return {
        "trade_id": tid,
        "broker_order_id": bid,
        "client_order_id": cid,
        "symbol": "IF2406",
        "side": "Buy",
        "quantity": 1.0,
        "price": 10.0,
    }


def test_broker_live_order_callbacks_route_to_owner_slot(
    gateway: _FakeTraderGateway,
) -> None:
    """两个槽位各下一单：柜台回报只回调下单的槽位；无 owner 的回报归默认槽位."""
    main, beta = _Recorder(), _Recorder()

    def scenario(runner: LiveRunner) -> None:
        # 等价于 submitter 下单成功后登记的归属(_bind_order_owner)。
        runner._bind_order_owner("c-a", "b-a", "alpha")
        runner._bind_order_owner("c-b", "b-b", "beta")
        gateway.emit("order", _order("c-a", "b-a"))
        gateway.emit("order", _order("c-b", "b-b"))
        gateway.emit("order", _order("c-x", "b-x"))  # 柜台侧产生, 无归属
        gateway.emit("trade", _trade("t-a", "c-a", "b-a"))
        gateway.emit("trade", _trade("t-b", "c-b", "b-b"))
        gateway.emit("trade", _trade("t-x", "c-x", "b-x"))
        # 执行回报是原始 payload(不经适配), 按订单号查归属表。
        gateway.emit("execution_report", {"id": "r-a", "broker_order_id": "b-a"})
        gateway.emit("execution_report", {"id": "r-b", "client_order_id": "c-b"})
        # 未登记的订单号(柜台侧产生的单)归默认槽位。
        gateway.emit("execution_report", {"id": "r-x", "broker_order_id": "b-y"})

    _run(main, {"beta": beta}, scenario)

    assert set(main.order_cids) == {"c-a", "c-x"}
    assert set(beta.order_cids) == {"c-b"}
    assert set(main.trade_ids) == {"t-a", "t-x"}
    assert set(beta.trade_ids) == {"t-b"}
    assert set(main.report_ids) == {"r-a", "r-x"}
    assert set(beta.report_ids) == {"r-b"}


def test_broker_live_multi_slot_oco_still_coordinated(
    gateway: _FakeTraderGateway,
) -> None:
    """多槽位下 OCO 一腿成交后，另一腿被撤（_process_order_groups 被调用）."""

    class _OcoSlot(_Recorder):
        def on_start(self) -> None:
            # 两柜台限价单绑 OCO(同 test_oco_bracket_broker_live_scenario)
            self._oco_groups = {"g": {"b-1", "b-2"}}
            self._oco_order_to_group = {"b-1": "g", "b-2": "g"}

    main, beta = _Recorder(), _OcoSlot()

    def scenario(runner: LiveRunner) -> None:
        runner._bind_order_owner("c-1", "b-1", "beta")
        runner._bind_order_owner("c-2", "b-2", "beta")
        gateway.emit("trade", _trade("t-1", "c-1", "b-1"))  # 一腿成交

    _run(main, {"beta": beta}, scenario)

    assert gateway.canceled == ["b-2"]
    assert beta.trade_ids == ["t-1"]
    assert main.trade_ids == []


def test_broker_live_portfolio_update_reaches_all_slots(
    gateway: _FakeTraderGateway,
) -> None:
    """账户回报广播给全部槽位."""
    main, beta = _Recorder(), _Recorder()
    account = SimpleNamespace(account_id="acct-1", equity=1000.0, cash=1000.0)

    def scenario(runner: LiveRunner) -> None:
        # 与 broker_recovery 拉到账户快照后的入队方式相同。
        runner._queue_broker_event("account", account)

    _run(main, {"beta": beta}, scenario)

    assert main.portfolio_updates == [account]
    assert beta.portfolio_updates == [account]


def test_broker_live_single_strategy_receives_all_callbacks(
    gateway: _FakeTraderGateway,
) -> None:
    """单策略: 回调目标仍是策略本身, 有无归属的回报、OCO 协调与账户回调全部送达."""

    class _OcoMain(_Recorder):
        def on_start(self) -> None:
            self._oco_groups = {"g": {"b-1", "b-2"}}
            self._oco_order_to_group = {"b-1": "g", "b-2": "g"}

    main = _OcoMain()
    account = SimpleNamespace(account_id="acct-1", equity=1000.0, cash=1000.0)
    targets: List[Any] = []

    def scenario(runner: LiveRunner) -> None:
        thread = runner._broker_dispatch_thread
        targets.extend(getattr(thread, "_args", ()))
        runner._bind_order_owner("c-1", "b-1", "alpha")
        gateway.emit("order", _order("c-1", "b-1"))
        gateway.emit("order", _order("c-x", "b-x"))
        gateway.emit("trade", _trade("t-1", "c-1", "b-1"))
        gateway.emit("execution_report", {"id": "r-x"})
        runner._queue_broker_event("account", account)

    _run(main, {}, scenario)

    # 派发线程的目标就是主策略本身(不经 fanout 包装)。
    assert targets == [main]
    assert main.order_cids == ["c-1", "c-x"]
    assert main.trade_ids == ["t-1"]
    assert main.report_ids == ["r-x"]
    assert main.portfolio_updates == [account]
    assert gateway.canceled == ["b-2"]


def test_broker_live_terminal_and_late_reports_stay_with_owner_slot(
    gateway: _FakeTraderGateway,
) -> None:
    """终态回报与迟到成交仍只到所属槽位.

    订单/执行回报进终态时 runner 会清掉活动归属表(``_close_order_mapping``);
    执行回报不经适配、迟到成交在清表之后才解析 owner, 二者都要能找回 beta,
    不能回落默认槽位。OCO 一腿以终态回报先到、成交后到, 对手腿仍要由 beta 撤。
    """

    class _OcoSlot(_Recorder):
        def on_start(self) -> None:
            self._oco_groups = {"g": {"b-1", "b-2"}}
            self._oco_order_to_group = {"b-1": "g", "b-2": "g"}

    main, beta = _Recorder(), _OcoSlot()

    def filled(cid: str, bid: str) -> Dict[str, Any]:
        return {**_order(cid, bid), "status": "Filled", "filled_quantity": 1.0}

    def scenario(runner: LiveRunner) -> None:
        # 停掉派发线程后手动 drain, 批次边界确定。
        target = runner._broker_dispatch_thread._args[0]  # type: ignore[union-attr]
        runner._stop_broker_dispatcher()
        runner._bind_order_owner("c-b", "b-b", "beta")
        runner._bind_order_owner("c-1", "b-1", "beta")
        runner._bind_order_owner("c-2", "b-2", "beta")
        # 同一批: order(Filled) → execution_report(Filled) → trade
        gateway.emit("order", filled("c-b", "b-b"))
        gateway.emit(
            "execution_report",
            {
                "id": "r-b",
                "broker_order_id": "b-b",
                "client_order_id": "c-b",
                "status": "Filled",
                "filled_quantity": 1.0,
            },
        )
        gateway.emit("trade", _trade("t-b", "c-b", "b-b"))
        # OCO 一腿: 终态回报先单独派发完, 成交在下一批迟到(只带柜台单号)。
        gateway.emit("order", filled("c-1", "b-1"))
        runner._drain_broker_events(target)
        gateway.emit("trade", _trade("t-1", "", "b-1"))
        runner._drain_broker_events(target)

    _run(main, {"beta": beta}, scenario)

    assert set(beta.order_cids) == {"c-b", "c-1"}
    assert beta.report_ids == ["r-b"]
    assert set(beta.trade_ids) == {"t-b", "t-1"}
    assert gateway.canceled == ["b-2"]
    assert main.order_cids == []
    assert main.report_ids == []
    assert main.trade_ids == []
