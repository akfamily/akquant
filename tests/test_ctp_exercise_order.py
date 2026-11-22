"""实盘期权行权指令(CTP 执行宣告): 按交易所与今昨仓拆腿、等柜台受理、拒绝原因上抛, 撤销.

拆腿规则来自 SimNow 实测: 上期所、能源中心今仓须报"平今"、昨仓报"平仓"; 大商所、
郑商所、广期所只接受"平仓"。这里用假的 CTP API 同步回调受理/拒绝。
"""

import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from akquant.gateway import ctp_native
from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.ctp_native import CTPTraderGateway

CONSTANTS = {
    "THOST_FTDC_OF_Close": "1",
    "THOST_FTDC_OF_CloseToday": "3",
    "THOST_FTDC_ACTP_Exec": "1",
    "THOST_FTDC_PD_Long": "2",
    "THOST_FTDC_HF_Speculation": "1",
    "THOST_FTDC_EOPF_UnReserve": "1",
    "THOST_FTDC_EOCF_NotToClose": "1",
    "THOST_FTDC_AF_Delete": "0",
}


def _gateway(
    monkeypatch: pytest.MonkeyPatch,
    exchange: str,
    today: float,
    yesterday: float,
    reject: Optional[Dict[int, str]] = None,
    silent: bool = False,
) -> tuple[CTPTraderGateway, List[Any]]:
    """假柜台: 第 i 条指令按 ``reject[i]`` 拒绝, 否则受理(未执行); silent 时不回报."""
    for name, value in CONSTANTS.items():
        monkeypatch.setattr(ctp_native.tdapi, name, value, raising=False)
    for field in (
        "CThostFtdcInputExecOrderField",
        "CThostFtdcQryExecOrderField",
        "CThostFtdcInputExecOrderActionField",
    ):
        monkeypatch.setattr(
            ctp_native.tdapi, field, lambda: SimpleNamespace(), raising=False
        )
    gateway = CTPTraderGateway.__new__(CTPTraderGateway)
    gateway.broker_id, gateway.user_id = "9999", "tester"
    gateway.connected = gateway.login_status = gateway.ready_to_trade = True
    gateway.front_id, gateway.session_id = 1, 7
    gateway.order_ref, gateway.req_id = 100, 0
    gateway._exercise_timeout_sec = 0.05
    gateway._exercise_waiters = {}
    gateway._exercise_cancel_waiters = {}
    gateway._exec_order_query_event = threading.Event()
    gateway.query_positions = lambda instrument_id=None: [  # type: ignore[method-assign]
        {
            "symbol": "ag2702C10300",
            "direction": "Buy",
            "available_today_quantity": today,
            "available_yesterday_quantity": yesterday,
            "exchange_id": exchange,
        }
    ]
    sent: List[Any] = []

    def insert(request: Any, req_id: int) -> int:
        sent.append(request)
        if silent:
            return 0
        message = (reject or {}).get(len(sent))
        if message is not None:
            gateway.OnRspExecOrderInsert(
                request, SimpleNamespace(ErrorID=74, ErrorMsg=message), req_id, True
            )
        else:
            gateway.OnRtnExecOrder(
                SimpleNamespace(
                    FrontID=1,
                    SessionID=7,
                    ExecOrderRef=request.ExecOrderRef,
                    InstrumentID=request.InstrumentID,
                    ExecResult="n",
                    OrderSubmitStatus="3",
                    StatusMsg="未执行",
                )
            )
        return 0

    gateway.api = SimpleNamespace(ReqExecOrderInsert=insert)
    return gateway, sent


def _legs(sent: List[Any]) -> List[tuple]:
    return [(r.ExchangeID, r.OffsetFlag, r.Volume, r.CloseFlag) for r in sent]


def test_shfe_splits_previous_and_today_positions() -> None:
    """上期所: 先昨仓"平仓"再今仓"平今", 行权后不自动平仓."""
    with pytest.MonkeyPatch.context() as mp:
        gateway, sent = _gateway(mp, "SHFE", today=1, yesterday=2)
        legs = gateway.exercise("ag2702C10300", 3)
    assert _legs(sent) == [("SHFE", "1", 2, "1"), ("SHFE", "3", 1, "1")]
    assert [(leg["offset"], leg["quantity"]) for leg in legs] == [("1", 2), ("3", 1)]
    assert all(r.PosiDirection == "2" and r.ActionType == "1" for r in sent)


def test_dce_sends_single_close_order() -> None:
    """大商所(郑商所、广期所同): 今仓也只报一条"平仓"."""
    with pytest.MonkeyPatch.context() as mp:
        gateway, sent = _gateway(mp, "DCE", today=2, yesterday=0)
        gateway.exercise("ag2702C10300", 2)
    assert _legs(sent) == [("DCE", "1", 2, "1")]


def test_exercise_reports_exact_causes() -> None:
    """超过柜台可用持仓不发指令; 柜台拒绝、部分受理、超时各自写明."""
    with pytest.MonkeyPatch.context() as mp:
        gateway, sent = _gateway(mp, "SHFE", today=1, yesterday=1)
        with pytest.raises(ValueError) as over:
            gateway.exercise("ag2702C10300", 3)
        assert sent == []
    assert str(over.value) == (
        "ag2702C10300 申请行权 3 手, 超过柜台可用多头持仓 2 手(今仓 1, 昨仓 1)"
    )

    with pytest.MonkeyPatch.context() as mp:
        gateway, _ = _gateway(
            mp, "SHFE", today=1, yesterday=1, reject={2: "CTP:执行仓位不足"}
        )
        with pytest.raises(RuntimeError) as partial:
            gateway.exercise("ag2702C10300", 2)
    assert str(partial.value) == (
        "ag2702C10300 行权指令 101(1 手)被拒: 74 CTP:执行仓位不足; 指令 100 1 手已受理"
    )

    with pytest.MonkeyPatch.context() as mp:
        gateway, _ = _gateway(mp, "DCE", today=1, yesterday=0, silent=True)
        with pytest.raises(RuntimeError) as timeout:
            gateway.exercise("ag2702C10300", 1)
    assert str(timeout.value) == (
        "ag2702C10300 行权指令 100(1 手) 0.05 秒内柜台未确认, "
        "状态未知, 请在交易终端核对"
    )
    assert gateway._exercise_waiters == {}


def test_broker_execution_routes_exercise_to_gateway() -> None:
    """实盘执行后端把行权交给柜台网关; 网关没有实现时报出网关类型."""
    calls: List[tuple] = []

    def cancel(symbol: str) -> int:
        calls.append((symbol,))
        return 2

    gateway = SimpleNamespace(
        exercise=lambda s, q: calls.append((s, q)), cancel_exercise=cancel
    )
    execution = BrokerExecution(object(), gateway, None, None)  # type: ignore[arg-type]
    execution.exercise("m2701-C-3000", 2)
    assert execution.cancel_exercise("m2701-C-3000") == 2.0
    assert calls == [("m2701-C-3000", 2), ("m2701-C-3000",)]

    with pytest.raises(RuntimeError, match="柜台网关 SimpleNamespace 没有实现"):
        BrokerExecution(object(), SimpleNamespace(), None, None).exercise(  # type: ignore[arg-type]
            "m2701-C-3000", 1
        )


def _exec_row(sys_id: str, result: str, volume: int) -> SimpleNamespace:
    return SimpleNamespace(
        InstrumentID="ag2702C10300",
        ExchangeID="SHFE",
        ExecOrderSysID=f"{sys_id:>12}",
        ExecResult=result,
        Volume=volume,
    )


def _cancel_gateway(
    mp: pytest.MonkeyPatch, reject: Optional[str] = None
) -> tuple[CTPTraderGateway, List[Any]]:
    """柜台上有两条未执行(其中一条为其他会话提交)与一条已取消的行权指令."""
    gateway, _ = _gateway(mp, "SHFE", today=0, yesterday=0)
    actions: List[Any] = []

    def query(request: Any, req_id: int) -> int:
        rows = [_exec_row("7", "n", 1), _exec_row("9", "n", 2), _exec_row("5", "c", 1)]
        for i, row in enumerate(rows):
            gateway.OnRspQryExecOrder(row, None, req_id, i == len(rows) - 1)
        return 0

    def action(request: Any, req_id: int) -> int:
        actions.append(request)
        if reject is not None and len(actions) == 2:
            gateway.OnRspExecOrderAction(
                request, SimpleNamespace(ErrorID=73, ErrorMsg=reject), req_id, True
            )
        else:
            gateway.OnRtnExecOrder(
                SimpleNamespace(
                    FrontID=9,  # 其他会话提交的指令同样按交易所编号确认撤销
                    SessionID=9,
                    ExecOrderSysID=request.ExecOrderSysID,
                    ExecResult="c",
                    StatusMsg="已取消",
                )
            )
        return 0

    gateway.api.ReqQryExecOrder = query
    gateway.api.ReqExecOrderAction = action
    return gateway, actions


def test_cancel_exercise_cancels_all_pending_orders() -> None:
    """撤销当日未执行的全部行权指令(含其他会话的), 已取消的不再撤; 返回手数."""
    with pytest.MonkeyPatch.context() as mp:
        gateway, actions = _cancel_gateway(mp)
        assert gateway.cancel_exercise("ag2702C10300") == 3.0
    assert [
        (a.ExchangeID, a.ExecOrderSysID.strip(), a.ActionFlag) for a in actions
    ] == [
        ("SHFE", "7", "0"),
        ("SHFE", "9", "0"),
    ]
    assert gateway._exercise_cancel_waiters == {}


def test_cancel_exercise_reports_rejection_and_done_part() -> None:
    """撤销被拒时写明柜台原因与已撤销的指令."""
    with pytest.MonkeyPatch.context() as mp:
        gateway, _ = _cancel_gateway(mp, reject="CTP:执行宣告撤单找不到相应执行宣告")
        with pytest.raises(RuntimeError) as exc:
            gateway.cancel_exercise("ag2702C10300")
    assert str(exc.value) == (
        "ag2702C10300 撤销行权指令 9(2 手)被拒: 73 CTP:执行宣告撤单找不到相应执行宣告; "
        "指令 7 1 手已撤销"
    )
